"""Camera placement recommendations and zone analysis."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy import ndimage
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class RecommendationZone:
    """Represents a priority zone for camera placement."""

    def __init__(
        self,
        zone_id: int,
        zone_type: str,
        avg_score: float,
        area_m2: float,
        centroid_lon: float,
        centroid_lat: float,
        recommended_cameras: int,
        priority: str,
        description: str,
        bounds_polygon: Polygon,
    ):
        """Initialize recommendation zone.

        Args:
            zone_id: Unique zone identifier
            zone_type: "pinch", "bedding", or "mixed"
            avg_score: Average score within zone
            area_m2: Zone area in square meters
            centroid_lon: Zone center longitude
            centroid_lat: Zone center latitude
            recommended_cameras: Suggested number of cameras for this zone
            priority: "high", "medium", or "low"
            description: Human-readable zone description
            bounds_polygon: Shapely polygon of zone boundary
        """
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.avg_score = avg_score
        self.area_m2 = area_m2
        self.centroid_lon = centroid_lon
        self.centroid_lat = centroid_lat
        self.recommended_cameras = recommended_cameras
        self.priority = priority
        self.description = description
        self.bounds_polygon = bounds_polygon

    def to_dict(self) -> dict:
        """Convert to dictionary for GeoJSON export."""
        return {
            "zone_id": self.zone_id,
            "type": self.zone_type,
            "avg_score": float(self.avg_score),
            "area_km2": float(self.area_m2 / 1e6),
            "centroid_lon": float(self.centroid_lon),
            "centroid_lat": float(self.centroid_lat),
            "recommended_cameras": self.recommended_cameras,
            "priority": self.priority,
            "description": self.description,
        }


class CameraPlacement:
    """Generates camera placement recommendation zones."""

    def __init__(
        self,
        pinch_score: np.ndarray,
        bedding_score: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        transform,
        crs: str,
        water_score: np.ndarray | None = None,
    ):
        """Initialize camera placement advisor.

        Args:
            pinch_score: Pinch point score array
            bedding_score: Bedding zone score array
            slope: Slope array in degrees
            aspect: Aspect array in degrees
            transform: Rasterio affine transform
            crs: Coordinate reference system
            water_score: Optional water proximity score array
        """
        self.pinch_score = pinch_score
        self.bedding_score = bedding_score
        self.slope = slope
        self.aspect = aspect
        self.transform = transform
        self.crs = crs
        self.water_score = water_score
        
        # Calculate pixel resolution in meters
        pixel_res_deg = abs(self.transform.a)
        self.pixel_res_m = pixel_res_deg * 111000  # Approximate meters per degree
        self.pixel_area_m2 = self.pixel_res_m ** 2

    def generate_corridor_zones(
        self,
        num_cameras: int = 6,
        min_zone_area_m2: float = 150000,  # ~0.15 km² minimum for corridors
        max_zones: int = 8,  # Total zones to prevent overlap
    ) -> list[RecommendationZone]:
        """Generate zones that capture high-traffic wildlife corridors.
        
        Focuses on areas with high corridor density - concentrated steep terrain
        that funnels wildlife movement into predictable travel routes.

        Args:
            num_cameras: Total number of cameras to distribute across zones
            min_zone_area_m2: Minimum zone area to consider
            max_zones: Maximum total zones to prevent overlap

        Returns:
            List of corridor recommendation zones
        """
        logger.info(f"Generating corridor zones for {num_cameras} cameras")

        # Compute corridor density from pinch scores
        # High pinch scores = steep terrain = wildlife corridors
        corridor_threshold = np.percentile(self.pinch_score[self.pinch_score > 0], 70)  # Top 30%
        high_corridors = (self.pinch_score > corridor_threshold).astype(np.float32)
        
        # Calculate corridor density in 30x30 window (~300m radius)
        corridor_density = ndimage.uniform_filter(high_corridors, size=30, mode='constant')
        
        # Normalize corridor density
        corridor_density_norm = corridor_density / (corridor_density.max() + 1e-6)
        
        logger.info(f"Corridor density: max={corridor_density.max():.3f}, mean={corridor_density.mean():.3f}")
        
        # High priority: Top 10% corridor density (90th percentile)
        high_threshold = np.percentile(corridor_density_norm[corridor_density_norm > 0], 90)
        high_zones = self._identify_zones(
            corridor_density_norm,
            "corridor",
            high_threshold,
            0.0,  # No medium threshold - only create high priority
            min_zone_area_m2,
        )
        
        # Medium priority: 75th-90th percentile
        med_threshold = np.percentile(corridor_density_norm[corridor_density_norm > 0], 75)
        med_zones = self._identify_zones(
            corridor_density_norm,
            "corridor",
            med_threshold,
            0.0,
            min_zone_area_m2,
        )
        
        # Mark mediums properly
        for zone in med_zones:
            if zone.avg_score < high_threshold:
                zone.priority = "medium"
        
        logger.info(f"Corridor thresholds: high={high_threshold:.3f}, med={med_threshold:.3f}")
        
        # Combine and sort all zones by score
        zones = high_zones + med_zones
        zones.sort(key=lambda z: z.avg_score, reverse=True)
        zones = zones[:max_zones]  # Keep top N zones
        
        # Sort by priority then score
        priority_order = {"high": 0, "medium": 1, "low": 2}
        zones.sort(key=lambda z: (priority_order[z.priority], -z.avg_score))

        # Allocate cameras to zones
        zones = self._allocate_cameras_to_zones(zones, num_cameras)

        logger.info(f"Generated {len(zones)} corridor zones")
        for zone in zones:
            logger.info(
                f"  Zone {zone.zone_id} (corridor, {zone.priority}): "
                f"{zone.recommended_cameras} cameras, {zone.area_m2/1e6:.3f} km², "
                f"density={zone.avg_score:.3f}"
            )

        return zones

    def _identify_zones(
        self,
        score: np.ndarray,
        zone_type: str,
        high_threshold: float,
        med_threshold: float,
        min_area_m2: float,
    ) -> list[RecommendationZone]:
        """Identify contiguous high-scoring zones.

        Args:
            score: Score array to analyze
            zone_type: "pinch" or "bedding"
            high_threshold: Score threshold for high-priority zones
            med_threshold: Score threshold for medium-priority zones
            min_area_m2: Minimum zone area

        Returns:
            List of RecommendationZone objects
        """
        zones = []
        min_area_pixels = max(1, int(min_area_m2 / self.pixel_area_m2))

        # Process high-priority zones first, then medium
        for priority, threshold in [("high", high_threshold), ("medium", med_threshold)]:
            # Create binary mask of pixels above threshold
            mask = score >= threshold

            # Label connected regions
            labeled, num_features = ndimage.label(mask)

            for region_id in range(1, num_features + 1):
                region_mask = labeled == region_id
                region_size = np.sum(region_mask)

                # Skip if too small
                if region_size < min_area_pixels:
                    continue

                # Calculate zone statistics
                region_scores = score[region_mask]
                avg_score = float(np.mean(region_scores))
                area_m2 = region_size * self.pixel_area_m2

                # Find centroid
                rows, cols = np.where(region_mask)
                centroid_row = int(np.mean(rows))
                centroid_col = int(np.mean(cols))
                centroid_lon, centroid_lat = self._pixel_to_coords(centroid_row, centroid_col)

                # Generate boundary polygon (simplified convex hull)
                boundary_points = self._get_zone_boundary(region_mask)

                # Generate description
                description = self._generate_zone_description(
                    zone_type, priority, avg_score, area_m2
                )

                zone = RecommendationZone(
                    zone_id=len(zones) + 1,
                    zone_type=zone_type,
                    avg_score=avg_score,
                    area_m2=area_m2,
                    centroid_lon=centroid_lon,
                    centroid_lat=centroid_lat,
                    recommended_cameras=0,  # Will be allocated later
                    priority=priority,
                    description=description,
                    bounds_polygon=boundary_points,
                )
                zones.append(zone)

        return zones

    def _get_zone_boundary(self, region_mask: np.ndarray) -> Polygon:
        """Extract boundary polygon from region mask.

        Args:
            region_mask: Boolean mask of region

        Returns:
            Shapely Polygon of zone boundary
        """
        # Find boundary pixels
        rows, cols = np.where(region_mask)

        if len(rows) < 3:
            # Too few points for polygon, create a small circle
            if len(rows) > 0:
                lon, lat = self._pixel_to_coords(rows[0], cols[0])
                # Create small square around point
                offset = self.pixel_res_m * 0.00001  # ~1m in degrees
                return Polygon([
                    (lon - offset, lat - offset),
                    (lon + offset, lat - offset),
                    (lon + offset, lat + offset),
                    (lon - offset, lat + offset),
                ])
            else:
                return Polygon()

        # Sample boundary points (every Nth pixel to avoid huge polygons)
        sample_every = max(1, len(rows) // 50)  # Max 50 points
        sampled_rows = rows[::sample_every]
        sampled_cols = cols[::sample_every]

        # Convert to coordinates
        coords = [self._pixel_to_coords(r, c) for r, c in zip(sampled_rows, sampled_cols)]

        # Create convex hull
        try:
            from shapely.ops import unary_union
            points = [Point(lon, lat) for lon, lat in coords]
            hull = unary_union(points).convex_hull
            if isinstance(hull, Polygon):
                return hull
            else:
                # Fallback to simple polygon
                return Polygon(coords)
        except:
            # If convex hull fails, use simple polygon
            if len(coords) >= 3:
                return Polygon(coords)
            else:
                # Fallback to small square
                lon, lat = coords[0] if coords else (0, 0)
                offset = self.pixel_res_m * 0.00001
                return Polygon([
                    (lon - offset, lat - offset),
                    (lon + offset, lat - offset),
                    (lon + offset, lat + offset),
                    (lon - offset, lat + offset),
                ])

    def _allocate_cameras_to_zones(
        self,
        zones: list[RecommendationZone],
        total_cameras: int,
    ) -> list[RecommendationZone]:
        """Allocate camera recommendations across zones.

        Args:
            zones: List of zones (sorted by priority/score)
            total_cameras: Total cameras to distribute

        Returns:
            Zones with updated recommended_cameras
        """
        if not zones:
            return zones

        # Calculate total "weight" for each zone (score * area)
        total_weight = sum(z.avg_score * z.area_m2 for z in zones)

        cameras_allocated = 0
        for zone in zones:
            if cameras_allocated >= total_cameras:
                zone.recommended_cameras = 0
                continue

            # Allocate based on zone weight
            zone_weight = zone.avg_score * zone.area_m2
            zone_share = zone_weight / total_weight if total_weight > 0 else 0

            # Round to nearest integer, minimum 1 for high-priority zones
            cameras_for_zone = round(zone_share * total_cameras)
            if zone.priority == "high" and cameras_for_zone == 0 and cameras_allocated < total_cameras:
                cameras_for_zone = 1

            # Cap at remaining cameras
            cameras_for_zone = min(cameras_for_zone, total_cameras - cameras_allocated)

            zone.recommended_cameras = cameras_for_zone
            cameras_allocated += cameras_for_zone

        # If we haven't allocated all cameras, give extras to top zones
        remaining = total_cameras - cameras_allocated
        for zone in zones:
            if remaining <= 0:
                break
            if zone.priority == "high":
                zone.recommended_cameras += 1
                remaining -= 1

        return zones

    def _generate_zone_description(
        self,
        zone_type: str,
        priority: str,
        avg_score: float,
        area_m2: float,
    ) -> str:
        """Generate human-readable zone description.

        Args:
            zone_type: "corridor" or other
            priority: "high", "medium", or "low"
            avg_score: Average zone score (corridor density)
            area_m2: Zone area

        Returns:
            Description string
        """
        area_km2 = area_m2 / 1e6

        if zone_type == "corridor":
            if priority == "high":
                return f"High-density corridor zone ({area_km2:.2f} km²). Heavy concentration of wildlife travel routes. Ideal for trail camera placement."
            else:
                return f"Moderate corridor zone ({area_km2:.2f} km²). Secondary movement area with good detection potential."
        else:
            # Fallback for other types
            return f"{priority.capitalize()} priority zone ({area_km2:.2f} km²)"

    def _pixel_to_coords(self, row: int, col: int) -> tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates.

        Args:
            row: Row index
            col: Column index

        Returns:
            (lon, lat) tuple
        """
        x, y = self.transform * (col + 0.5, row + 0.5)  # Center of pixel
        return float(x), float(y)

    def export_zones_geojson(self, zones: list[RecommendationZone], output_path: Path):
        """Export recommendation zones to GeoJSON.

        Args:
            zones: List of RecommendationZone objects
            output_path: Output GeoJSON path
        """
        # Create GeoDataFrame with zone polygons
        geometries = [zone.bounds_polygon for zone in zones]
        properties = [zone.to_dict() for zone in zones]

        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Exported {len(zones)} recommendation zones to {output_path}")

