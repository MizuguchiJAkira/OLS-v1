"""Main orchestration logic for land audit."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping

from land_audit.cameras import CameraPlacement
from land_audit.dem import DEMManager
from land_audit.report import ReportGenerator
from land_audit.scoring import HeuristicScorer
from land_audit.terrain import TerrainAnalyzer

logger = logging.getLogger(__name__)


class LandAudit:
    """Main land audit orchestrator."""

    def __init__(self, output_dir: Path, cache_dir: Path | None = None):
        """Initialize land audit.

        Args:
            output_dir: Output directory for reports
            cache_dir: Cache directory for DEM files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)

        self.dem_manager = DEMManager(cache_dir)
        self.report_generator = ReportGenerator(self.output_dir)

    def run_point_audit(
        self,
        lat: float,
        lon: float,
        radius_m: float = 2000,
        cameras_n: int = 6,
        dem_resolution_m: int = 10,
    ) -> Path:
        """Run audit for a point location with radius.

        Args:
            lat: Latitude
            lon: Longitude
            radius_m: Radius in meters
            cameras_n: Number of cameras to place
            dem_resolution_m: DEM resolution in meters

        Returns:
            Path to generated report PDF
        """
        logger.info(f"Running point audit: ({lat}, {lon}) with {radius_m}m radius")

        # Convert radius to degrees (approximate)
        radius_deg = radius_m / 111000  # 1 degree ~ 111km

        # Create bounding box
        bounds = (
            lon - radius_deg,
            lat - radius_deg,
            lon + radius_deg,
            lat + radius_deg,
        )

        # Create AOI polygon
        aoi_geom = box(*bounds)
        aoi_gdf = gpd.GeoDataFrame([{"name": "AOI"}], geometry=[aoi_geom], crs="EPSG:4326")

        return self._run_audit(bounds, aoi_gdf, cameras_n, dem_resolution_m, lat=lat, lon=lon, radius_m=radius_m)

    def run_polygon_audit(
        self,
        geojson_path: Path,
        cameras_n: int = 6,
        dem_resolution_m: int = 10,
    ) -> Path:
        """Run audit for a polygon boundary.

        Args:
            geojson_path: Path to GeoJSON file with polygon
            cameras_n: Number of cameras to place
            dem_resolution_m: DEM resolution in meters

        Returns:
            Path to generated report PDF
        """
        logger.info(f"Running polygon audit: {geojson_path}")

        # Read GeoJSON
        aoi_gdf = gpd.read_file(geojson_path)

        if len(aoi_gdf) == 0:
            raise ValueError("GeoJSON file contains no features")

        # Ensure EPSG:4326
        if aoi_gdf.crs != "EPSG:4326":
            aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

        # Get bounds
        bounds = aoi_gdf.total_bounds  # (minx, miny, maxx, maxy)
        
        # Calculate center lat/lon and radius for OSM query
        minx, miny, maxx, maxy = bounds
        lat = (miny + maxy) / 2
        lon = (minx + maxx) / 2
        radius_m = max((maxx - minx), (maxy - miny)) * 111000 / 2  # Convert degrees to meters

        return self._run_audit(bounds, aoi_gdf, cameras_n, dem_resolution_m, lat=lat, lon=lon, radius_m=radius_m)

    def _run_audit(
        self,
        bounds: tuple[float, float, float, float],
        aoi_gdf: gpd.GeoDataFrame,
        cameras_n: int,
        dem_resolution_m: int,
        lat: float = None,
        lon: float = None,
        radius_m: float = None,
    ) -> Path:
        """Internal method to run complete audit.

        Args:
            bounds: Bounding box (minx, miny, maxx, maxy)
            aoi_gdf: GeoDataFrame with AOI boundary
            cameras_n: Number of cameras
            dem_resolution_m: DEM resolution
            lat: Optional center latitude for OSM queries
            lon: Optional center longitude for OSM queries
            radius_m: Optional radius for OSM queries

        Returns:
            Path to report PDF
        """
        # Step 1: Fetch DEM
        logger.info("Step 1/6: Fetching DEM data")
        dem_path = self.dem_manager.fetch_dem(bounds, resolution=dem_resolution_m)
        dem_info = self.dem_manager.get_dem_info(dem_path)

        # Copy DEM to debug folder
        debug_dem = self.debug_dir / "clipped_dem.tif"
        import shutil

        shutil.copy(dem_path, debug_dem)

        # Step 2: Compute terrain derivatives
        logger.info("Step 2/7: Computing terrain derivatives")
        terrain = TerrainAnalyzer(dem_path)

        slope = terrain.compute_slope(self.debug_dir / "slope.tif")
        aspect = terrain.compute_aspect(self.debug_dir / "aspect.tif")
        terrain.compute_hillshade(output_path=self.debug_dir / "hillshade.tif")
        tpi = terrain.compute_tpi(window_size=7)
        ruggedness = terrain.compute_ruggedness(window_size=5)

        # Step 3: Analyze hydrology (water features)
        logger.info("Step 3/7: Analyzing water features")
        from land_audit.hydrology import HydrologyAnalysis
        import numpy as np

        # Fetch real water bodies from OpenStreetMap
        hydro = HydrologyAnalysis(
            elevation=terrain.elevation,
            transform=terrain.transform,
            crs=terrain.crs,
            historical_rainfall_mm=1000.0,
        )

        # Get real water bodies from OpenStreetMap if location provided
        if lat is not None and lon is not None and radius_m is not None:
            logger.info("Fetching water bodies from OpenStreetMap")
            streams = hydro.fetch_osm_water_bodies(
                lat=lat,
                lon=lon,
                radius_m=radius_m,
            )
        else:
            # Calculate from bounds if not provided
            minx, miny, maxx, maxy = bounds
            lat = (miny + maxy) / 2
            lon = (minx + maxx) / 2
            radius_m = max((maxx - minx), (maxy - miny)) * 111000 / 2
            logger.info("Fetching water bodies from OpenStreetMap")
            streams = hydro.fetch_osm_water_bodies(
                lat=lat,
                lon=lon,
                radius_m=radius_m,
            )
        
        if np.sum(streams) == 0:
            logger.warning("No water bodies found in OpenStreetMap, falling back to terrain analysis")
            # Fallback to basic flow analysis if no OSM data
            flow_acc = hydro.compute_flow_accumulation()
            streams = hydro.identify_streams(flow_acc, threshold_percentile=98.0)
            
        water_distance = hydro.compute_water_proximity(streams, max_distance_m=200.0)

        hydro.save_hydrology_outputs(
            output_dir=self.debug_dir,
            flow_acc=streams.astype(np.float32),  # Save stream mask
            streams=streams,
            water_distance=water_distance,
        )

        # Step 4: Compute heuristic scores
        logger.info("Step 4/7: Computing heuristic scores")
        scorer = HeuristicScorer(
            slope=slope,
            aspect=aspect,
            tpi=tpi,
            ruggedness=ruggedness,
            transform=terrain.transform,
            crs=terrain.crs,
        )

        bedding_score = scorer.compute_bedding_score(self.debug_dir / "bedding_score.tif")
        pinch_score = scorer.compute_pinch_score(
            elevation=terrain.elevation,
            resolution=terrain.resolution,
            output_path=self.debug_dir / "pinch_score.tif",
        )

        # Compute water proximity score
        water_score = scorer.compute_water_score(
            water_distance=water_distance,
            output_path=self.debug_dir / "water_score.tif",
        )

        # Step 5: Generate camera recommendation zones
        logger.info("Step 5/7: Generating camera recommendation zones")
        camera_placer = CameraPlacement(
            pinch_score=pinch_score,
            bedding_score=bedding_score,
            slope=slope,
            aspect=aspect,
            transform=terrain.transform,
            crs=str(terrain.crs),
            water_score=water_score,  # Add water score
        )

        zones = camera_placer.generate_corridor_zones(num_cameras=cameras_n)

        # Step 6: Generate GeoJSON output
        logger.info("Step 6/7: Generating GeoJSON outputs")
        self._export_geojson(
            zones=zones,
            aoi_gdf=aoi_gdf,
            pinch_score=pinch_score,
            bedding_score=bedding_score,
            transform=terrain.transform,
            crs=terrain.crs,
        )

        # Step 6: Generate PDF report
        logger.info("Step 7/7: Generating PDF report")
        report_path = self.report_generator.generate_report(
            zones=zones,
            aoi_boundary=aoi_gdf,
            hillshade_path=self.debug_dir / "hillshade.tif",
            pinch_score_path=self.debug_dir / "pinch_score.tif",
            bedding_score_path=self.debug_dir / "bedding_score.tif",
            dem_info=dem_info,
            streams_path=self.debug_dir / "streams.tif",  # Add streams
        )

        logger.info(f"Audit complete! Report: {report_path}")
        return report_path

    def _export_geojson(
        self,
        zones,
        aoi_gdf,
        pinch_score,
        bedding_score,
        transform,
        crs,
    ):
        """Export all layers to GeoJSON.

        Args:
            zones: List of RecommendationZone objects
            aoi_gdf: AOI boundary GeoDataFrame
            pinch_score: Pinch score array
            bedding_score: Bedding score array
            transform: Rasterio transform
            crs: Coordinate reference system
        """
        output_path = self.output_dir / "audit.geojson"

        # Create feature collection
        features = []

        # Add AOI boundary
        for _, row in aoi_gdf.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "properties": {"layer": "aoi_boundary", "name": "Area of Interest"},
                    "geometry": mapping(row.geometry),
                }
            )

        # Add recommendation zones
        for zone in zones:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "layer": "recommendation_zones",
                        "zone_id": zone.zone_id,
                        **zone.to_dict(),
                    },
                    "geometry": mapping(zone.bounds_polygon),
                }
            )

        # Add score summary polygons (simplified - just overall stats)
        # In a more complete version, we'd create vector polygons from high-score regions
        pinch_stats = {
            "layer": "pinch_score_summary",
            "mean": float(np.mean(pinch_score)),
            "max": float(np.max(pinch_score)),
            "std": float(np.std(pinch_score)),
        }

        bedding_stats = {
            "layer": "bedding_score_summary",
            "mean": float(np.mean(bedding_score)),
            "max": float(np.max(bedding_score)),
            "std": float(np.std(bedding_score)),
        }

        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": features,
            "metadata": {
                "pinch_score_stats": pinch_stats,
                "bedding_score_stats": bedding_stats,
            },
        }

        # Write to file
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Exported GeoJSON to {output_path}")
