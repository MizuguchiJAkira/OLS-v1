"""PDF report generation."""

import logging
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from land_audit.cameras import RecommendationZone

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Corridor clustering will be skipped.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. 3D visualizations will be skipped.")


class ReportGenerator:
    """Generates PDF reports for land audits."""

    def __init__(self, output_dir: Path):
        """Initialize report generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        zones: list[RecommendationZone],
        aoi_boundary: gpd.GeoDataFrame,
        hillshade_path: Path,
        pinch_score_path: Path,
        bedding_score_path: Path,
        dem_info: dict,
        streams_path: Path | None = None,
    ) -> Path:
        """Generate complete PDF report.

        Args:
            zones: List of recommendation zones for camera placement
            aoi_boundary: GeoDataFrame with AOI boundary
            hillshade_path: Path to hillshade raster
            pinch_score_path: Path to pinch score raster
            bedding_score_path: Path to bedding score raster
            dem_info: DEM metadata dictionary
            streams_path: Optional path to streams raster

        Returns:
            Path to generated PDF
        """
        output_path = self.output_dir / "report.pdf"
        logger.info(f"Generating report: {output_path}")

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )

        # Build content
        story = []
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=24,
            textColor=colors.HexColor("#2C3E50"),
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=colors.HexColor("#34495E"),
            spaceAfter=12,
        )
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
        )
        heading3_style = ParagraphStyle(
            "CustomHeading3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
        )

        # Title
        story.append(Paragraph("Orome Land Audit Report", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        total_cameras = sum(z.recommended_cameras for z in zones)
        high_priority_zones = len([z for z in zones if z.priority == "high"])
        
        summary_text = f"""
        This report presents the results of a terrain-based land audit for wildlife management.
        Analysis covers an area of interest with bounds {self._format_bounds(dem_info['bounds'])}.
        Using high-resolution elevation data ({dem_info['resolution'][0]:.1f}m resolution),
        we identified {len(zones)} priority zones for camera placement. These zones represent high-value areas
        where {total_cameras} trail cameras should be deployed based on travel corridors, bedding habitat, and terrain characteristics.
        Rather than prescribing exact coordinates, this advisory approach allows for on-the-ground judgment
        that accounts for factors like vegetation, property boundaries, and field conditions.
        """
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.3 * inch))

        # Note about 3D visualization
        story.append(Paragraph("Interactive 3D Visualization", heading_style))
        viz_note = """
        An interactive 3D terrain visualization has been generated as <b>terrain_3d.html</b>.
        Open this file in a web browser to explore the terrain with heat maps showing pinch points (red) and bedding areas (green).
        Recommendation zones are displayed with boundaries and camera counts. You can rotate, zoom, and toggle different terrain features.
        """
        story.append(Paragraph(viz_note, body_style))
        story.append(Spacer(1, 0.3 * inch))

        # Zone Recommendations Table
        story.append(PageBreak())
        story.append(Paragraph("Priority Zones for Camera Placement", heading_style))
        
        placement_note = f"""
        Instead of placing cameras at exact coordinates, we identify {high_priority_zones} high-priority zones
        and {len(zones) - high_priority_zones} secondary zones based on terrain analysis. Each zone shows
        the recommended number of cameras and area covered. This approach allows you to:<br/><br/>
        • Account for ground conditions not visible in terrain data (vegetation density, property lines, access roads)<br/>
        • Adapt placement based on historical sightings and hunter knowledge<br/>
        • Adjust for wind direction, sun exposure, and seasonal factors<br/>
        • Make informed decisions while maintaining strategic coverage
        """
        story.append(Paragraph(placement_note, body_style))
        story.append(Spacer(1, 0.2 * inch))

        zone_table = self._create_zone_table(zones)
        story.append(zone_table)
        story.append(Spacer(1, 0.3 * inch))

        # Methodology
        story.append(Paragraph("Methodology Notes", heading_style))
        methodology_text = """
        <b>Zone-Based Recommendations:</b> Rather than prescribing exact camera coordinates, the system identifies
        priority zones where cameras should be concentrated. High-priority zones (★★★) receive the most cameras,
        while medium-priority zones (★★) provide secondary coverage opportunities.<br/><br/>
        <b>Slope-Based Scoring:</b> Terrain is analyzed using elevation derivatives to identify
        optimal camera placement zones. Steep slopes (20-35°) indicate travel corridors (pinch points)
        where movement is funneled. Moderate slopes (8-15°) indicate bedding zones with good drainage
        and visibility. Even slight elevation changes (2-8°) in otherwise flat terrain are prioritized.<br/><br/>
        <b>Heat Map Visualization:</b> The 3D terrain map shows continuous heat maps of pinch point scores (red gradient) 
        and bedding area scores (green gradient). Darker/brighter colors indicate higher-value terrain for each camera type.
        """
        story.append(Paragraph(methodology_text, body_style))

        # Build PDF
        doc.build(story)
        logger.info(f"Report generated: {output_path}")
        
        # Generate 3D interactive visualization
        if PLOTLY_AVAILABLE:
            try:
                viz_path = self._create_3d_visualization(
                    hillshade_path.parent / "clipped_dem.tif",
                    zones,
                    pinch_score_path,
                    bedding_score_path,
                    aoi_boundary,
                    streams_path,
                )
                logger.info(f"3D visualization generated: {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to create 3D visualization: {e}")

        return output_path

    def _create_3d_visualization(
        self,
        dem_path: Path,
        zones: list[RecommendationZone],
        pinch_score_path: Path,
        bedding_score_path: Path,
        aoi_boundary: gpd.GeoDataFrame,
        streams_path: Path | None = None,
    ) -> Path:
        """Create interactive 3D terrain visualization with heat maps and recommendation zones.
        
        Args:
            dem_path: Path to DEM raster
            zones: List of recommendation zones
            pinch_score_path: Path to pinch score raster
            bedding_score_path: Path to bedding score raster
            aoi_boundary: GeoDataFrame with AOI boundary
            streams_path: Optional path to streams raster
            
        Returns:
            Path to generated HTML file
        """
        output_path = self.output_dir / "terrain_3d.html"
        
        # Load original DEM data for camera placement reference
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            bounds = src.bounds
        
        # Expand bounds by 30% in each direction for better context
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom
        buffer_x = width * 0.3
        buffer_y = height * 0.3
        
        expanded_bounds = (
            bounds.left - buffer_x,
            bounds.bottom - buffer_y,
            bounds.right + buffer_x,
            bounds.top + buffer_y
        )
        
        # Fetch larger DEM for visualization context
        try:
            from py3dep import get_map
            logger.info(f"Fetching expanded DEM: {expanded_bounds}")
            expanded_dem_data = get_map(
                "DEM",
                expanded_bounds,
                resolution=10,
                geo_crs="EPSG:4326",
                crs="EPSG:4326"
            )
            vis_dem = expanded_dem_data.values
            vis_bounds = expanded_dem_data.rio.bounds()
            logger.info(f"Expanded DEM shape: {vis_dem.shape}, bounds: {vis_bounds}")
        except Exception as e:
            logger.warning(f"Could not fetch expanded DEM, using original: {e}")
            vis_dem = dem
            vis_bounds = bounds
            
        # Load score data for coloring and resize to match expanded DEM
        with rasterio.open(pinch_score_path) as src:
            pinch_scores = src.read(1)
        with rasterio.open(bedding_score_path) as src:
            bedding_scores = src.read(1)
            
        # Resize scores to match visualization DEM if needed
        if vis_dem.shape != pinch_scores.shape:
            from scipy.ndimage import zoom
            zoom_factors = (vis_dem.shape[0] / pinch_scores.shape[0], 
                          vis_dem.shape[1] / pinch_scores.shape[1])
            pinch_scores = zoom(pinch_scores, zoom_factors, order=1)
            bedding_scores = zoom(bedding_scores, zoom_factors, order=1)
            logger.info(f"Resized scores from original to expanded: {zoom_factors}")
        
        # Subsample for performance (every 2nd point)
        dem_sub = vis_dem[::2, ::2]
        pinch_sub = pinch_scores[::2, ::2]
        bedding_sub = bedding_scores[::2, ::2]
        
        # Create coordinate grids for surface
        rows, cols = dem_sub.shape
        x_coords = np.linspace(vis_bounds[0], vis_bounds[2], cols)
        y_coords = np.linspace(vis_bounds[1], vis_bounds[3], rows)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Get elevation using bilinear interpolation to match surface exactly
        def get_elevation(cam):
            # Find position in the coordinate grid
            x_idx = np.interp(cam.lon, x_coords, np.arange(len(x_coords)))
            y_idx = np.interp(cam.lat, y_coords, np.arange(len(y_coords)))
            
            # Bilinear interpolation
            x0, x1 = int(np.floor(x_idx)), int(np.ceil(x_idx))
            y0, y1 = int(np.floor(y_idx)), int(np.ceil(y_idx))
            
            # Clamp to valid range
            x0 = max(0, min(x0, dem_sub.shape[1] - 1))
            x1 = max(0, min(x1, dem_sub.shape[1] - 1))
            y0 = max(0, min(y0, dem_sub.shape[0] - 1))
            y1 = max(0, min(y1, dem_sub.shape[0] - 1))
            
            # Get the four corner elevations
            if x0 == x1 and y0 == y1:
                elev = float(dem_sub[y0, x0])
            elif x0 == x1:
                wy = y_idx - y0
                elev = float(dem_sub[y0, x0] * (1 - wy) + dem_sub[y1, x0] * wy)
            elif y0 == y1:
                wx = x_idx - x0
                elev = float(dem_sub[y0, x0] * (1 - wx) + dem_sub[y0, x1] * wx)
            else:
                wx = x_idx - x0
                wy = y_idx - y0
                elev = float(
                    dem_sub[y0, x0] * (1 - wx) * (1 - wy) +
                    dem_sub[y0, x1] * wx * (1 - wy) +
                    dem_sub[y1, x0] * (1 - wx) * wy +
                    dem_sub[y1, x1] * wx * wy
                )
            
            logger.info(f"Camera at ({cam.lon:.5f}, {cam.lat:.5f}) -> grid ({x_idx:.1f}, {y_idx:.1f}) -> elev {elev:.1f}m")
            return elev
        
        # Load water mask and integrate into terrain coloring
        water_mask = np.zeros_like(dem_sub, dtype=bool)
        if streams_path and streams_path.exists():
            try:
                with rasterio.open(streams_path) as src:
                    streams_original = src.read(1)
                
                from PIL import Image
                water_mask_resized = np.array(Image.fromarray(streams_original.astype(np.float32)).resize(
                    (dem_sub.shape[1], dem_sub.shape[0]), Image.NEAREST
                ))
                water_mask = water_mask_resized > 0
                logger.info(f"Loaded water mask: {np.sum(water_mask)} water pixels")
            except Exception as e:
                logger.warning(f"Failed to load water mask: {e}")
        
        # Create custom surface coloring: blue for water, elevation for land
        surface_colors = np.where(water_mask, dem_sub.min() - 100, dem_sub)
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add elevation-colored terrain surface (visible by default)
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=dem_sub,
            surfacecolor=surface_colors,
            colorscale=[
                [0.0, '#0066cc'],   # Deep blue for water
                [0.05, '#0066cc'],  # Keep blue for water range
                [0.051, '#1B5E20'], # Dark green (low valleys)
                [0.2, '#2E7D32'],   # Medium green
                [0.3, '#66BB6A'],   # Light green
                [0.4, '#9CCC65'],   # Yellow-green
                [0.5, '#FFD54F'],   # Yellow (mid elevation)
                [0.6, '#D4A574'],   # Tan
                [0.7, '#A1887F'],   # Brown
                [0.8, '#8D6E63'],   # Dark brown
                [0.9, '#BDBDBD'],   # Gray (high elevation)
                [1.0, '#E0E0E0'],   # Pale gray/white (peaks)
            ],
            name='Elevation',
            showscale=True,
            colorbar=dict(title="Elevation (m)", x=1.15, len=0.9, y=0.5),
            hovertemplate='Lon: %{x:.5f}<br>Lat: %{y:.5f}<br>Elevation: %{z:.1f}m<extra></extra>',
            visible=True,
        ))
        
        # Add satellite-style terrain surface (hidden by default)
        # Create grayscale hillshade-like appearance for satellite effect
        hillshade_colors = np.where(water_mask, dem_sub.min() - 100, dem_sub)
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=dem_sub,
            surfacecolor=hillshade_colors,
            colorscale=[
                [0.0, '#0047AB'],   # Deep blue for water
                [0.05, '#0066cc'], # Blue for water
                [0.051, '#4A4A4A'], # Dark gray (satellite-like)
                [0.3, '#6B6B6B'],   # Medium dark gray
                [0.5, '#8A8A8A'],   # Medium gray
                [0.7, '#A8A8A8'],   # Light gray
                [0.9, '#C8C8C8'],   # Very light gray
                [1.0, '#E8E8E8'],   # Near white
            ],
            name='Satellite',
            showscale=True,
            colorbar=dict(title="Elevation (m)", x=1.15, len=0.9, y=0.5),
            hovertemplate='Lon: %{x:.5f}<br>Lat: %{y:.5f}<br>Elevation: %{z:.1f}m<extra></extra>',
            visible=False,
        ))
        
        # Track trace indices for features
        trace_indices = {
            'terrain_elevation': 0,
            'terrain_satellite': 1,
            'water_features': None,
            'corridors': None,
            'boundary': None,
        }
        
        # Skip zones - removed per user request
        
        # Add terrain feature annotations
        # Show high-score corridors for pinch points
        with rasterio.open(pinch_score_path) as src:
            pinch_scores_original = src.read(1)
        
        # Resize to expanded grid
        pinch_scores_expanded = np.array(Image.fromarray(pinch_scores_original).resize(
            (dem_sub.shape[1], dem_sub.shape[0]), Image.BILINEAR
        ))
        
        # Find high-score corridor zones (top 5% of pinch scores)
        high_corridor_threshold = np.percentile(pinch_scores_expanded[pinch_scores_expanded > 0], 95)
        corridor_mask = pinch_scores_expanded >= high_corridor_threshold
        
        # Cluster corridor points into movement channels
        if np.any(corridor_mask) and SKLEARN_AVAILABLE:
            # Get all corridor points
            corridor_points = np.argwhere(corridor_mask)
            if len(corridor_points) > 0:
                # Convert to lon/lat coordinates
                corridor_lons = x_coords[corridor_points[:, 1]]
                corridor_lats = y_coords[corridor_points[:, 0]]
                corridor_elevs = dem_sub[corridor_points[:, 0], corridor_points[:, 1]]
                
                # Filter out water points to prevent trails from encroaching on water
                if np.any(water_mask):
                    # Check which corridor points are on water
                    water_at_corridors = water_mask[corridor_points[:, 0], corridor_points[:, 1]]
                    non_water_mask = ~water_at_corridors
                    
                    # Keep only non-water points
                    corridor_lons = corridor_lons[non_water_mask]
                    corridor_lats = corridor_lats[non_water_mask]
                    corridor_elevs = corridor_elevs[non_water_mask]
                    
                    logger.info(f"Filtered {np.sum(water_at_corridors)} water points from corridors")
                
                if len(corridor_lons) > 0:
                    # Cluster using DBSCAN to identify distinct movement channels
                    coords_2d = np.column_stack([corridor_lons, corridor_lats])
                    clustering = DBSCAN(eps=0.002, min_samples=10).fit(coords_2d)  # ~200m clusters
                    labels = clustering.labels_
                    
                    # Get unique clusters (exclude noise labeled as -1)
                    unique_labels = set(labels)
                    unique_labels.discard(-1)
                    
                    logger.info(f"Identified {len(unique_labels)} movement channels from {len(corridor_lons)} corridor points")
                    
                    # Draw each cluster as a line (movement channel)
                    trace_indices['corridors'] = []
                    for label in sorted(unique_labels):
                        cluster_mask = labels == label
                        cluster_lons = corridor_lons[cluster_mask]
                        cluster_lats = corridor_lats[cluster_mask]
                        cluster_elevs = corridor_elevs[cluster_mask]
                        
                        # Sort points to form a connected path (simple nearest neighbor)
                        if len(cluster_lons) > 1:
                            # Find path through cluster points
                            remaining = set(range(len(cluster_lons)))
                            path = [remaining.pop()]  # Start with first point
                            
                            while remaining:
                                last_idx = path[-1]
                                # Find nearest unvisited point
                                distances = (
                                    (cluster_lons[list(remaining)] - cluster_lons[last_idx])**2 +
                                    (cluster_lats[list(remaining)] - cluster_lats[last_idx])**2
                                )
                                nearest_idx = list(remaining)[np.argmin(distances)]
                                path.append(nearest_idx)
                                remaining.remove(nearest_idx)
                        
                            # Create ordered line
                            path_lons = cluster_lons[path]
                            path_lats = cluster_lats[path]
                            path_elevs = cluster_elevs[path] + 3  # Elevate above terrain
                            
                            trace_indices['corridors'].append(len(fig.data))
                            fig.add_trace(go.Scatter3d(
                                x=path_lons,
                                y=path_lats,
                                z=path_elevs,
                                mode='lines+markers',
                                line=dict(width=6, color='#DC143C'),  # Crimson red lines
                                marker=dict(size=2, color='#DC143C'),
                                name=f'Movement Channel {label + 1}',
                                showlegend=bool(label == min(unique_labels)),  # Show only one legend entry
                                legendgroup='channels',
                                hovertemplate=f'Channel {label + 1}<br>Predicted Trail<extra></extra>',
                            ))
        elif np.any(corridor_mask):
            # Fallback to dot cloud if sklearn not available
            corridor_points = np.argwhere(corridor_mask)[::10]
            if len(corridor_points) > 0:
                corridor_lons = x_coords[corridor_points[:, 1]]
                corridor_lats = y_coords[corridor_points[:, 0]]
                corridor_elevs = dem_sub[corridor_points[:, 0], corridor_points[:, 1]]
                
                trace_indices['corridors'] = len(fig.data)
                fig.add_trace(go.Scatter3d(
                    x=corridor_lons,
                    y=corridor_lats,
                    z=corridor_elevs + 2,
                    mode='markers',
                    marker=dict(size=3, color='#ff1493', symbol='circle', opacity=0.8),
                    name='Predicted Movement Channels',
                    showlegend=True,
                    hovertemplate='Corridor Zone<br>Score: >%.2f<extra></extra>' % high_corridor_threshold,
                ))
        
        # Run camera optimization to cover movement channels
        optimized_cameras = []
        use_modal = True  # Set to True to use Modal cloud compute, False for local
        
        if len(corridor_lons) > 0:
            try:
                from .optimize import generate_candidate_sites
                from .viewshed import compute_coverage
                
                # Prepare deer activity points (corridor points)
                deer_points = list(zip(corridor_lons, corridor_lats))
                
                # Create transform for DEM
                from rasterio.transform import from_bounds
                vis_transform = from_bounds(
                    vis_bounds[0], vis_bounds[1], vis_bounds[2], vis_bounds[3],
                    vis_dem.shape[1], vis_dem.shape[0]
                )
                
                # Generate candidate camera sites from corridor points
                candidate_sites = generate_candidate_sites(
                    corridor_points=np.column_stack([corridor_lons, corridor_lats]),
                    dem=dem_sub,
                    dem_transform=vis_transform,
                    sample_fraction=0.05  # Sample 5% as candidates for faster computation
                )
                
                logger.info(f"Running camera placement optimization (using {'Modal cloud' if use_modal else 'local compute'})...")
                
                if use_modal:
                    # Use Modal for cloud compute
                    from .modal_compute import run_optimization_on_modal
                    
                    # Prepare target points with elevations
                    target_points = np.column_stack([
                        corridor_lons,
                        corridor_lats,
                        corridor_elevs
                    ])
                    
                    # Prepare candidates with elevations
                    candidates_array = np.array(candidate_sites)
                    
                    terrain_params = {
                        'fov': 45.0,
                        'max_range': 20.0,
                        'camera_height': 1.5,
                        'step_size': 1.0,
                        'vegetation_tolerance': 2.0,
                    }
                    
                    # Run on Modal cloud (8 cores, 32GB RAM)
                    modal_results = run_optimization_on_modal(
                        candidates=candidates_array,
                        target_points=target_points,
                        elevation=vis_dem,
                        transform=vis_transform,
                        bounds=vis_bounds,
                        num_cameras=min(6, len(candidate_sites)),
                        terrain_params=terrain_params,
                    )
                    
                    # Convert Modal results to expected format
                    optimized_cameras = []
                    for cam in modal_results:
                        optimized_cameras.append({
                            'x': cam['position'][0],
                            'y': cam['position'][1],
                            'azimuth': cam['azimuth'],
                            'coverage': cam['new_points'],
                        })
                else:
                    # Use local compute
                    from .optimize import optimize_cameras
                    
                    optimized_cameras = optimize_cameras(
                        deer_points=deer_points,
                        candidate_sites=candidate_sites,
                        dem=vis_dem,
                        dem_transform=vis_transform,
                        dem_crs=str(rasterio.CRS.from_epsg(4326)),  # WGS84
                        k=min(6, len(candidate_sites)),
                        camera_range=20.0,  # 20m visibility
                        fov_angle=45.0,  # 45 degree field of view
                        num_azimuths=8,
                    overlap_penalty=0.5  # Penalize overlap
                )
                
                # Track covered points for coloring
                covered_point_set = set()
                for camera in optimized_cameras:
                    covered_point_set.update(camera.covered_points)
                
                # Color corridor points: green if covered, red if not
                point_colors = []
                for i in range(len(deer_points)):
                    if i in covered_point_set:
                        point_colors.append('#00FF00')  # Green: covered
                    else:
                        point_colors.append('#DC143C')  # Red: uncovered
                
                # Re-draw movement channels with coverage coloring
                if unique_labels:
                    trace_indices['corridors'] = []
                    for label in sorted(unique_labels):
                        cluster_mask = labels == label
                        cluster_indices = np.where(cluster_mask)[0]
                        
                        cluster_lons_colored = corridor_lons[cluster_mask]
                        cluster_lats_colored = corridor_lats[cluster_mask]
                        cluster_elevs_colored = corridor_elevs[cluster_mask]
                        cluster_colors = [point_colors[i] for i in cluster_indices]
                        
                        # Sort points for path
                        if len(cluster_lons_colored) > 1:
                            remaining = set(range(len(cluster_lons_colored)))
                            path = [remaining.pop()]
                            
                            while remaining:
                                last_idx = path[-1]
                                distances = (
                                    (cluster_lons_colored[list(remaining)] - cluster_lons_colored[last_idx])**2 +
                                    (cluster_lats_colored[list(remaining)] - cluster_lats_colored[last_idx])**2
                                )
                                nearest_idx = list(remaining)[np.argmin(distances)]
                                path.append(nearest_idx)
                                remaining.remove(nearest_idx)
                            
                            path_lons = cluster_lons_colored[path]
                            path_lats = cluster_lats_colored[path]
                            path_elevs = cluster_elevs_colored[path] + 3
                            path_colors = [cluster_colors[i] for i in path]
                            
                            trace_indices['corridors'].append(len(fig.data))
                            fig.add_trace(go.Scatter3d(
                                x=path_lons,
                                y=path_lats,
                                z=path_elevs,
                                mode='lines+markers',
                                line=dict(width=4, color='#666666'),  # Gray lines
                                marker=dict(
                                    size=3,
                                    color=path_colors,  # Green where covered, red where not
                                    line=dict(width=0)
                                ),
                                name=f'Movement Channel {label + 1}',
                                showlegend=bool(label == min(unique_labels)),
                                legendgroup='channels',
                                hovertemplate=f'Channel {label + 1}<br>Predicted Trail<extra></extra>',
                            ))
                
                # Add camera markers
                if optimized_cameras:
                    camera_lons = [cam.location[0] for cam in optimized_cameras]
                    camera_lats = [cam.location[1] for cam in optimized_cameras]
                    camera_elevs = []
                    
                    for lon, lat in zip(camera_lons, camera_lats):
                        # Get elevation at camera location
                        x_idx = np.interp(lon, x_coords, np.arange(len(x_coords)))
                        y_idx = np.interp(lat, y_coords, np.arange(len(y_coords)))
                        x_idx = int(np.clip(x_idx, 0, dem_sub.shape[1] - 1))
                        y_idx = int(np.clip(y_idx, 0, dem_sub.shape[0] - 1))
                        camera_elevs.append(dem_sub[y_idx, x_idx] + 5)  # Elevated above terrain
                    
                    fig.add_trace(go.Scatter3d(
                        x=camera_lons,
                        y=camera_lats,
                        z=camera_elevs,
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color='#FFA500',  # Orange camera markers
                            symbol='diamond',
                            line=dict(width=2, color='#000000')
                        ),
                        text=[f'📷{i+1}' for i in range(len(optimized_cameras))],
                        textposition='top center',
                        textfont=dict(size=10, color='#000000'),
                        name='Optimized Cameras',
                        showlegend=True,
                        hovertemplate='Camera %{text}<br>Coverage: High<extra></extra>',
                    ))
                    
                    # Add coverage statistics annotation
                    coverage_pct = 100 * len(covered_point_set) / len(deer_points) if deer_points else 0
                    logger.info(f"Camera coverage: {len(covered_point_set)}/{len(deer_points)} points ({coverage_pct:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"Camera optimization failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Water feature dots removed per user request
        
        # Skip parcel boundary - removed per user request
        
        # Add compass rose annotations
        # Get map center and extent for positioning
        lon_center = (x_coords[0] + x_coords[-1]) / 2
        lat_center = (y_coords[0] + y_coords[-1]) / 2
        lon_range = x_coords[-1] - x_coords[0]
        lat_range = y_coords[-1] - y_coords[0]
        compass_offset = max(lon_range, lat_range) * 0.35  # Position at edge
        compass_z = dem_sub.max() * 1.1  # Above terrain
        
        # Add NESW compass markers
        compass_points = [
            ('N', lon_center, lat_center + compass_offset, 'North'),
            ('S', lon_center, lat_center - compass_offset, 'South'),
            ('E', lon_center + compass_offset, lat_center, 'East'),
            ('W', lon_center - compass_offset, lat_center, 'West'),
        ]
        
        for label, lon, lat, direction in compass_points:
            fig.add_trace(go.Scatter3d(
                x=[lon],
                y=[lat],
                z=[compass_z],
                mode='text',
                text=[label],
                textfont=dict(size=24, color='black', family='Arial Black'),
                textposition='middle center',
                showlegend=False,
                hovertemplate=f'{direction}<extra></extra>',
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive 3D Terrain with Movement Channels',
                font=dict(size=22, color='#2C3E50'),
                x=0.5,
                xanchor='center',
            ),
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5),  # Pull back camera for better view
                    center=dict(x=0, y=0, z=-0.05),
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.2, y=1.2, z=0.35),  # Slightly wider view
            ),
            width=1600,  # Larger display
            height=1000,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#2C3E50',
                borderwidth=2,
                font=dict(size=12),
            ),
            annotations=[
                dict(
                    text='Use toggle button to switch between Elevation and Satellite views',
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.02,
                    xanchor='center',
                    yanchor='bottom',
                    showarrow=False,
                    font=dict(size=10, color='#555'),
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='#999',
                    borderwidth=1,
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            label="Elevation",
                            method="update",
                            args=[
                                {"visible": [True, False] + [True] * (len(fig.data) - 2)},  # Show elevation, hide satellite, show all others
                                {"title": "Interactive 3D Terrain - Elevation View"}
                            ]
                        ),
                        dict(
                            label="Satellite",
                            method="update",
                            args=[
                                {"visible": [False, True] + [True] * (len(fig.data) - 2)},  # Hide elevation, show satellite, show all others
                                {"title": "Interactive 3D Terrain - Satellite View"}
                            ]
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=1.08,
                    yanchor="top",
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='#2C3E50',
                    borderwidth=2,
                )
            ],
        )
        
        # Save to HTML
        fig.write_html(
            str(output_path),
            config=dict(
                displayModeBar=True,
                displaylogo=False,
                modeBarButtonsToRemove=['select2d', 'lasso2d'],
            ),
        )
        
        return output_path

    def _format_bounds(self, bounds) -> str:
        """Format bounds tuple for display."""
        return f"({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})"

    def _create_overview_map(
        self,
        hillshade_path: Path,
        aoi_boundary: gpd.GeoDataFrame,
        cameras: list,
        dem_info: dict,
    ) -> RLImage | None:
        """Create overview map with hillshade, boundary, and camera locations.

        Args:
            hillshade_path: Path to hillshade raster
            aoi_boundary: AOI boundary GeoDataFrame
            cameras: List of camera points to overlay
            dem_info: DEM metadata with transform info

        Returns:
            ReportLab Image object or None
        """
        try:
            with rasterio.open(hillshade_path) as src:
                hillshade = src.read(1)
                transform = src.transform

                # Apply contrast stretching for better visibility
                p2, p98 = np.percentile(hillshade, (2, 98))
                if p98 - p2 > 0:
                    hillshade_stretched = np.clip((hillshade - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                else:
                    hillshade_stretched = hillshade.astype(np.uint8)

                # Convert to RGB for overlays
                img_array = np.stack([hillshade_stretched, hillshade_stretched, hillshade_stretched], axis=-1)
                img = Image.fromarray(img_array, mode="RGB")

                # Draw camera markers
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                
                for cam in cameras:
                    # Convert lat/lon to pixel coordinates
                    col, row = ~transform * (cam.lon, cam.lat)
                    col, row = int(col), int(row)
                    
                    # Skip if outside bounds
                    if 0 <= row < hillshade.shape[0] and 0 <= col < hillshade.shape[1]:
                        # Draw different colors for different camera types
                        color = (255, 0, 0) if cam.camera_type == "pinch" else (0, 255, 0)
                        radius = 15
                        
                        # Draw circle marker with thick outline
                        draw.ellipse(
                            [col - radius, row - radius, col + radius, row + radius],
                            fill=color,
                            outline=(0, 0, 0),
                            width=4,
                        )
                        
                        # Draw camera number with larger font
                        cam_num = cameras.index(cam) + 1
                        # Draw text with black background for readability
                        text = str(cam_num)
                        text_bbox = draw.textbbox((col, row), text)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        text_x = col - text_width // 2
                        text_y = row - text_height // 2
                        # Black outline for text
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                draw.text((text_x + dx, text_y + dy), text, fill=(0, 0, 0))
                        draw.text((text_x, text_y), text, fill=(255, 255, 255))

                # Resize for report (max width 5 inches)
                max_width = int(5 * 72)  # 5 inches in points
                aspect = img.height / img.width
                new_width = min(img.width, max_width)
                new_height = int(new_width * aspect)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save debug PNG with overlays
                debug_png = hillshade_path.parent / "overview_map.png"
                img.save(debug_png)
                logger.debug(f"Saved overview map with overlays: {debug_png}")

                # Save to BytesIO
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                # Create ReportLab image
                return RLImage(img_buffer, width=new_width, height=new_height)

        except Exception as e:
            logger.warning(f"Failed to create overview map: {e}")
            return None

    def _create_heatmap_image(
        self,
        score_path: Path,
        colormap: str = "viridis",
        cameras: list | None = None,
        dem_info: dict | None = None,
        camera_filter: str | None = None,
    ) -> RLImage | None:
        """Create heatmap image from score raster with optional camera overlay.

        Args:
            score_path: Path to score raster
            colormap: Matplotlib-style colormap name
            cameras: Optional list of camera points to overlay
            dem_info: DEM metadata with transform info
            camera_filter: Filter cameras by type ('pinch' or 'bedding')

        Returns:
            ReportLab Image object or None
        """
        try:
            with rasterio.open(score_path) as src:
                score = src.read(1)
                transform = src.transform

                # Apply contrast stretching for better visibility
                p2, p98 = np.percentile(score[score > 0], (5, 95))
                score_clipped = np.clip(score, p2, p98)
                score_norm = ((score_clipped - score_clipped.min()) / (score_clipped.max() - score_clipped.min()) * 255).astype(
                    np.uint8
                )

                # Apply colormap with better contrast
                if colormap == "Reds":
                    rgb = np.stack([score_norm, score_norm // 4, score_norm // 4], axis=-1)
                elif colormap == "Greens":
                    rgb = np.stack([score_norm // 4, score_norm, score_norm // 4], axis=-1)
                else:
                    rgb = np.stack([score_norm, score_norm, score_norm], axis=-1)

                # Convert to PIL Image
                img = Image.fromarray(rgb, mode="RGB")

                # Overlay camera markers if provided
                if cameras and dem_info:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(img)
                    
                    # Filter cameras by type if specified
                    filtered_cameras = cameras
                    if camera_filter:
                        filtered_cameras = [
                            cam for cam in cameras if cam.camera_type == camera_filter
                        ]
                    
                    for cam in filtered_cameras:
                        # Convert lat/lon to pixel coordinates
                        col, row = ~transform * (cam.lon, cam.lat)
                        col, row = int(col), int(row)
                        
                        # Skip if outside bounds
                        if 0 <= row < score.shape[0] and 0 <= col < score.shape[1]:
                            # Draw marker with high contrast
                            color = (255, 255, 0)  # Yellow for visibility
                            radius = 15
                            
                            # Draw circle marker with thick outline
                            draw.ellipse(
                                [col - radius, row - radius, col + radius, row + radius],
                                fill=color,
                                outline=(0, 0, 0),
                                width=5,
                            )
                            
                            # Draw camera number with outline
                            cam_num = cameras.index(cam) + 1
                            text = str(cam_num)
                            text_bbox = draw.textbbox((col, row), text)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            text_x = col - text_width // 2
                            text_y = row - text_height // 2
                            # Black outline for text
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    draw.text((text_x + dx, text_y + dy), text, fill=(0, 0, 0))
                            draw.text((text_x, text_y), text, fill=(255, 255, 255))

                # Resize for report (max width 4 inches)
                max_width = int(4 * 72)
                aspect = img.height / img.width
                new_width = min(img.width, max_width)
                new_height = int(new_width * aspect)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save debug PNG with overlays
                filter_suffix = f"_{camera_filter}" if camera_filter else ""
                debug_png = score_path.parent / f"heatmap{filter_suffix}.png"
                img.save(debug_png)
                logger.debug(f"Saved heatmap with overlays: {debug_png}")

                # Save to BytesIO
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                return RLImage(img_buffer, width=new_width, height=new_height)

        except Exception as e:
            logger.warning(f"Failed to create heatmap: {e}")
            return None

    def _create_zone_table(self, zones: list[RecommendationZone]) -> Table:
        """Create table of recommendation zones with camera allocation.

        Args:
            zones: List of RecommendationZone objects

        Returns:
            ReportLab Table object
        """
        # Table data
        data = [["Zone", "Type", "Priority", "Area (km²)", "Cameras", "Zone Description"]]

        for zone in zones:
            # Priority indicator
            priority_mark = "★★★" if zone.priority == "high" else "★★" if zone.priority == "medium" else "★"
            
            # Type indicator  
            type_icon = "🎯" if zone.zone_type == "pinch" else "🛏️" if zone.zone_type == "bedding" else "🔄"
            
            data.append(
                [
                    f"{zone.zone_id}",
                    f"{type_icon} {zone.zone_type.title()}",
                    priority_mark,
                    f"{zone.area_m2/1e6:.3f}",
                    str(zone.recommended_cameras),
                    zone.description,
                ]
            )

        # Create table
        table = Table(
            data, colWidths=[0.5 * inch, 1.0 * inch, 0.6 * inch, 0.8 * inch, 0.6 * inch, 3.5 * inch]
        )

        # Style table
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("TOPPADDING", (0, 1), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#ECF0F1")],
                    ),
                ]
            )
        )

        return table

    def _create_camera_table(self, cameras: list) -> Table:
        """Legacy camera table - kept for compatibility."""
        # Table data
        data = [["#", "Location", "Score", "Terrain", "Placement Rationale"]]

        for i, cam in enumerate(cameras, 1):
            # Format terrain info
            terrain_info = f"Slope: {cam.slope:.1f}°\nAspect: {cam.aspect:.0f}°"
            
            # Location string
            location_str = f"{cam.lat:.5f}°N\n{cam.lon:.5f}°W"
            
            data.append(
                [
                    str(i),
                    location_str,
                    f"{cam.score:.3f}",
                    terrain_info,
                    cam.reason,
                ]
            )

        # Create table
        table = Table(
            data, colWidths=[0.35 * inch, 1.1 * inch, 0.5 * inch, 0.8 * inch, 4.25 * inch]
        )

        # Style table
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("TOPPADDING", (0, 1), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#ECF0F1")],
                    ),
                ]
            )
        )

        return table
