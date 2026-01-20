"""Viewshed computation for camera placement optimization.

This module calculates the visible ground area (viewshed) from a camera position,
accounting for terrain occlusion and camera field of view.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

logger = logging.getLogger(__name__)


def compute_wedge(
    observer_pt: Tuple[float, float],
    target_azimuth: float,
    radius: float,
    fov_angle: float,
    dem: np.ndarray,
    dem_transform: rasterio.transform.Affine,
    dem_crs: str,
    camera_height: float = 1.5,
    step_size: float = 1.0,
) -> Polygon:
    """Compute the visible ground area (viewshed wedge) from a camera position.
    
    Args:
        observer_pt: Camera location as (lon, lat) in degrees
        target_azimuth: Direction camera is facing (degrees, 0=North, 90=East)
        radius: Maximum camera range in meters
        fov_angle: Camera field of view in degrees (e.g., 45)
        dem: Digital elevation model as 2D numpy array
        dem_transform: Affine transform from DEM rasterio dataset
        dem_crs: Coordinate reference system of DEM
        camera_height: Height of camera above ground in meters
        step_size: Distance between sample points along rays in meters
        
    Returns:
        Shapely Polygon representing the visible ground area (viewshed cone)
    """
    # Create transformer from WGS84 to DEM CRS
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    
    # Convert observer to DEM coordinates
    obs_x, obs_y = transformer.transform(observer_pt[0], observer_pt[1])
    obs_row, obs_col = rowcol(dem_transform, obs_x, obs_y)
    
    # Clamp to valid DEM range
    obs_row = max(0, min(obs_row, dem.shape[0] - 1))
    obs_col = max(0, min(obs_col, dem.shape[1] - 1))
    
    # Get observer elevation
    obs_elevation = float(dem[obs_row, obs_col]) + camera_height
    
    # Calculate wedge angles
    half_fov = fov_angle / 2.0
    start_azimuth = target_azimuth - half_fov
    end_azimuth = target_azimuth + half_fov
    
    # Sample angles within the wedge
    num_rays = max(8, int(fov_angle / 5))  # At least 8 rays, more for wider FOV
    azimuths = np.linspace(start_azimuth, end_azimuth, num_rays)
    
    # Store visible points for each ray
    visible_points = [Point(obs_x, obs_y)]  # Include observer position
    
    for azimuth in azimuths:
        # Convert azimuth to radians (0=North, clockwise)
        azimuth_rad = np.radians(90 - azimuth)  # Convert to math convention
        
        # Direction vector
        dx = np.cos(azimuth_rad)
        dy = np.sin(azimuth_rad)
        
        # Sample points along this ray
        num_steps = int(radius / step_size)
        ray_visible = []
        
        for i in range(1, num_steps + 1):
            distance = i * step_size
            
            # Target point in map coordinates
            target_x = obs_x + dx * distance
            target_y = obs_y + dy * distance
            
            # Convert to DEM row/col
            target_row, target_col = rowcol(dem_transform, target_x, target_y)
            
            # Check if within DEM bounds
            if (target_row < 0 or target_row >= dem.shape[0] or
                target_col < 0 or target_col >= dem.shape[1]):
                break
            
            # Get terrain elevation at target
            terrain_elevation = float(dem[target_row, target_col])
            
            # Line-of-sight check: simple slope comparison
            # If terrain rises above line from observer to target, ray is blocked
            required_elevation = obs_elevation - (obs_elevation - terrain_elevation) * (distance / radius)
            
            if terrain_elevation > required_elevation + 2.0:  # 2m tolerance for vegetation
                # Ray blocked, stop extending this ray
                break
            
            ray_visible.append(Point(target_x, target_y))
        
        # Add the farthest visible point on this ray
        if ray_visible:
            visible_points.append(ray_visible[-1])
    
    # Close the polygon back to observer
    if len(visible_points) < 3:
        # Degenerate case: very small viewshed, return tiny buffer
        return Point(obs_x, obs_y).buffer(step_size)
    
    try:
        # Create polygon from visible points
        viewshed = Polygon([(p.x, p.y) for p in visible_points])
        
        # Convert back to WGS84 for consistency
        transformer_back = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        coords = []
        for x, y in viewshed.exterior.coords:
            lon, lat = transformer_back.transform(x, y)
            coords.append((lon, lat))
        
        return Polygon(coords)
    
    except Exception as e:
        logger.warning(f"Failed to create viewshed polygon: {e}")
        # Return minimal buffer around observer
        lon, lat = observer_pt
        return Point(lon, lat).buffer(0.0001)  # ~10m buffer in degrees


def compute_coverage(
    viewshed: Polygon,
    target_points: list[Tuple[float, float]]
) -> Tuple[int, list[int]]:
    """Count how many target points are covered by a viewshed.
    
    Args:
        viewshed: Shapely Polygon representing visible area
        target_points: List of (lon, lat) points to check coverage
        
    Returns:
        Tuple of (count of covered points, list of covered point indices)
    """
    covered_indices = []
    
    for i, (lon, lat) in enumerate(target_points):
        point = Point(lon, lat)
        if viewshed.contains(point):
            covered_indices.append(i)
    
    return len(covered_indices), covered_indices
