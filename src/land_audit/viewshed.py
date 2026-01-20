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
    obs_lon, obs_lat = observer_pt

    # Convert radius from meters to degrees (approximate)
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    meters_per_degree_lat = 111000.0
    meters_per_degree_lon = 111000.0 * np.cos(np.radians(obs_lat))

    # Convert step size to degrees
    step_size_lat = step_size / meters_per_degree_lat
    step_size_lon = step_size / meters_per_degree_lon

    # Get observer elevation from DEM
    obs_row, obs_col = rowcol(dem_transform, obs_lon, obs_lat)
    obs_row = max(0, min(int(obs_row), dem.shape[0] - 1))
    obs_col = max(0, min(int(obs_col), dem.shape[1] - 1))
    obs_elevation = float(dem[obs_row, obs_col]) + camera_height

    # Calculate wedge angles
    half_fov = fov_angle / 2.0
    start_azimuth = target_azimuth - half_fov
    end_azimuth = target_azimuth + half_fov

    # Sample angles within the wedge
    num_rays = max(8, int(fov_angle / 5))
    azimuths = np.linspace(start_azimuth, end_azimuth, num_rays)

    # Store visible points for each ray (in lon/lat degrees)
    visible_points = [(obs_lon, obs_lat)]

    for azimuth in azimuths:
        # Convert azimuth to radians (0=North, clockwise)
        azimuth_rad = np.radians(90 - azimuth)

        # Direction vector (in degrees, accounting for lat/lon scaling)
        dx_deg = np.cos(azimuth_rad) * step_size_lon / step_size  # Scale by step
        dy_deg = np.sin(azimuth_rad) * step_size_lat / step_size

        # Sample points along this ray
        num_steps = int(radius / step_size)
        ray_end = None

        for i in range(1, num_steps + 1):
            # Target point in degrees
            target_lon = obs_lon + dx_deg * i * step_size
            target_lat = obs_lat + dy_deg * i * step_size

            # Convert to DEM row/col
            target_row, target_col = rowcol(dem_transform, target_lon, target_lat)

            # Check if within DEM bounds
            if (target_row < 0 or target_row >= dem.shape[0] or
                target_col < 0 or target_col >= dem.shape[1]):
                break

            target_row = int(target_row)
            target_col = int(target_col)

            # Get terrain elevation at target
            terrain_elevation = float(dem[target_row, target_col])

            # Line-of-sight check
            distance_m = i * step_size
            required_elevation = obs_elevation - (obs_elevation - terrain_elevation) * (distance_m / radius)

            if terrain_elevation > required_elevation + 2.0:  # 2m tolerance
                break

            ray_end = (target_lon, target_lat)

        if ray_end:
            visible_points.append(ray_end)

    # Close the polygon back to observer
    if len(visible_points) < 3:
        # Degenerate case: return tiny buffer around observer
        buffer_deg = radius / meters_per_degree_lat  # Convert radius to degrees
        return Point(obs_lon, obs_lat).buffer(buffer_deg)

    try:
        # Create polygon from visible points (already in WGS84 degrees)
        viewshed = Polygon(visible_points)
        if not viewshed.is_valid:
            viewshed = viewshed.buffer(0)  # Fix self-intersections
        return viewshed

    except Exception as e:
        logger.warning(f"Failed to create viewshed polygon: {e}")
        buffer_deg = radius / meters_per_degree_lat
        return Point(obs_lon, obs_lat).buffer(buffer_deg)


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
