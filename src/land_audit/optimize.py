"""Camera placement optimization using greedy maximum coverage algorithm.

This module solves the camera placement problem: given K cameras and N deer activity
points, select camera positions that maximize coverage of deer activity.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, Point
import rasterio
from .viewshed import compute_wedge, compute_coverage

logger = logging.getLogger(__name__)


@dataclass
class CameraPlacement:
    """Represents an optimized camera placement."""
    location: Tuple[float, float]  # (lon, lat)
    azimuth: float  # Direction camera faces (degrees)
    coverage_polygon: Polygon  # Viewshed area
    covered_points: List[int]  # Indices of covered deer points
    marginal_gain: int  # Number of new points covered by this camera
    
    @property
    def coverage_count(self) -> int:
        """Number of deer points covered by this camera."""
        return len(self.covered_points)


def generate_candidate_sites(
    corridor_points: np.ndarray,
    dem: np.ndarray,
    dem_transform: rasterio.transform.Affine,
    sample_fraction: float = 0.2
) -> List[Tuple[float, float]]:
    """Generate candidate camera sites from corridor points.
    
    Samples a subset of high-value corridor points as potential camera locations.
    
    Args:
        corridor_points: Array of shape (N, 2) with (lon, lat) coordinates
        dem: Digital elevation model
        dem_transform: DEM affine transform  
        sample_fraction: Fraction of points to sample as candidates
        
    Returns:
        List of (lon, lat) candidate camera locations
    """
    # Sample a subset to reduce computation
    n_samples = max(20, int(len(corridor_points) * sample_fraction))
    n_samples = min(n_samples, len(corridor_points))
    
    # Random sampling with seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.choice(len(corridor_points), size=n_samples, replace=False)
    
    candidates = [tuple(corridor_points[i]) for i in indices]
    
    logger.info(f"Generated {len(candidates)} candidate camera sites from {len(corridor_points)} corridor points")
    
    return candidates


def optimize_cameras(
    deer_points: List[Tuple[float, float]],
    candidate_sites: List[Tuple[float, float]],
    dem: np.ndarray,
    dem_transform: rasterio.transform.Affine,
    dem_crs: str,
    k: int = 6,
    camera_range: float = 20.0,
    fov_angle: float = 45.0,
    num_azimuths: int = 8,
    overlap_penalty: float = 0.5
) -> List[CameraPlacement]:
    """Optimize camera placement using greedy maximum coverage algorithm with overlap penalty.
    
    Args:
        deer_points: List of (lon, lat) deer activity points to cover
        candidate_sites: List of (lon, lat) potential camera locations
        dem: Digital elevation model
        dem_transform: DEM affine transform
        dem_crs: DEM coordinate reference system
        k: Number of cameras to place
        camera_range: Maximum camera visibility range in meters
        fov_angle: Camera field of view in degrees
        num_azimuths: Number of azimuths to test per candidate site
        overlap_penalty: Penalty factor for overlapping coverage (0-1)
        
    Returns:
        List of CameraPlacement objects (length k) representing optimal camera positions
    """
    logger.info(f"Starting camera optimization: {k} cameras, {len(deer_points)} deer points, {len(candidate_sites)} candidates")
    
    selected_cameras: List[CameraPlacement] = []
    selected_positions: List[Tuple[float, float]] = []  # Track camera positions for distance checks
    covered_point_indices = set()  # Track globally covered points
    covered_count_per_point = np.zeros(len(deer_points), dtype=int)  # Track how many cameras cover each point
    
    # Anti-clustering parameters
    min_camera_distance = 50.0  # Minimum 50m between cameras to prevent clustering
    distance_penalty_factor = 2.0  # Heavy penalty for cameras too close together
    
    logger.info(f"Overlap penalty: {overlap_penalty}, Min camera distance: {min_camera_distance}m")
    
    # Test different azimuths (directions camera can face)
    test_azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
    
    for iteration in range(k):
        best_camera: Optional[CameraPlacement] = None
        best_score = -float('inf')  # Use -inf to allow negative scores
        
        logger.info(f"Iteration {iteration + 1}/{k}: Evaluating {len(candidate_sites)} candidates x {num_azimuths} azimuths")
        
        # Evaluate each candidate site at each azimuth
        for candidate in candidate_sites:
            for azimuth in test_azimuths:
                # Compute viewshed for this camera configuration
                try:
                    viewshed = compute_wedge(
                        observer_pt=candidate,
                        target_azimuth=azimuth,
                        radius=camera_range,
                        fov_angle=fov_angle,
                        dem=dem,
                        dem_transform=dem_transform,
                        dem_crs=dem_crs
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute viewshed for {candidate} @ {azimuth}°: {e}")
                    continue
                
                # Count coverage of points
                _, covered_indices = compute_coverage(viewshed, deer_points)
                
                # Calculate base score: new points - overlap penalty
                new_covered = set(covered_indices) - covered_point_indices
                new_points_score = len(new_covered)
                
                # Penalty for covering already-covered points (reduces redundant overlap)
                overlapping_points = set(covered_indices) & covered_point_indices
                overlap_penalty_score = len(overlapping_points) * overlap_penalty
                
                # Base score: maximize new coverage, minimize overlap
                score = new_points_score - overlap_penalty_score
                
                # Apply spatial diversity penalty to prevent clustering
                cx, cy = candidate
                if selected_positions:
                    # Calculate distance to all existing cameras
                    distances = [
                        np.sqrt((cx - ex)**2 + (cy - ey)**2)
                        for ex, ey in selected_positions
                    ]
                    min_distance_to_existing = min(distances)
                    
                    # Apply distance penalty if too close to existing cameras
                    if min_distance_to_existing < min_camera_distance:
                        distance_penalty = distance_penalty_factor * (min_camera_distance - min_distance_to_existing)
                        score -= distance_penalty
                
                # Update best camera if this configuration is better
                if score > best_score:
                    best_score = score
                    best_camera = CameraPlacement(
                        location=candidate,
                        azimuth=azimuth,
                        coverage_polygon=viewshed,
                        covered_points=covered_indices,
                        marginal_gain=len(new_covered)
                    )
        
        # If no improvement possible, stop early
        if best_camera is None or best_score <= -float('inf'):
            logger.info(f"No valid camera placement after {iteration} cameras. Stopping early.")
            break
        
        # Add best camera to solution
        selected_cameras.append(best_camera)
        selected_positions.append(best_camera.location)  # Track position for distance checks
        
        # Update coverage tracking
        for idx in best_camera.covered_points:
            covered_count_per_point[idx] += 1
        covered_point_indices.update(best_camera.covered_points)
        
        # Calculate overlap metrics
        overlap_count = np.sum(covered_count_per_point > 1)
        overlap_pct = 100 * overlap_count / len(deer_points) if deer_points else 0
        
        # Calculate distance to nearest existing camera
        if len(selected_positions) > 1:
            cx, cy = best_camera.location
            distances = [
                np.sqrt((cx - ex)**2 + (cy - ey)**2)
                for ex, ey in selected_positions[:-1]
            ]
            min_dist = min(distances)
        else:
            min_dist = float('inf')
        
        logger.info(
            f"Camera {iteration + 1}: Location={best_camera.location}, "
            f"Azimuth={best_camera.azimuth:.1f}°, "
            f"New Points={best_camera.marginal_gain}, "
            f"Total Coverage={len(covered_point_indices)}/{len(deer_points)} "
            f"({100 * len(covered_point_indices) / len(deer_points):.1f}%), "
            f"Overlap={overlap_count} ({overlap_pct:.1f}%), "
            f"Distance to nearest camera={min_dist:.1f}m"
        )
        
        # Stop if we've achieved 100% coverage
        if len(covered_point_indices) == len(deer_points):
            logger.info(f"✅ Achieved 100% coverage with {iteration + 1} cameras")
            break
    
    coverage_pct = 100 * len(covered_point_indices) / len(deer_points) if deer_points else 0
    logger.info(
        f"Optimization complete: {len(selected_cameras)} cameras placed, "
        f"{len(covered_point_indices)}/{len(deer_points)} points covered ({coverage_pct:.1f}%)"
    )
    
    return selected_cameras
