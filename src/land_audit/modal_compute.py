"""
Modal-based cloud compute for CPU-intensive land audit operations.
Offloads heavy computations (viewshed calculations, optimization) to Modal's cloud infrastructure.
"""

import modal
import numpy as np
from typing import List, Tuple, Dict, Any
import pickle

# Create Modal app
app = modal.App("orome-land-audit")

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "shapely>=2.0.0",
        "rasterio>=1.3.0",
        "pyproj>=3.5.0",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,  # 32GB RAM
    timeout=3600,  # 1 hour timeout
)
def compute_viewshed_batch(
    candidates: List[Tuple[float, float, float]],  # (x, y, azimuth)
    azimuths: List[float],
    dem_data: bytes,  # Pickled DEM arrays
    target_points: bytes,  # Pickled target point coordinates
    terrain_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compute viewsheds for a batch of candidate camera positions in parallel.
    
    Args:
        candidates: List of (x, y, elevation) tuples for candidate positions
        azimuths: List of azimuth angles to test for each candidate
        dem_data: Pickled dict with 'elevation', 'transform', 'bounds'
        target_points: Pickled array of (x, y, z) target point coordinates
        terrain_params: Dict with 'fov', 'max_range', 'camera_height', 'step_size'
    
    Returns:
        List of dicts with keys: 'candidate_idx', 'azimuth', 'coverage_count', 'covered_indices'
    """
    import numpy as np
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    import pickle
    
    # Unpickle data
    dem = pickle.loads(dem_data)
    targets = pickle.loads(target_points)
    
    elevation = dem['elevation']
    transform = dem['transform']
    bounds = dem['bounds']
    
    # Extract terrain parameters
    fov = terrain_params.get('fov', 45.0)
    max_range = terrain_params.get('max_range', 20.0)
    camera_height = terrain_params.get('camera_height', 1.5)
    step_size = terrain_params.get('step_size', 1.0)
    vegetation_tolerance = terrain_params.get('vegetation_tolerance', 2.0)
    
    results = []
    
    def world_to_pixel(x, y, transform):
        """Convert world coordinates to pixel coordinates."""
        col = int((x - transform[2]) / transform[0])
        row = int((y - transform[5]) / transform[4])
        return row, col
    
    def compute_wedge(cam_x, cam_y, cam_z, azimuth):
        """Compute viewshed wedge with terrain occlusion."""
        # Generate rays within FOV
        half_fov = fov / 2.0
        angles = np.linspace(azimuth - half_fov, azimuth + half_fov, 20)
        
        wedge_points = [(cam_x, cam_y)]
        
        for angle in angles:
            rad = np.radians(angle)
            dx = max_range * np.cos(rad)
            dy = max_range * np.sin(rad)
            
            # Sample along ray
            num_steps = int(max_range / step_size)
            for i in range(1, num_steps + 1):
                frac = i / num_steps
                ray_x = cam_x + dx * frac
                ray_y = cam_y + dy * frac
                
                # Check bounds
                if not (bounds[0] <= ray_x <= bounds[2] and bounds[1] <= ray_y <= bounds[3]):
                    break
                
                # Get terrain elevation
                row, col = world_to_pixel(ray_x, ray_y, transform)
                if 0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1]:
                    terrain_z = elevation[row, col]
                    
                    # Line-of-sight check with vegetation tolerance
                    expected_z = cam_z + camera_height - (frac * 0.5)  # Simple slope model
                    if terrain_z > expected_z + vegetation_tolerance:
                        # Occluded
                        break
                
                if i == num_steps:
                    wedge_points.append((ray_x, ray_y))
        
        if len(wedge_points) > 2:
            return Polygon(wedge_points)
        return None
    
    def compute_coverage(wedge_poly):
        """Count how many target points are covered by the viewshed."""
        if wedge_poly is None:
            return 0, []
        
        covered = []
        for idx, (tx, ty, tz) in enumerate(targets):
            if wedge_poly.contains(Point(tx, ty)):
                covered.append(idx)
        
        return len(covered), covered
    
    # Process each candidate
    for cand_idx, (cx, cy, cz) in enumerate(candidates):
        for azimuth in azimuths:
            wedge = compute_wedge(cx, cy, cz, azimuth)
            coverage_count, covered_indices = compute_coverage(wedge)
            
            results.append({
                'candidate_idx': cand_idx,
                'azimuth': azimuth,
                'coverage_count': coverage_count,
                'covered_indices': covered_indices,
            })
    
    return results


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600,
)
def optimize_camera_placement_cloud(
    candidates: List[Tuple[float, float, float]],
    target_points: List[Tuple[float, float, float]],
    dem_data: bytes,
    num_cameras: int,
    terrain_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Full camera optimization in the cloud using greedy algorithm.
    
    Returns:
        List of selected cameras with positions, azimuths, and coverage metrics
    """
    import numpy as np
    import pickle
    
    # Unpickle
    dem = pickle.loads(dem_data)
    targets = pickle.loads(target_points)
    
    # Generate azimuths
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    
    selected_cameras = []
    selected_positions = []  # Track camera positions to enforce minimum distance
    covered_points = set()
    coverage_count_per_point = np.zeros(len(targets))
    
    overlap_penalty = 0.5
    min_camera_distance = 50.0  # Minimum 50m between cameras to prevent clustering
    distance_penalty_factor = 2.0  # Heavy penalty for cameras too close together
    
    print(f"Starting optimization: {num_cameras} cameras, {len(targets)} targets, {len(candidates)} candidates")
    print(f"Overlap penalty: {overlap_penalty}, Min camera distance: {min_camera_distance}m")
    
    for iteration in range(num_cameras):
        print(f"Iteration {iteration + 1}/{num_cameras}: Evaluating {len(candidates)} candidates x {len(azimuths)} azimuths")
        
        best_score = -float('inf')  # Use -inf to allow negative scores
        best_candidate = None
        best_azimuth = None
        best_covered = None
        
        # Batch compute all viewsheds for this iteration
        # Split into smaller batches to avoid memory issues
        batch_size = 50
        
        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]
            
            # Call viewshed computation
            batch_results = compute_viewshed_batch.local(
                batch_candidates,
                azimuths,
                dem_data,
                target_points,
                terrain_params,
            )
            
            # Evaluate scores
            for result in batch_results:
                covered_set = set(result['covered_indices'])
                new_covered = covered_set - covered_points
                overlap = covered_set & covered_points
                
                # Base score: new coverage minus overlap penalty
                score = len(new_covered) - (len(overlap) * overlap_penalty)
                
                # Apply spatial diversity penalty to prevent clustering
                cand_idx = batch_start + result['candidate_idx']
                cx, cy, cz = candidates[cand_idx]
                
                # Check distance to all existing cameras
                min_distance_to_existing = float('inf')
                for existing_pos in selected_positions:
                    ex, ey, _ = existing_pos
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                    min_distance_to_existing = min(min_distance_to_existing, distance)
                
                # Apply distance penalty if too close to existing cameras
                if len(selected_positions) > 0 and min_distance_to_existing < min_camera_distance:
                    distance_penalty = distance_penalty_factor * (min_camera_distance - min_distance_to_existing)
                    score -= distance_penalty
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand_idx
                    best_azimuth = result['azimuth']
                    best_covered = covered_set
        
        if best_candidate is None:
            print(f"No improvement found at iteration {iteration + 1}")
            break
        
        # Add camera
        cx, cy, cz = candidates[best_candidate]
        selected_cameras.append({
            'position': (cx, cy, cz),
            'azimuth': best_azimuth,
            'new_points': len(best_covered - covered_points),
            'overlap': len(best_covered & covered_points),
        })
        selected_positions.append((cx, cy, cz))  # Track position for distance checks
        
        # Update covered points
        old_covered_count = len(covered_points)
        for idx in best_covered:
            coverage_count_per_point[idx] += 1
        covered_points.update(best_covered)
        
        coverage_pct = (len(covered_points) / len(targets)) * 100
        overlap_count = len(best_covered & set(list(covered_points)[:old_covered_count]))
        overlap_pct = (overlap_count / len(best_covered)) * 100 if best_covered else 0
        
        # Calculate distance to nearest existing camera
        if len(selected_positions) > 1:
            distances = [np.sqrt((cx - ex)**2 + (cy - ey)**2) for ex, ey, _ in selected_positions[:-1]]
            min_dist = min(distances)
        else:
            min_dist = float('inf')
        
        print(f"Camera {iteration + 1}: Position={cx:.6f},{cy:.6f}, Azimuth={best_azimuth}°")
        print(f"  New Points={len(best_covered) - overlap_count}, Total Coverage={len(covered_points)}/{len(targets)} ({coverage_pct:.1f}%)")
        print(f"  Overlap={overlap_count} ({overlap_pct:.1f}%), Distance to nearest camera={min_dist:.1f}m")
        
        # Stop if we've achieved 100% coverage
        if len(covered_points) == len(targets):
            print(f"✅ Achieved 100% coverage with {iteration + 1} cameras")
            break
        print(f"  Overlap={len(best_covered & covered_points)} ({overlap_pct:.1f}%)")
    
    return selected_cameras


# Local wrapper functions for easy calling
def run_optimization_on_modal(
    candidates: np.ndarray,
    target_points: np.ndarray,
    elevation: np.ndarray,
    transform: Tuple,
    bounds: Tuple,
    num_cameras: int,
    terrain_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Local function to invoke Modal optimization.
    
    Args:
        candidates: Array of shape (N, 3) with (x, y, z) coordinates
        target_points: Array of shape (M, 3) with (x, y, z) coordinates
        elevation: 2D array of terrain elevations
        transform: Affine transform tuple
        bounds: (minx, miny, maxx, maxy) bounds
        num_cameras: Number of cameras to place
        terrain_params: Dict with viewshed parameters
    
    Returns:
        List of selected camera configurations
    """
    import pickle
    
    # Pickle data for transmission
    dem_data = pickle.dumps({
        'elevation': elevation,
        'transform': transform,
        'bounds': bounds,
    })
    
    target_data = pickle.dumps(target_points)
    
    # Convert to lists
    candidates_list = [tuple(c) for c in candidates]
    
    # Run on Modal
    try:
        with app.run():
            result = optimize_camera_placement_cloud.remote(
                candidates_list,
                target_points.tolist(),
                dem_data,
                num_cameras,
                terrain_params,
            )
        return result
    except Exception as e:
        print(f"Modal execution failed: {e}")
        return []
