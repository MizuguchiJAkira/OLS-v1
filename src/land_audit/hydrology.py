"""Hydrology and water feature detection with particle simulation."""

import logging
from pathlib import Path

import numpy as np
import rasterio
import requests
from scipy import ndimage
from shapely.geometry import shape, Point

logger = logging.getLogger(__name__)


class HydrologyAnalysis:
    """Analyzes water features using particle-based simulation."""

    def __init__(
        self,
        elevation: np.ndarray,
        transform: rasterio.Affine,
        crs: rasterio.crs.CRS,
        historical_rainfall_mm: float = 800.0,  # Annual rainfall (default ~800mm for NY)
    ):
        """Initialize hydrology analysis.

        Args:
            elevation: Elevation array (meters)
            transform: Affine transform for georeferencing
            crs: Coordinate reference system
            historical_rainfall_mm: Historical annual rainfall in millimeters
        """
        self.elevation = elevation
        self.transform = transform
        self.crs = crs
        self.historical_rainfall_mm = historical_rainfall_mm
    
    def fetch_osm_water_bodies(
        self,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> np.ndarray:
        """Fetch real water bodies from OpenStreetMap.
        
        Queries OSM Overpass API for waterways (rivers, streams) and water bodies
        (lakes, ponds, reservoirs) within the area of interest.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_m: Radius in meters
            
        Returns:
            Binary mask of water pixels (1=water, 0=land)
        """
        logger.info(f"Querying OpenStreetMap for water bodies within {radius_m}m of ({lat}, {lon})")
        
        # Create empty water mask
        rows, cols = self.elevation.shape
        water_mask = np.zeros((rows, cols), dtype=np.uint8)
        
        # Overpass API query for water features
        # natural=water covers lakes, ponds, reservoirs
        # waterway covers rivers, streams, canals
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="water"](around:{radius_m},{lat},{lon});
          way["waterway"](around:{radius_m},{lat},{lon});
          relation["natural"="water"](around:{radius_m},{lat},{lon});
          relation["waterway"](around:{radius_m},{lat},{lon});
        );
        out geom;
        """
        
        try:
            response = requests.post(overpass_url, data={'data': query}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('elements'):
                logger.warning("No water bodies found in OpenStreetMap data")
                return water_mask
            
            logger.info(f"Found {len(data['elements'])} water features in OSM")
            
            # Convert OSM geometries to raster mask
            for element in data['elements']:
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    self._rasterize_line(coords, water_mask, buffer_pixels=2)
                elif element['type'] == 'relation' and 'members' in element:
                    # Handle multipolygons (lakes, etc.)
                    for member in element['members']:
                        if 'geometry' in member:
                            coords = [(node['lon'], node['lat']) for node in member['geometry']]
                            self._rasterize_line(coords, water_mask, buffer_pixels=3)
            
            water_pixels = np.sum(water_mask)
            logger.info(f"Rasterized {water_pixels} water pixels from OSM data")
            
            return water_mask
            
        except requests.exceptions.Timeout:
            logger.error("OpenStreetMap query timed out")
            return water_mask
        except Exception as e:
            logger.error(f"Failed to fetch OSM water data: {e}")
            return water_mask
    
    def _rasterize_line(
        self,
        coords: list[tuple[float, float]],
        mask: np.ndarray,
        buffer_pixels: int = 1,
    ):
        """Rasterize a line of coordinates onto a mask.
        
        Args:
            coords: List of (lon, lat) coordinates
            mask: Output mask to draw on (modified in place)
            buffer_pixels: Width of the line in pixels
        """
        if len(coords) < 2:
            return
        
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            
            # Convert to pixel coordinates
            col1, row1 = ~self.transform * (lon1, lat1)
            col2, row2 = ~self.transform * (lon2, lat2)
            
            # Bresenham's line algorithm
            row1, col1 = int(round(row1)), int(round(col1))
            row2, col2 = int(round(row2)), int(round(col2))
            
            # Draw line with buffer
            rows, cols = [], []
            dx = abs(col2 - col1)
            dy = abs(row2 - row1)
            sx = 1 if col1 < col2 else -1
            sy = 1 if row1 < row2 else -1
            err = dx - dy
            
            row, col = row1, col1
            while True:
                # Add buffered pixels
                for dr in range(-buffer_pixels, buffer_pixels + 1):
                    for dc in range(-buffer_pixels, buffer_pixels + 1):
                        r, c = row + dr, col + dc
                        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                            mask[r, c] = 1
                
                if row == row2 and col == col2:
                    break
                    
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    col += sx
                if e2 < dx:
                    err += dx
                    row += sy
        
    def compute_particle_water_distribution(
        self,
        particles_per_km2: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate water distribution using gravity-based particle simulation.
        
        Drops particles evenly across terrain based on historical rainfall,
        then simulates gravity to pool water in depressions.
        
        Args:
            particles_per_km2: Number of water particles per square kilometer
            
        Returns:
            Tuple of (pooled_water, flowing_water) arrays
        """
        logger.info("Simulating water distribution with particle physics")
        
        rows, cols = self.elevation.shape
        pixel_res_deg = abs(self.transform.a)
        pixel_res_m = pixel_res_deg * 111000
        area_km2 = (rows * cols * pixel_res_m * pixel_res_m) / 1e6
        
        # Calculate particle count based on area and rainfall
        total_particles = int(area_km2 * particles_per_km2)
        rainfall_factor = self.historical_rainfall_mm / 800.0  # Relative to 800mm baseline
        total_particles = int(total_particles * rainfall_factor)
        
        logger.info(f"Dropping {total_particles} water particles across {area_km2:.1f} km² (rainfall: {self.historical_rainfall_mm}mm/yr)")
        
        # Drop particles evenly across terrain
        particle_positions = self._drop_particles_evenly(rows, cols, total_particles)
        
        # Simulate gravity - particles flow downhill and pool
        pooled_water, flowing_water = self._simulate_gravity(particle_positions)
        
        logger.info(f"Water pooled in {np.sum(pooled_water > 0)} cells, flowing in {np.sum(flowing_water > 0)} cells")
        
        return pooled_water, flowing_water

    def _drop_particles_evenly(
        self,
        rows: int,
        cols: int,
        num_particles: int,
    ) -> np.ndarray:
        """Drop particles evenly across terrain using stratified sampling.
        
        Args:
            rows: Grid rows
            cols: Grid columns  
            num_particles: Total particles to drop
            
        Returns:
            Array of particle positions (Nx2: row, col)
        """
        # Create grid of potential drop locations
        grid_density = int(np.sqrt(num_particles) * 1.2)  # Oversample slightly
        row_indices = np.linspace(0, rows - 1, grid_density, dtype=int)
        col_indices = np.linspace(0, cols - 1, grid_density, dtype=int)
        
        # Create meshgrid
        row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing='ij')
        
        # Flatten and add jitter for natural distribution
        positions = np.column_stack([row_grid.ravel(), col_grid.ravel()])
        
        # Add random jitter (±50% of grid spacing)
        grid_spacing = rows / grid_density
        jitter = np.random.uniform(-grid_spacing * 0.5, grid_spacing * 0.5, positions.shape)
        positions = positions + jitter
        
        # Clip to valid range
        positions[:, 0] = np.clip(positions[:, 0], 0, rows - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, cols - 1)
        
        # Sample to exact count
        if len(positions) > num_particles:
            indices = np.random.choice(len(positions), num_particles, replace=False)
            positions = positions[indices]
        
        return positions.astype(np.int32)
    
    def _simulate_gravity(
        self,
        initial_positions: np.ndarray,
        max_iterations: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate particles flowing downhill under gravity.
        
        Args:
            initial_positions: Initial particle positions (Nx2)
            max_iterations: Maximum flow iterations
            
        Returns:
            Tuple of (pooled_water, flowing_water) density arrays
        """
        rows, cols = self.elevation.shape
        
        # Track particle positions over time
        positions = initial_positions.copy().astype(np.float32)
        velocities = np.zeros_like(positions, dtype=np.float32)
        
        # Arrays to track water states
        pooled_water = np.zeros((rows, cols), dtype=np.float32)
        flowing_water = np.zeros((rows, cols), dtype=np.float32)
        flow_history = []  # Track positions over time
        
        # Simulation parameters
        gravity = 1.0
        friction = 0.8
        min_velocity = 0.1
        
        for iteration in range(max_iterations):
            # Get elevation at current positions (bilinear interpolation)
            row_coords = np.clip(positions[:, 0], 0, rows - 1)
            col_coords = np.clip(positions[:, 1], 0, cols - 1)
            
            # Calculate gradient (downhill direction)
            gradients = self._calculate_gradient_at_positions(row_coords, col_coords)
            
            # Apply gravity in downhill direction
            velocities += gradients * gravity
            velocities *= friction  # Apply friction
            
            # Update positions
            new_positions = positions + velocities
            
            # Clip to terrain bounds
            new_positions[:, 0] = np.clip(new_positions[:, 0], 0, rows - 1)
            new_positions[:, 1] = np.clip(new_positions[:, 1], 0, cols - 1)
            
            # Check if particles have stopped (pooled)
            movement = np.linalg.norm(new_positions - positions, axis=1)
            stopped = movement < min_velocity
            
            # Mark flowing particles
            flowing_idx = ~stopped
            if np.any(flowing_idx):
                flowing_rows = new_positions[flowing_idx, 0].astype(int)
                flowing_cols = new_positions[flowing_idx, 1].astype(int)
                np.add.at(flowing_water, (flowing_rows, flowing_cols), 1)
            
            # Mark pooled particles
            if np.any(stopped):
                pooled_rows = new_positions[stopped, 0].astype(int)
                pooled_cols = new_positions[stopped, 1].astype(int)
                np.add.at(pooled_water, (pooled_rows, pooled_cols), 1)
                
                # Remove stopped particles from simulation
                positions = new_positions[~stopped]
                velocities = velocities[~stopped]
            else:
                positions = new_positions
            
            # Store intermediate positions for flow visualization
            if iteration % 10 == 0:
                flow_history.append(positions.copy())
            
            # Stop if all particles have pooled
            if len(positions) == 0:
                logger.info(f"All particles pooled after {iteration} iterations")
                break
        
        # Normalize to density (particles per pixel)
        pooled_water = pooled_water / (pooled_water.max() + 1e-6)
        flowing_water = flowing_water / (flowing_water.max() + 1e-6)
        
        return pooled_water, flowing_water
    
    def _calculate_gradient_at_positions(
        self,
        row_coords: np.ndarray,
        col_coords: np.ndarray,
    ) -> np.ndarray:
        """Calculate terrain gradient at given positions.
        
        Args:
            row_coords: Row coordinates  
            col_coords: Column coordinates
            
        Returns:
            Gradient vectors (Nx2: drow, dcol) pointing downhill
        """
        rows, cols = self.elevation.shape
        gradients = np.zeros((len(row_coords), 2), dtype=np.float32)
        
        for i, (r, c) in enumerate(zip(row_coords, col_coords)):
            r_int, c_int = int(r), int(c)
            
            # Get local 3x3 neighborhood
            r_min = max(0, r_int - 1)
            r_max = min(rows, r_int + 2)
            c_min = max(0, c_int - 1)
            c_max = min(cols, c_int + 2)
            
            if r_max - r_min < 2 or c_max - c_min < 2:
                continue  # Edge case
            
            local_elevation = self.elevation[r_min:r_max, c_min:c_max]
            center_elev = self.elevation[r_int, c_int]
            
            # Find steepest downhill direction
            local_rows, local_cols = local_elevation.shape
            center_r, center_c = 1, 1  # Center of 3x3
            
            max_drop = 0
            best_dir = np.array([0.0, 0.0])
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = center_r + dr, center_c + dc
                    if 0 <= nr < local_rows and 0 <= nc < local_cols:
                        drop = center_elev - local_elevation[nr, nc]
                        if drop > max_drop:
                            max_drop = drop
                            best_dir = np.array([dr, dc], dtype=np.float32)
            
            # Normalize direction
            if max_drop > 0:
                best_dir = best_dir / (np.linalg.norm(best_dir) + 1e-6)
                gradients[i] = best_dir * max_drop * 0.1  # Scale by drop magnitude
        
        return gradients

    def compute_flow_accumulation(self) -> np.ndarray:
        """Legacy method - kept for compatibility.
        
        Use compute_particle_water_distribution() for better results.
        """
        logger.warning("Using legacy flow accumulation - consider compute_particle_water_distribution() instead")
        logger.info("Computing flow accumulation")

        # Fill sinks/depressions in DEM to ensure continuous drainage
        filled_dem = self._fill_sinks(self.elevation)

        # Compute flow direction (D8 - 8 directions)
        flow_dir = self._compute_flow_direction_d8(filled_dem)

        # Accumulate flow
        flow_acc = self._accumulate_flow(flow_dir)

        logger.info(
            f"Flow accumulation: min={np.min(flow_acc)}, "
            f"max={np.max(flow_acc)}, mean={np.mean(flow_acc):.1f}"
        )

        return flow_acc

    def _fill_sinks(self, dem: np.ndarray) -> np.ndarray:
        """Fill sinks and local depressions in DEM.

        Uses a simple filling algorithm where each cell is at least as high
        as the minimum of its neighbors (iteratively).

        Args:
            dem: Digital elevation model

        Returns:
            Filled DEM with no sinks
        """
        filled = dem.copy()
        mask = np.isfinite(filled)

        # Iteratively fill sinks (max 10 iterations to avoid infinite loops)
        for _ in range(10):
            # Get minimum neighbor value for each cell
            min_neighbor = ndimage.minimum_filter(filled, size=3, mode='constant', cval=np.inf)

            # Fill cells that are lower than all neighbors
            needs_fill = (filled < min_neighbor) & mask
            if not np.any(needs_fill):
                break

            filled[needs_fill] = min_neighbor[needs_fill]

        return filled

    def _compute_flow_direction_d8(self, dem: np.ndarray) -> np.ndarray:
        """Compute flow direction using D8 algorithm.

        Each cell flows to the steepest downslope neighbor (8 directions).
        Direction codes: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE, 8=no flow

        Args:
            dem: Filled digital elevation model

        Returns:
            Flow direction array (0-8)
        """
        rows, cols = dem.shape
        flow_dir = np.full((rows, cols), 8, dtype=np.int8)  # 8 = no flow

        # D8 neighbor offsets (row, col) and their distances
        neighbors = [
            (0, 1, 1.0),    # E
            (1, 1, 1.414),  # SE
            (1, 0, 1.0),    # S
            (1, -1, 1.414), # SW
            (0, -1, 1.0),   # W
            (-1, -1, 1.414),# NW
            (-1, 0, 1.0),   # N
            (-1, 1, 1.414), # NE
        ]

        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                center_elev = dem[row, col]
                max_slope = -np.inf
                steepest_dir = 8

                for direction, (dr, dc, dist) in enumerate(neighbors):
                    neighbor_elev = dem[row + dr, col + dc]
                    slope = (center_elev - neighbor_elev) / dist

                    if slope > max_slope:
                        max_slope = slope
                        steepest_dir = direction

                # Only assign direction if there's downslope flow
                if max_slope > 0:
                    flow_dir[row, col] = steepest_dir

        return flow_dir

    def _accumulate_flow(self, flow_dir: np.ndarray) -> np.ndarray:
        """Accumulate flow based on flow directions.

        Counts how many upstream cells contribute flow to each cell.

        Args:
            flow_dir: Flow direction array from D8 algorithm

        Returns:
            Flow accumulation array (upstream cell count)
        """
        rows, cols = flow_dir.shape
        flow_acc = np.ones((rows, cols), dtype=np.float32)  # Use float to normalize later

        # D8 direction offsets
        dir_offsets = [
            (0, 1),   # E
            (1, 1),   # SE
            (1, 0),   # S
            (1, -1),  # SW
            (0, -1),  # W
            (-1, -1), # NW
            (-1, 0),  # N
            (-1, 1),  # NE
        ]

        # Process cells in multiple passes (limit to 15 iterations to prevent overflow)
        for iteration in range(15):
            changed = False
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    direction = flow_dir[row, col]
                    if direction < 8:  # Has downslope flow
                        dr, dc = dir_offsets[direction]
                        target_row = row + dr
                        target_col = col + dc

                        if 0 <= target_row < rows and 0 <= target_col < cols:
                            old_val = flow_acc[target_row, target_col]
                            # Cap contribution to prevent runaway accumulation
                            contribution = min(flow_acc[row, col], 1e6)
                            flow_acc[target_row, target_col] += contribution
                            if abs(flow_acc[target_row, target_col] - old_val) > 0.01:
                                changed = True

            if not changed:
                break

        # Normalize using log scale (streams are where accumulation is high)
        flow_log = np.log1p(flow_acc)  # log(1 + x)
        flow_norm = (flow_log - flow_log.min()) / (flow_log.max() - flow_log.min() + 1e-10)
        
        return flow_norm

    def identify_streams(
        self,
        flow_acc: np.ndarray,
        threshold_percentile: float = 98.0,
    ) -> np.ndarray:
        """Identify stream cells based on flow accumulation threshold.

        Args:
            flow_acc: Flow accumulation array (normalized 0-1)
            threshold_percentile: Percentile threshold for stream identification
                                 (higher = only major streams)

        Returns:
            Binary mask of stream cells (1=stream, 0=not stream)
        """
        threshold = np.percentile(flow_acc, threshold_percentile)
        streams = (flow_acc >= threshold).astype(np.uint8)

        logger.info(
            f"Identified streams: threshold={threshold:.3f} (normalized), "
            f"stream pixels={np.sum(streams)}"
        )

        return streams

    def compute_water_proximity(
        self,
        streams: np.ndarray,
        max_distance_m: float = 200.0,
    ) -> np.ndarray:
        """Compute distance to nearest water feature.

        Args:
            streams: Binary stream mask
            max_distance_m: Maximum distance to consider (meters)

        Returns:
            Distance array (meters, 0 at streams)
        """
        logger.info("Computing distance to water features")

        # Convert max distance from meters to pixels
        pixel_res_deg = abs(self.transform.a)
        pixel_res_m = pixel_res_deg * 111000  # Approximate meters per degree
        max_distance_pixels = max_distance_m / pixel_res_m

        # Compute Euclidean distance transform
        distance_pixels = ndimage.distance_transform_edt(~streams.astype(bool))

        # Clip to max distance and convert to meters
        distance_m = np.clip(distance_pixels * pixel_res_m, 0, max_distance_m)

        logger.info(
            f"Water proximity: mean={np.mean(distance_m):.1f}m, "
            f"max={np.max(distance_m):.1f}m"
        )

        return distance_m

    def save_hydrology_outputs(
        self,
        output_dir: Path,
        flow_acc: np.ndarray | None = None,
        streams: np.ndarray | None = None,
        water_distance: np.ndarray | None = None,
        pooled_water: np.ndarray | None = None,
        flowing_water: np.ndarray | None = None,
    ) -> None:
        """Save hydrology analysis outputs as GeoTIFFs.

        Args:
            output_dir: Directory to save outputs
            flow_acc: Flow accumulation or combined water array
            streams: Stream mask
            water_distance: Distance to water array
            pooled_water: Particle simulation - pooled water density
            flowing_water: Particle simulation - flowing water density
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "driver": "GTiff",
            "height": self.elevation.shape[0],
            "width": self.elevation.shape[1],
            "count": 1,
            "dtype": rasterio.float32,
            "crs": self.crs,
            "transform": self.transform,
            "compress": "lzw",
        }

        if flow_acc is not None:
            with rasterio.open(output_dir / "flow_accumulation.tif", "w", **meta) as dst:
                dst.write(flow_acc.astype(np.float32), 1)
            logger.debug("Saved flow_accumulation to %s", output_dir / "flow_accumulation.tif")

        if streams is not None:
            with rasterio.open(output_dir / "streams.tif", "w", **meta) as dst:
                dst.write(streams.astype(np.float32), 1)
            logger.debug("Saved streams to %s", output_dir / "streams.tif")

        if water_distance is not None:
            with rasterio.open(output_dir / "water_distance.tif", "w", **meta) as dst:
                dst.write(water_distance.astype(np.float32), 1)
            logger.debug("Saved water_distance to %s", output_dir / "water_distance.tif")
        
        if pooled_water is not None:
            with rasterio.open(output_dir / "pooled_water.tif", "w", **meta) as dst:
                dst.write(pooled_water.astype(np.float32), 1)
            logger.debug("Saved pooled_water to %s", output_dir / "pooled_water.tif")
        
        if flowing_water is not None:
            with rasterio.open(output_dir / "flowing_water.tif", "w", **meta) as dst:
                dst.write(flowing_water.astype(np.float32), 1)
            logger.debug("Saved flowing_water to %s", output_dir / "flowing_water.tif")
