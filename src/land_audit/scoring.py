"""Heuristic scoring for bedding zones and pinch points."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


class HeuristicScorer:
    """Computes heuristic scores for wildlife habitat features."""

    def __init__(
        self,
        slope: np.ndarray,
        aspect: np.ndarray,
        tpi: np.ndarray,
        ruggedness: np.ndarray,
        transform,
        crs,
        water_mask: np.ndarray | None = None,
    ):
        """Initialize scorer with terrain derivatives.

        Args:
            slope: Slope array in degrees
            aspect: Aspect array in degrees (0-360)
            tpi: Topographic Position Index array
            ruggedness: Terrain ruggedness array
            transform: Rasterio affine transform
            crs: Coordinate reference system
            water_mask: Optional binary mask of water areas (1=water, 0=land)
        """
        self.slope = slope
        self.aspect = aspect
        self.tpi = tpi
        self.ruggedness = ruggedness
        self.transform = transform
        self.crs = crs
        self.water_mask = water_mask

        # Normalize inputs for scoring
        self.slope_norm = self._normalize(slope)
        self.tpi_norm = self._normalize(tpi)
        self.ruggedness_norm = self._normalize(ruggedness)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 range.

        Args:
            arr: Input array

        Returns:
            Normalized array
        """
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min == 0:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    def compute_bedding_score(self, output_path: Path | None = None) -> np.ndarray:
        """Compute bedding zone score.

        Good bedding areas prioritize moderate slopes with terrain variation:
        - Moderate slope (5-20 degrees preferred) - provides drainage and visibility
        - Mid-slope position (TPI near 0) - avoid exposed ridges and wet valleys
        - South/SE facing aspect (warmth) - secondary factor
        - Terrain variation (ruggedness) - indicates habitat edges

        Slope is the PRIMARY factor - even slight hills in flat areas score well.
        TPI penalizes ridgetops and valley bottoms - mid-slopes are preferred.

        Args:
            output_path: Optional path to save score raster

        Returns:
            Bedding score array (0-1)
        """
        logger.info("Computing bedding zone scores")

        # Slope preference: Emphasize ANY slope as better than flat
        # Even 2-3 degrees is notably better than 0 degrees
        # Peak scoring at 8-15 degrees (moderate hills)
        slope_score = np.zeros_like(self.slope, dtype=np.float32)
        
        # Very flat (0-2°): Low score
        mask = self.slope < 2
        slope_score[mask] = 0.2 + (self.slope[mask] / 2.0) * 0.3  # 0.2 to 0.5
        
        # Gentle slopes (2-8°): Good score, increasing
        mask = (self.slope >= 2) & (self.slope < 8)
        slope_score[mask] = 0.5 + ((self.slope[mask] - 2) / 6.0) * 0.3  # 0.5 to 0.8
        
        # Ideal slopes (8-15°): Best score
        mask = (self.slope >= 8) & (self.slope < 15)
        slope_score[mask] = 0.8 + ((self.slope[mask] - 8) / 7.0) * 0.2  # 0.8 to 1.0
        
        # Moderate steep (15-25°): Still good
        mask = (self.slope >= 15) & (self.slope < 25)
        slope_score[mask] = 1.0 - ((self.slope[mask] - 15) / 10.0) * 0.4  # 1.0 to 0.6
        
        # Too steep (>25°): Declining score
        mask = self.slope >= 25
        slope_score[mask] = 0.6 * np.exp(-(self.slope[mask] - 25) / 20.0)  # Exponential decay

        # TPI preference: Favor mid-slope positions (TPI near 0)
        # Penalize both ridges (positive TPI) and valleys (negative TPI)
        # Wildlife prefer sheltered mid-slopes over exposed ridges or wet valleys
        tpi_score = np.ones_like(self.tpi_norm, dtype=np.float32)
        
        # TPI normalized to 0-1, with 0.5 being neutral (mid-slope)
        # Deviation from mid-slope reduces score
        tpi_deviation = np.abs(self.tpi_norm - 0.5) * 2.0  # 0 at mid-slope, 1 at extremes
        tpi_score = 1.0 - (tpi_deviation * 0.6)  # Max 60% penalty at extremes
        tpi_score = np.clip(tpi_score, 0.4, 1.0)  # Keep minimum at 0.4

        # Aspect preference: South to SE (135-225 degrees)
        # Much less important than slope and position
        aspect_diff = np.minimum(
            np.abs(self.aspect - 180),
            360 - np.abs(self.aspect - 180),
        )
        aspect_score = 1.0 - (aspect_diff / 180.0) * 0.3  # Max 30% penalty

        # Ruggedness: Prefer some texture (edge habitat)
        # Higher ruggedness = more habitat complexity
        ruggedness_score = np.clip(self.ruggedness_norm * 1.5, 0, 1)

        # Weighted combination - slope dominates, TPI prevents ridge/valley placement
        bedding_score = (
            0.60 * slope_score +      # Dominant factor
            0.15 * tpi_score +        # Avoid ridges and valleys
            0.10 * aspect_score +     # Warmth preference
            0.15 * ruggedness_score   # Habitat complexity
        )

        bedding_score = np.clip(bedding_score, 0, 1)

        if output_path:
            self._save_score(bedding_score, output_path, "bedding_score")

        return bedding_score

    def compute_pinch_score(
        self,
        elevation: np.ndarray,
        resolution: float,
        output_path: Path | None = None,
    ) -> np.ndarray:
        """Compute pinch point score based on terrain convergence and steep slopes.

        Pinch points are narrow travel corridors created by:
        - Steep surrounding slopes (ridges, valleys)
        - Terrain convergence (saddles, passes, valley bottoms)
        - High local relief (TPI extremes)

        Slope is PRIMARY - steep areas force movement into predictable corridors.

        Args:
            elevation: Elevation array
            resolution: Pixel resolution in meters
            output_path: Optional path to save score raster

        Returns:
            Pinch score array (0-1)
        """
        logger.info("Computing pinch point scores")

        # Slope emphasis: Steeper = better for pinch points
        # Want 20-40° slopes (significant terrain barriers)
        slope_score = np.zeros_like(self.slope, dtype=np.float32)
        
        # Flat/gentle (0-10°): Poor for pinch points
        mask = self.slope < 10
        slope_score[mask] = (self.slope[mask] / 10.0) * 0.3  # 0 to 0.3
        
        # Moderate (10-20°): Getting better
        mask = (self.slope >= 10) & (self.slope < 20)
        slope_score[mask] = 0.3 + ((self.slope[mask] - 10) / 10.0) * 0.4  # 0.3 to 0.7
        
        # Steep (20-35°): Ideal for creating pinch points
        mask = (self.slope >= 20) & (self.slope < 35)
        slope_score[mask] = 0.7 + ((self.slope[mask] - 20) / 15.0) * 0.3  # 0.7 to 1.0
        
        # Very steep (>35°): Maximum pinch effect
        mask = self.slope >= 35
        slope_score[mask] = 1.0  # Maximum score

        # TPI: Extreme values indicate ridges (+) or valleys (-)
        # Both create travel corridors
        tpi_score = np.abs(self.tpi_norm - 0.5) * 2.0  # 0 at center, 1 at extremes
        tpi_score = np.clip(tpi_score, 0, 1)

        # Ruggedness: Higher = more terrain complexity and funneling
        ruggedness_score = self.ruggedness_norm

        # Terrain convergence: Identify valley/saddle features
        # Use local slope variation - low local variation in high-slope areas = corridors
        slope_variation = ndimage.generic_filter(self.slope, np.std, size=5)
        slope_var_norm = self._normalize(slope_variation)
        convergence_score = 1.0 - slope_var_norm  # Low variation = corridor

        # Focal maximum to identify concentrated steep areas (pinch zones)
        steep_zones = ndimage.maximum_filter(slope_score, size=7)
        
        # Weighted combination - slope and steep zones dominate
        pinch_score = (
            0.50 * slope_score +           # Primary: steepness
            0.25 * steep_zones +           # Concentrated steep areas
            0.15 * tpi_score +             # Ridge/valley features
            0.10 * ruggedness_score        # General terrain complexity
        )

        pinch_score = np.clip(pinch_score, 0, 1)

        # Apply focal filter to smooth and identify pinch corridors
        pinch_score = ndimage.maximum_filter(pinch_score, size=5)
        pinch_score = np.clip(pinch_score, 0, 1)
        
        # CORRIDOR DENSITY ENHANCEMENT
        # Boost scores in areas with high density of corridor pixels
        # This makes scores correlate with the NUMBER of nearby high-traffic corridor points
        corridor_threshold = 0.6  # Pixels above this are "corridor"
        high_corridors = (pinch_score > corridor_threshold).astype(np.float32)

        # Count corridor pixels in 30x30 window (~300m radius)
        corridor_density = ndimage.uniform_filter(high_corridors, size=30, mode='constant')
        corridor_density_norm = self._normalize(corridor_density)

        # Boost base score by corridor density (up to 30% bonus)
        density_boost = corridor_density_norm * 0.3
        pinch_score = pinch_score + (pinch_score * density_boost)
        pinch_score = np.clip(pinch_score, 0, 1)

        logger.info(f"Corridor density: max={corridor_density.max():.3f}, mean={corridor_density.mean():.3f}")
        logger.info(f"High-density areas boosted by up to {density_boost.max()*100:.1f}%")

        # WATER EXCLUSION: Zero out pinch scores in water areas
        # Wildlife corridors should not cross through water bodies
        if self.water_mask is not None:
            water_pixels_before = np.sum(pinch_score[self.water_mask > 0] > 0.3)
            # Zero out scores in water areas
            pinch_score[self.water_mask > 0] = 0.0
            # Also create a buffer around water (dilate water mask by ~20m)
            water_buffer = ndimage.binary_dilation(self.water_mask > 0, iterations=2)
            # Reduce scores near water (not eliminate, just reduce)
            near_water = water_buffer & (self.water_mask == 0)
            pinch_score[near_water] *= 0.3  # 70% reduction near water
            logger.info(f"Water exclusion: zeroed {water_pixels_before} corridor pixels in water areas")

        if output_path:
            self._save_score(pinch_score, output_path, "pinch_score")

        return pinch_score

    def compute_water_score(
        self,
        water_distance: np.ndarray,
        output_path: Path | None = None,
    ) -> np.ndarray:
        """Compute water proximity score for wildlife camera placement.

        Wildlife congregate near water sources for drinking. Cameras near streams,
        ponds, and water crossings capture high activity.

        Args:
            water_distance: Distance to nearest water feature (meters)
            output_path: Optional path to save score raster

        Returns:
            Water proximity score array (0-1, higher near water)
        """
        logger.info("Computing water proximity scores")

        # Score decays with distance from water
        # 0-20m: 1.0 score (immediate vicinity - water crossings)
        # 20-50m: 1.0-0.7 score (near water - high activity)
        # 50-100m: 0.7-0.4 score (moderate proximity)
        # 100-200m: 0.4-0.0 score (distant from water)
        # >200m: 0.0 score (no water influence)

        water_score = np.zeros_like(water_distance, dtype=np.float32)

        # Immediate vicinity (0-20m)
        mask = water_distance <= 20
        water_score[mask] = 1.0

        # Near water (20-50m)
        mask = (water_distance > 20) & (water_distance <= 50)
        water_score[mask] = 1.0 - ((water_distance[mask] - 20) / 30) * 0.3  # 1.0 to 0.7

        # Moderate proximity (50-100m)
        mask = (water_distance > 50) & (water_distance <= 100)
        water_score[mask] = 0.7 - ((water_distance[mask] - 50) / 50) * 0.3  # 0.7 to 0.4

        # Distant (100-200m)
        mask = (water_distance > 100) & (water_distance <= 200)
        water_score[mask] = 0.4 - ((water_distance[mask] - 100) / 100) * 0.4  # 0.4 to 0.0

        # Beyond 200m: 0.0 (already initialized to 0)

        logger.info(
            f"Water scores: min={np.min(water_score):.3f}, "
            f"max={np.max(water_score):.3f}, mean={np.mean(water_score):.3f}"
        )

        if output_path:
            self._save_score(water_score, output_path, "water_score")

        return water_score

    def _create_edge_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Create mask of edge pixels.

        Args:
            shape: Array shape

        Returns:
            Boolean mask with True on edges
        """
        mask = np.zeros(shape, dtype=bool)
        # Mark outer 3-pixel border as edge
        mask[:3, :] = True
        mask[-3:, :] = True
        mask[:, :3] = True
        mask[:, -3:] = True
        return mask

    def _cost_distance(self, cost: np.ndarray, start_point: tuple[int, int]) -> np.ndarray:
        """Compute cost-weighted distance from start point.

        Simplified version using Euclidean distance weighted by cost.

        Args:
            cost: Cost surface array
            start_point: (row, col) starting point

        Returns:
            Cost-distance array
        """
        # Create binary mask with start point
        start_mask = np.zeros_like(cost, dtype=bool)
        start_mask[start_point[0], start_point[1]] = True

        # Compute Euclidean distance
        dist = distance_transform_edt(~start_mask)

        # Weight by cost
        cost_dist = dist * cost

        return cost_dist

    def _save_score(self, score: np.ndarray, output_path: Path, description: str):
        """Save score raster as GeoTIFF.

        Args:
            score: Score array
            output_path: Output path
            description: Band description
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=score.shape[0],
            width=score.shape[1],
            count=1,
            dtype=np.float32,
            crs=self.crs,
            transform=self.transform,
            compress="lzw",
        ) as dst:
            dst.write(score.astype(np.float32), 1)
            dst.set_band_description(1, description)

        logger.debug(f"Saved {description} to {output_path}")
