"""Terrain derivatives computation from DEM."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

logger = logging.getLogger(__name__)


class TerrainAnalyzer:
    """Computes terrain derivatives from DEM."""

    def __init__(self, dem_path: Path):
        """Initialize terrain analyzer with DEM.

        Args:
            dem_path: Path to DEM GeoTIFF file
        """
        self.dem_path = dem_path
        with rasterio.open(dem_path) as src:
            self.elevation = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.resolution = src.res[0]  # Assume square pixels

        # Mask nodata values
        if self.nodata is not None:
            self.elevation = np.ma.masked_equal(self.elevation, self.nodata)

    def compute_slope(self, output_path: Path | None = None) -> np.ndarray:
        """Compute slope in degrees.

        Uses Horn's method (3x3 kernel) for slope calculation.
        Handles geographic coordinates (lat/lon) by converting to meters.

        Args:
            output_path: Optional path to save slope raster

        Returns:
            Slope array in degrees
        """
        logger.info("Computing slope")

        # Get resolution in appropriate units
        res_x, res_y = self.resolution, self.resolution
        
        # If CRS is geographic (degrees), convert to meters
        if self.crs and self.crs.is_geographic:
            # Approximate conversion at this latitude
            # 1 degree latitude ≈ 111,000 meters
            # 1 degree longitude ≈ 111,000 * cos(latitude) meters
            center_lat = (self.transform.f + self.transform.f + res_y * self.elevation.shape[0]) / 2
            res_x_m = res_x * 111000 * np.cos(np.radians(center_lat))
            res_y_m = abs(res_y * 111000)
            logger.debug(f"Converting geographic resolution: {res_x:.6f}° x {res_y:.6f}° -> {res_x_m:.1f}m x {res_y_m:.1f}m")
        else:
            res_x_m = res_x
            res_y_m = res_y

        # Compute gradients using Sobel filters
        # Sobel kernel has a factor of 8 in the denominator for proper weighting
        dx = ndimage.sobel(self.elevation, axis=1) / (8 * res_x_m)
        dy = ndimage.sobel(self.elevation, axis=0) / (8 * res_y_m)

        # Compute slope in radians then convert to degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)

        if output_path:
            self._save_raster(slope_deg, output_path, "slope_degrees")

        return slope_deg

    def compute_aspect(self, output_path: Path | None = None) -> np.ndarray:
        """Compute aspect in degrees (0-360, 0=North, clockwise).

        Handles geographic coordinates (lat/lon) by converting to meters.

        Args:
            output_path: Optional path to save aspect raster

        Returns:
            Aspect array in degrees
        """
        logger.info("Computing aspect")

        # Get resolution in appropriate units
        res_x, res_y = self.resolution, self.resolution
        
        # If CRS is geographic (degrees), convert to meters
        if self.crs and self.crs.is_geographic:
            center_lat = (self.transform.f + self.transform.f + res_y * self.elevation.shape[0]) / 2
            res_x_m = res_x * 111000 * np.cos(np.radians(center_lat))
            res_y_m = abs(res_y * 111000)
        else:
            res_x_m = res_x
            res_y_m = res_y

        # Compute gradients
        dx = ndimage.sobel(self.elevation, axis=1) / (8 * res_x_m)
        dy = ndimage.sobel(self.elevation, axis=0) / (8 * res_y_m)

        # Compute aspect in radians
        aspect_rad = np.arctan2(-dx, dy)

        # Convert to degrees (0-360, 0=North)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (90 - aspect_deg) % 360

        if output_path:
            self._save_raster(aspect_deg, output_path, "aspect_degrees")

        return aspect_deg

    def compute_hillshade(
        self,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        output_path: Path | None = None,
    ) -> np.ndarray:
        """Compute hillshade for visualization.

        Args:
            azimuth: Light source azimuth in degrees (0-360)
            altitude: Light source altitude in degrees (0-90)
            output_path: Optional path to save hillshade raster

        Returns:
            Hillshade array (0-255)
        """
        logger.info(f"Computing hillshade (azimuth={azimuth}, altitude={altitude})")

        # Compute gradients
        dx = ndimage.sobel(self.elevation, axis=1) / (8 * self.resolution)
        dy = ndimage.sobel(self.elevation, axis=0) / (8 * self.resolution)

        # Compute slope and aspect
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)

        # Convert azimuth and altitude to radians
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)

        # Compute hillshade
        shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(
            slope
        ) * np.cos(azimuth_rad - aspect)

        # Scale to 0-255
        hillshade = ((shaded + 1) / 2 * 255).astype(np.uint8)

        if output_path:
            self._save_raster(hillshade, output_path, "hillshade", dtype=np.uint8)

        return hillshade

    def compute_tpi(self, window_size: int = 5, output_path: Path | None = None) -> np.ndarray:
        """Compute Topographic Position Index.

        TPI = elevation - mean(neighborhood elevation)
        Positive values indicate ridges, negative values indicate valleys.

        Args:
            window_size: Size of neighborhood window (odd number)
            output_path: Optional path to save TPI raster

        Returns:
            TPI array
        """
        logger.info(f"Computing TPI (window_size={window_size})")

        # Compute mean of neighborhood
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        mean_elevation = ndimage.convolve(self.elevation, kernel, mode="reflect")

        # TPI is difference from mean
        tpi = self.elevation - mean_elevation

        if output_path:
            self._save_raster(tpi, output_path, "tpi")

        return tpi

    def compute_ruggedness(
        self, window_size: int = 3, output_path: Path | None = None
    ) -> np.ndarray:
        """Compute terrain ruggedness (local elevation variability).

        Uses standard deviation of elevation in neighborhood.

        Args:
            window_size: Size of neighborhood window (odd number)
            output_path: Optional path to save ruggedness raster

        Returns:
            Ruggedness array
        """
        logger.info(f"Computing ruggedness (window_size={window_size})")

        # Use generic filter to compute local std dev
        ruggedness = ndimage.generic_filter(
            self.elevation,
            np.std,
            size=window_size,
            mode="reflect",
        )

        if output_path:
            self._save_raster(ruggedness, output_path, "ruggedness")

        return ruggedness

    def _save_raster(
        self,
        data: np.ndarray,
        output_path: Path,
        description: str,
        dtype: np.dtype | None = None,
    ):
        """Save numpy array as GeoTIFF.

        Args:
            data: Array to save
            output_path: Output path
            description: Band description
            dtype: Output data type (defaults to data.dtype)
        """
        if dtype is None:
            dtype = data.dtype

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs=self.crs,
            transform=self.transform,
            compress="lzw",
        ) as dst:
            dst.write(data.astype(dtype), 1)
            dst.set_band_description(1, description)

        logger.debug(f"Saved {description} to {output_path}")
