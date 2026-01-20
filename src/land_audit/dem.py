"""DEM acquisition and caching using py3dep."""

import hashlib
import logging
from pathlib import Path

import py3dep
import rasterio
from shapely.geometry import box

logger = logging.getLogger(__name__)


class DEMManager:
    """Manages DEM data fetching and caching."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize DEM manager with optional cache directory.

        Args:
            cache_dir: Directory to cache DEM files. Defaults to ./data/cache
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, bounds: tuple[float, float, float, float], resolution: int) -> str:
        """Generate cache key from bounds and resolution.

        Args:
            bounds: (minx, miny, maxx, maxy) in EPSG:4326
            resolution: DEM resolution in meters

        Returns:
            MD5 hash string
        """
        key_str = f"{bounds[0]:.6f}_{bounds[1]:.6f}_{bounds[2]:.6f}_{bounds[3]:.6f}_{resolution}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def fetch_dem(
        self,
        bounds: tuple[float, float, float, float],
        resolution: int = 10,
        crs: str = "EPSG:4326",
    ) -> Path:
        """Fetch or retrieve cached DEM for given bounds.

        Args:
            bounds: (minx, miny, maxx, maxy) bounding box
            resolution: DEM resolution in meters (10 or 30 recommended)
            crs: Coordinate reference system of bounds

        Returns:
            Path to DEM GeoTIFF file

        Raises:
            RuntimeError: If DEM fetch fails
        """
        cache_key = self._get_cache_key(bounds, resolution)
        cache_path = self.cache_dir / f"dem_{cache_key}.tif"

        if cache_path.exists():
            logger.info(f"Using cached DEM: {cache_path}")
            return cache_path

        logger.info(f"Fetching DEM for bounds {bounds} at {resolution}m resolution")

        try:
            # Create bounding box geometry
            bbox = box(*bounds)

            # Fetch DEM from py3dep
            # py3dep returns data in the target CRS, we'll keep it in WGS84
            dem_data = py3dep.get_map(
                "DEM",
                bbox,
                resolution=resolution,
                geo_crs=crs,
                crs=crs,  # Keep in WGS84 for simplicity
            )

            # Save to cache
            self._save_dem(dem_data, cache_path)
            logger.info(f"DEM cached to {cache_path}")

            return cache_path

        except Exception as e:
            logger.error(f"Failed to fetch DEM: {e}")
            raise RuntimeError(f"DEM acquisition failed: {e}") from e

    def _save_dem(self, dem_data, output_path: Path):
        """Save xarray DataArray DEM to GeoTIFF.

        Args:
            dem_data: xarray DataArray from py3dep
            output_path: Path to save GeoTIFF
        """
        # Extract data and spatial reference
        elevation = dem_data.values
        transform = dem_data.rio.transform()
        crs = dem_data.rio.crs

        # Write to GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype=elevation.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(elevation, 1)
            dst.set_band_description(1, "elevation")

    def get_dem_info(self, dem_path: Path) -> dict:
        """Get metadata about a DEM file.

        Args:
            dem_path: Path to DEM GeoTIFF

        Returns:
            Dictionary with bounds, resolution, CRS, etc.
        """
        with rasterio.open(dem_path) as src:
            return {
                "bounds": src.bounds,
                "crs": src.crs.to_string(),
                "shape": src.shape,
                "resolution": src.res,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
            }
