"""Test fixtures and utilities."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def create_mock_dem(output_path: Path, shape=(100, 100), bounds=None):
    """Create a mock DEM for testing.

    Args:
        output_path: Path to save mock DEM
        shape: Array shape (height, width)
        bounds: Geographic bounds (minx, miny, maxx, maxy)

    Returns:
        Path to created DEM
    """
    if bounds is None:
        bounds = (-76.5, 42.4, -76.4, 42.5)

    # Create synthetic elevation with some features
    rows, cols = shape
    y, x = np.ogrid[0:rows, 0:cols]

    # Create a ridge running diagonally
    ridge = 100 * np.exp(-((y - x) ** 2) / 1000)

    # Create a valley
    valley = -50 * np.exp(-((y - rows / 2) ** 2 + (x - cols / 2) ** 2) / 500)

    # Add random noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5, shape)

    # Combine features
    elevation = 500 + ridge + valley + noise

    # Create transform
    transform = from_bounds(*bounds, cols, rows)

    # Write GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elevation.astype(np.float32), 1)

    return output_path
