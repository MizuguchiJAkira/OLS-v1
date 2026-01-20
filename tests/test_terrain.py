"""Tests for terrain analysis module."""

import numpy as np

from land_audit.terrain import TerrainAnalyzer
from tests.conftest import create_mock_dem


def test_terrain_analyzer_init(tmp_path):
    """Test TerrainAnalyzer initialization."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    assert analyzer.elevation.shape == (100, 100)
    assert analyzer.resolution > 0
    assert analyzer.crs is not None


def test_compute_slope(tmp_path):
    """Test slope computation."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    slope = analyzer.compute_slope()

    assert slope.shape == analyzer.elevation.shape
    assert slope.min() >= 0
    assert slope.max() <= 90
    assert not np.isnan(slope).any()


def test_compute_aspect(tmp_path):
    """Test aspect computation."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    aspect = analyzer.compute_aspect()

    assert aspect.shape == analyzer.elevation.shape
    assert aspect.min() >= 0
    assert aspect.max() <= 360


def test_compute_hillshade(tmp_path):
    """Test hillshade computation."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    hillshade = analyzer.compute_hillshade()

    assert hillshade.shape == analyzer.elevation.shape
    assert hillshade.dtype == np.uint8
    assert hillshade.min() >= 0
    assert hillshade.max() <= 255


def test_compute_tpi(tmp_path):
    """Test TPI computation."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    tpi = analyzer.compute_tpi(window_size=5)

    assert tpi.shape == analyzer.elevation.shape
    # TPI should have both positive and negative values
    assert tpi.min() < 0
    assert tpi.max() > 0


def test_compute_ruggedness(tmp_path):
    """Test ruggedness computation."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    ruggedness = analyzer.compute_ruggedness(window_size=3)

    assert ruggedness.shape == analyzer.elevation.shape
    assert ruggedness.min() >= 0
    assert not np.isnan(ruggedness).any()


def test_save_raster(tmp_path):
    """Test raster saving."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    output_path = tmp_path / "slope.tif"
    slope = analyzer.compute_slope(output_path=output_path)

    assert output_path.exists()

    # Verify saved file
    import rasterio

    with rasterio.open(output_path) as src:
        saved_slope = src.read(1)
        np.testing.assert_array_almost_equal(slope, saved_slope, decimal=5)
