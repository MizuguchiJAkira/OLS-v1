"""Tests for camera placement module."""

import numpy as np
import pytest

from land_audit.cameras import CameraPlacement, RecommendationZone
from land_audit.terrain import TerrainAnalyzer
from tests.conftest import create_mock_dem


@pytest.fixture
def camera_placer(tmp_path):
    """Create camera placement with mock data."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    slope = analyzer.compute_slope()
    aspect = analyzer.compute_aspect()

    # Create simple score surfaces
    pinch_score = np.random.rand(*slope.shape).astype(np.float32)
    bedding_score = np.random.rand(*slope.shape).astype(np.float32)

    return CameraPlacement(
        pinch_score=pinch_score,
        bedding_score=bedding_score,
        slope=slope,
        aspect=aspect,
        transform=analyzer.transform,
        crs=str(analyzer.crs),
    )


def test_generate_corridor_zones(camera_placer):
    """Test corridor zone generation."""
    zones = camera_placer.generate_corridor_zones(num_cameras=6)

    assert isinstance(zones, list)
    assert all(isinstance(zone, RecommendationZone) for zone in zones)


def test_zone_properties(camera_placer):
    """Test recommendation zone properties."""
    zones = camera_placer.generate_corridor_zones(num_cameras=3)

    for zone in zones:
        assert zone.zone_type == "corridor"
        assert zone.priority in ["high", "medium", "low"]
        assert 0 <= zone.avg_score <= 1
        assert zone.area_m2 > 0
        assert zone.bounds_polygon is not None


def test_camera_allocation(camera_placer):
    """Test that cameras are allocated to zones."""
    zones = camera_placer.generate_corridor_zones(num_cameras=6)

    total_cameras = sum(zone.recommended_cameras for zone in zones)
    # Should allocate up to the requested number
    assert total_cameras <= 6


def test_zone_to_dict(camera_placer):
    """Test RecommendationZone dictionary conversion."""
    zones = camera_placer.generate_corridor_zones(num_cameras=3)

    if zones:
        zone_dict = zones[0].to_dict()

        assert "zone_id" in zone_dict
        assert "type" in zone_dict
        assert "avg_score" in zone_dict
        assert "area_km2" in zone_dict
        assert "priority" in zone_dict
        assert "description" in zone_dict


def test_export_zones_geojson(camera_placer, tmp_path):
    """Test GeoJSON export for zones."""
    zones = camera_placer.generate_corridor_zones(num_cameras=3)
    output_path = tmp_path / "zones.geojson"

    camera_placer.export_zones_geojson(zones, output_path)

    assert output_path.exists()

    # Verify GeoJSON structure
    import json

    with open(output_path) as f:
        data = json.load(f)

    assert "type" in data
    assert "features" in data
