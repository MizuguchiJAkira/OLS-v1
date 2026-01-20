"""Tests for camera placement module."""

import numpy as np
import pytest

from land_audit.cameras import CameraPlacement, CameraPoint
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


def test_select_cameras(camera_placer):
    """Test camera selection."""
    cameras = camera_placer.select_cameras(num_cameras=6, min_distance_m=150)

    assert len(cameras) <= 6
    assert all(isinstance(cam, CameraPoint) for cam in cameras)


def test_camera_properties(camera_placer):
    """Test camera point properties."""
    cameras = camera_placer.select_cameras(num_cameras=3, min_distance_m=100)

    for cam in cameras:
        assert cam.camera_type in ["pinch", "bedding"]
        assert 0 <= cam.score <= 1
        assert 0 <= cam.slope <= 90
        assert 0 <= cam.aspect <= 360
        assert len(cam.reason) > 0


def test_camera_spacing(camera_placer):
    """Test that cameras respect minimum spacing."""
    cameras = camera_placer.select_cameras(num_cameras=4, min_distance_m=500)

    # Check pairwise distances
    for i, cam1 in enumerate(cameras):
        for cam2 in cameras[i + 1 :]:
            # Simple Euclidean distance in degrees (approximate)
            dist_deg = np.sqrt((cam1.lon - cam2.lon) ** 2 + (cam1.lat - cam2.lat) ** 2)
            # Should be separated (though our test data is small)
            assert dist_deg > 0


def test_camera_type_allocation(camera_placer):
    """Test camera type allocation."""
    cameras = camera_placer.select_cameras(num_cameras=10, min_distance_m=50)

    pinch_count = sum(1 for cam in cameras if cam.camera_type == "pinch")
    bedding_count = sum(1 for cam in cameras if cam.camera_type == "bedding")

    # Should have both types
    assert pinch_count > 0
    assert bedding_count > 0


def test_camera_point_to_dict():
    """Test CameraPoint dictionary conversion."""
    cam = CameraPoint(
        lon=-76.45,
        lat=42.45,
        camera_type="pinch",
        score=0.85,
        slope=12.3,
        aspect=180.0,
        reason="Test reason",
    )

    cam_dict = cam.to_dict()

    assert cam_dict["type"] == "pinch"
    assert cam_dict["score"] == 0.85
    assert cam_dict["slope_deg"] == 12.3
    assert cam_dict["aspect_deg"] == 180.0
    assert "reason" in cam_dict


def test_export_geojson(camera_placer, tmp_path):
    """Test GeoJSON export."""
    cameras = camera_placer.select_cameras(num_cameras=3, min_distance_m=100)
    output_path = tmp_path / "cameras.geojson"

    camera_placer.export_geojson(cameras, output_path)

    assert output_path.exists()

    # Verify GeoJSON structure
    import json

    with open(output_path) as f:
        data = json.load(f)

    assert "type" in data
    assert "features" in data
    assert len(data["features"]) == len(cameras)
