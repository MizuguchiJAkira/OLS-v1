"""Integration smoke test."""

import json

import pytest

from land_audit.audit import LandAudit
from tests.conftest import create_mock_dem


@pytest.mark.integration
def test_point_audit_smoke(tmp_path, monkeypatch):
    """Smoke test for point-based audit with mocked DEM."""

    # Create a mock DEM and patch the DEM fetcher to use it
    mock_dem_path = create_mock_dem(tmp_path / "cache" / "mock_dem.tif")

    def mock_fetch_dem(self, bounds, resolution=10, crs="EPSG:4326"):
        """Return pre-created mock DEM."""
        return mock_dem_path

    # Monkey-patch the fetch_dem method
    from land_audit import dem

    monkeypatch.setattr(dem.DEMManager, "fetch_dem", mock_fetch_dem)

    # Run audit
    output_dir = tmp_path / "output"
    audit = LandAudit(output_dir=output_dir, cache_dir=tmp_path / "cache")

    report_path = audit.run_point_audit(
        lat=42.45,
        lon=-76.45,
        radius_m=1000,
        cameras_n=4,
        dem_resolution_m=10,
    )

    # Verify outputs exist
    assert report_path.exists()
    assert (output_dir / "audit.geojson").exists()
    assert (output_dir / "debug" / "slope.tif").exists()
    assert (output_dir / "debug" / "aspect.tif").exists()
    assert (output_dir / "debug" / "hillshade.tif").exists()

    # Verify GeoJSON structure
    with open(output_dir / "audit.geojson") as f:
        geojson = json.load(f)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0

    # Check for camera features
    camera_features = [
        f for f in geojson["features"] if f["properties"]["layer"] == "recommended_cameras"
    ]
    assert len(camera_features) <= 4


@pytest.mark.integration
def test_polygon_audit_smoke(tmp_path, monkeypatch):
    """Smoke test for polygon-based audit with mocked DEM."""

    # Create mock DEM
    mock_dem_path = create_mock_dem(tmp_path / "cache" / "mock_dem.tif")

    def mock_fetch_dem(self, bounds, resolution=10, crs="EPSG:4326"):
        return mock_dem_path

    from land_audit import dem

    monkeypatch.setattr(dem.DEMManager, "fetch_dem", mock_fetch_dem)

    # Create test GeoJSON
    test_geojson = tmp_path / "test_property.geojson"
    geojson_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Test Property"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-76.48, 42.43],
                            [-76.42, 42.43],
                            [-76.42, 42.47],
                            [-76.48, 42.47],
                            [-76.48, 42.43],
                        ]
                    ],
                },
            }
        ],
    }

    with open(test_geojson, "w") as f:
        json.dump(geojson_content, f)

    # Run audit
    output_dir = tmp_path / "output"
    audit = LandAudit(output_dir=output_dir, cache_dir=tmp_path / "cache")

    report_path = audit.run_polygon_audit(
        geojson_path=test_geojson,
        cameras_n=3,
        dem_resolution_m=10,
    )

    # Verify outputs
    assert report_path.exists()
    assert (output_dir / "audit.geojson").exists()


def test_audit_error_handling(tmp_path):
    """Test error handling for invalid inputs."""
    audit = LandAudit(output_dir=tmp_path / "output")

    # Test invalid GeoJSON path (should raise error from geopandas/pyogrio)
    error_raised = False
    try:
        audit.run_polygon_audit(
            geojson_path=tmp_path / "nonexistent.geojson",
            cameras_n=3,
        )
    except Exception as e:
        # Expect some kind of file/data error
        error_raised = any(
            x in str(type(e).__name__).lower() for x in ["file", "data", "source", "io", "value"]
        )

    assert error_raised, "Expected error for nonexistent file"
