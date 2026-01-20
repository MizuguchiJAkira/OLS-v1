"""Tests for scoring module."""

import numpy as np
import pytest

from land_audit.scoring import HeuristicScorer
from land_audit.terrain import TerrainAnalyzer
from tests.conftest import create_mock_dem


@pytest.fixture
def scorer(tmp_path):
    """Create scorer with mock terrain data."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    slope = analyzer.compute_slope()
    aspect = analyzer.compute_aspect()
    tpi = analyzer.compute_tpi()
    ruggedness = analyzer.compute_ruggedness()

    return HeuristicScorer(
        slope=slope,
        aspect=aspect,
        tpi=tpi,
        ruggedness=ruggedness,
        transform=analyzer.transform,
        crs=analyzer.crs,
    )


def test_bedding_score_computation(scorer):
    """Test bedding score computation."""
    bedding_score = scorer.compute_bedding_score()

    assert bedding_score.shape == scorer.slope.shape
    assert bedding_score.min() >= 0
    assert bedding_score.max() <= 1
    assert not np.isnan(bedding_score).any()


def test_pinch_score_computation(scorer, tmp_path):
    """Test pinch score computation."""
    dem_path = create_mock_dem(tmp_path / "test_dem.tif")
    analyzer = TerrainAnalyzer(dem_path)

    pinch_score = scorer.compute_pinch_score(
        elevation=analyzer.elevation,
        resolution=analyzer.resolution,
    )

    assert pinch_score.shape == scorer.slope.shape
    assert pinch_score.min() >= 0
    assert pinch_score.max() <= 1


def test_normalize(scorer):
    """Test normalization function."""
    arr = np.array([0, 5, 10, 15, 20])
    normalized = scorer._normalize(arr)

    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalized.shape == arr.shape


def test_bedding_score_prefers_gentle_slopes(scorer):
    """Test that bedding score prefers gentle slopes."""
    # Manually create scenario with gentle vs steep slopes
    gentle_mask = scorer.slope < 10
    steep_mask = scorer.slope > 30

    if gentle_mask.any() and steep_mask.any():
        bedding_score = scorer.compute_bedding_score()
        gentle_mean = bedding_score[gentle_mask].mean()
        steep_mean = bedding_score[steep_mask].mean()

        # Gentle slopes should generally score higher
        assert gentle_mean > steep_mean


def test_score_output_range(scorer, tmp_path):
    """Test that scores are properly bounded."""
    bedding_score = scorer.compute_bedding_score()

    # All scores should be in valid range
    assert np.all(bedding_score >= 0)
    assert np.all(bedding_score <= 1)

    # Should have some variation
    assert bedding_score.std() > 0
