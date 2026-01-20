# Orome Land Audit Engine

[![Lint and Test](https://github.com/MizuguchiJAkira/OLS-v1/actions/workflows/test.yml/badge.svg)](https://github.com/MizuguchiJAkira/OLS-v1/actions/workflows/test.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A terrain-based wildlife habitat analysis tool that identifies optimal camera trap locations for wildlife management. The Land Audit Engine analyzes elevation data to identify travel corridors (pinch points) and bedding zones, then recommends strategic camera placement.

## Features

- **Terrain Analysis**: Computes slope, aspect, hillshade, TPI, and ruggedness from DEM data
- **Intelligent Scoring**: Heuristic-based identification of:
  - Travel corridors and pinch points (movement funnels)
  - Bedding zones (resting areas with optimal characteristics)
- **Camera Optimization**: Smart placement algorithm with minimum spacing constraints
- **Professional Reporting**: PDF reports with maps, heatmaps, and detailed recommendations
- **GeoJSON Export**: Portable spatial data for use in GIS applications

## Quick Start

### Installation

Requires Python 3.11+

```bash
# Clone the repository
git clone https://github.com/MizuguchiJAkira/OLS-v1.git
cd OLS-v1

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"
```

### Basic Usage

#### Point Mode (circular AOI)

```bash
land-audit --lat 42.45 --lon -76.48 --radius-m 2000 --cameras-n 6 --output-dir out/
```

#### Polygon Mode (custom boundary)

```bash
land-audit --geojson property.geojson --cameras-n 8 --output-dir out/
```

#### Options

```
--lat FLOAT              Latitude for point mode
--lon FLOAT              Longitude for point mode
--radius-m FLOAT         Radius in meters for point mode (default: 2000)
--geojson PATH           Path to GeoJSON file for polygon mode
--cameras-n INTEGER      Number of cameras to place (default: 6)
--output-dir PATH        Output directory (default: out/)
--dem-resolution-m INT   DEM resolution: 10 or 30 meters (default: 10)
--cache-dir PATH         Cache directory for DEM files
--verbose                Enable verbose logging
```

### Example GeoJSON Input

Create a `property.geojson` file:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "My Property"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-76.50, 42.40],
          [-76.45, 42.40],
          [-76.45, 42.45],
          [-76.50, 42.45],
          [-76.50, 42.40]
        ]]
      }
    }
  ]
}
```

## Output Structure

After running an audit, you'll find:

```
out/
├── report.pdf              # Professional PDF report
├── audit.geojson          # All layers and camera points
└── debug/
    ├── clipped_dem.tif    # Input elevation data
    ├── slope.tif          # Slope in degrees
    ├── aspect.tif         # Aspect (0-360°)
    ├── hillshade.tif      # Visualization layer
    ├── pinch_score.tif    # Travel corridor scores
    └── bedding_score.tif  # Bedding zone scores
```

## How It Works

### 1. Data Acquisition
- Fetches USGS 3DEP elevation data via [py3dep](https://github.com/hyriver/py3dep)
- Caches downloads to avoid redundant API calls
- Works anywhere in the United States

### 2. Terrain Analysis
- **Slope**: Gradient calculation using Sobel filters
- **Aspect**: Slope direction (0° = North, clockwise)
- **TPI**: Topographic Position Index (ridges vs valleys)
- **Ruggedness**: Local elevation variability

### 3. Heuristic Scoring

**Bedding Zones** (preferred characteristics):
- Gentle slopes (<15°) for comfort
- South/SE facing (warmth)
- Moderate ruggedness (edge habitat)
- Slightly elevated (drainage, visibility)

**Pinch Points** (movement corridors):
- Cost-distance analysis with slope-based friction
- Identifies natural terrain funnels
- High-density travel corridors

### 4. Camera Placement
- Allocates cameras: ~60% pinch points, ~40% bedding zones
- Enforces minimum spacing (default: 150m)
- Generates rationales for each location

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=land_audit --cov-report=html

# Run only unit tests (fast)
pytest tests/ -m "not integration"

# Run integration tests
pytest tests/ -m integration
```

### Code Quality

```bash
# Lint
ruff check src/ tests/

# Format
black src/ tests/

# Type hints (future)
# mypy src/
```

### Project Structure

```
OLS-v1/
├── src/land_audit/
│   ├── __init__.py
│   ├── audit.py         # Main orchestration
│   ├── cli.py           # Command-line interface
│   ├── dem.py           # DEM acquisition & caching
│   ├── terrain.py       # Terrain derivatives
│   ├── scoring.py       # Heuristic scoring
│   ├── cameras.py       # Camera placement
│   └── report.py        # PDF generation
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_terrain.py
│   ├── test_scoring.py
│   ├── test_cameras.py
│   └── test_integration.py
├── pyproject.toml       # Project configuration
├── .github/workflows/   # CI/CD
└── README.md
```

## Real-World Example

```bash
# Analyze a property in the Finger Lakes region of NY
land-audit \
  --lat 42.4473 \
  --lon -76.4840 \
  --radius-m 3000 \
  --cameras-n 10 \
  --output-dir finger_lakes_audit \
  --dem-resolution-m 10 \
  --verbose
```

Expected runtime: 30-90 seconds (depending on DEM download)

## Limitations & Known Issues

- **Geographic Coverage**: U.S. only (due to 3DEP data source)
- **Resolution**: 10m DEM is best; 30m available but less detailed
- **Simplified Heuristics**: MVP uses terrain-only analysis
  - No landcover classification (forest, fields, water)
  - No hydrography (streams, ponds)
  - No roads/infrastructure data

## Next Steps (Future Enhancements)

### Phase 2: Landcover Integration
- Add NLCD (National Land Cover Database) data
- Incorporate forest/field edges into bedding scores
- Improve cover proxy for bedding zones
- Priority: High | Effort: Medium

### Phase 3: Hydrography
- Integrate NHD (National Hydrography Dataset)
- Model water source proximity
- Identify stream crossings (high-value pinch points)
- Priority: Medium | Effort: Medium

### Phase 4: Advanced Features
- Multi-season analysis (different DEM dates)
- User-defined scoring weights
- Interactive web interface
- Export to KML for Google Earth
- Trail camera network optimization
- Priority: Low | Effort: High

### Phase 5: Machine Learning
- Train models on confirmed wildlife sightings
- Refine heuristics with ground truth data
- Species-specific scoring (deer vs bear vs turkey)
- Priority: Low | Effort: Very High

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure `ruff` and `black` pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in research, please cite:

```bibtex
@software{orome_land_audit_2026,
  author = {Orome},
  title = {Land Audit Engine: Terrain-based Wildlife Camera Placement},
  year = {2026},
  url = {https://github.com/MizuguchiJAkira/OLS-v1}
}
```

## Support

- **Issues**: https://github.com/MizuguchiJAkira/OLS-v1/issues
- **Discussions**: https://github.com/MizuguchiJAkira/OLS-v1/discussions

## Acknowledgments

- USGS 3DEP for elevation data
- [py3dep](https://github.com/hyriver/py3dep) for seamless DEM access
- [rasterio](https://github.com/rasterio/rasterio) for raster processing

---

**Built with ❤️ for wildlife managers and conservation professionals**
