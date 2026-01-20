# Modal Cloud Compute Integration

This system uses [Modal](https://modal.com) to offload CPU-intensive computations to the cloud, preventing your Codespace from melting during heavy viewshed calculations.

## Why Modal?

- **No Codespace meltdown**: Heavy computations run on Modal's cloud (8 cores, 32GB RAM)
- **Pay-per-second**: Only pay when computing (not 24/7 like a dedicated server)
- **Automatic scaling**: Modal handles all infrastructure
- **Fast**: Parallel processing across multiple cores

## Setup (One-time)

### 1. Install Modal Client
Already done! Modal is installed as part of the project dependencies.

### 2. Create Modal Account
Visit [modal.com/signup](https://modal.com/signup) and create a free account.

### 3. Authenticate
Run:
```bash
python modal_setup.py auth
```

This will open your browser for authentication. Follow the prompts.

### 4. Test Connection (Optional)
```bash
python modal_setup.py test
```

## Usage

### Option 1: Automatic (Recommended)
The system automatically uses Modal cloud compute when available. Just run your normal commands:

```bash
python -m land_audit.cli --lat 42.5 --lon -76.8 --radius-m 2000 --cameras-n 6 --output-dir test_zones
```

The optimization will run on Modal's cloud (8 cores, 32GB RAM) instead of your Codespace.

### Option 2: Force Local Compute
Edit `src/land_audit/report.py` line 541:

```python
use_modal = False  # Set to False to force local compute
```

## Performance Comparison

| Method | CPU Usage | RAM Usage | Time (6 cameras) |
|--------|-----------|-----------|------------------|
| **Local (Codespace)** | 100% (melts) | 8-16GB | ~48 minutes |
| **Modal Cloud** | 0% (local) | Minimal | ~8-12 minutes |

## How It Works

1. **Local orchestration**: Your terminal just coordinates the work
2. **Cloud execution**: Heavy viewshed calculations run on Modal (8 cores, 32GB RAM)
3. **Results return**: Optimized camera positions come back to your machine
4. **Visualization**: 3D terrain visualization generated locally

## Architecture

```
┌─────────────────┐
│  Your Codespace │  ← Just orchestrating
└────────┬────────┘
         │ Sends: DEM data, candidates, parameters
         ▼
┌─────────────────┐
│  Modal Cloud    │  ← 8 cores, 32GB RAM doing heavy lifting
│  (8 × 4GB vCPU) │  ← Viewshed calculations, optimization
└────────┬────────┘
         │ Returns: Optimized camera positions
         ▼
┌─────────────────┐
│  Your Codespace │  ← Generates visualization
└─────────────────┘
```

## Cost

Modal has a generous free tier:
- **Free**: $30/month compute credits
- **Typical cost**: ~$0.50-1.00 per full audit (6 cameras, 2km radius)
- **Viewshed calculation**: ~$0.10 per camera iteration

For reference:
- Our test run (6 cameras, 292 candidates × 8 azimuths) = ~$0.60
- Previous run (6 cameras, 607 candidates × 8 azimuths) = ~$1.20

## Troubleshooting

### "modal: command not found"
Run: `pip install modal`

### "Not authenticated"
Run: `python modal_setup.py auth`

### "Modal app not found"
Deploy the app: `python modal_setup.py deploy`

### Still using local compute
Check `src/land_audit/report.py` line 541:
```python
use_modal = True  # Must be True
```

## Toggle Between Local and Cloud

**For cloud compute (recommended):**
```python
# src/land_audit/report.py, line 541
use_modal = True
```

**For local compute (testing only):**
```python
# src/land_audit/report.py, line 541
use_modal = False
```

## Advanced: Direct Modal Usage

You can also call Modal functions directly:

```python
from src.land_audit.modal_compute import run_optimization_on_modal

results = run_optimization_on_modal(
    candidates=candidate_array,
    target_points=target_array,
    elevation=dem_array,
    transform=affine_transform,
    bounds=(minx, miny, maxx, maxy),
    num_cameras=6,
    terrain_params={
        'fov': 45.0,
        'max_range': 20.0,
        'camera_height': 1.5,
    }
)
```

## What Gets Offloaded?

✅ **Offloaded to Modal (CPU-intensive):**
- Viewshed calculations (line-of-sight ray tracing)
- Camera optimization (greedy algorithm iterations)
- Coverage computations

❌ **Stays Local (lightweight):**
- DEM downloads
- DBSCAN clustering
- Visualization generation
- PDF report creation

## Files

- `src/land_audit/modal_compute.py` - Modal cloud functions
- `src/land_audit/report.py` - Integration point (line 541: `use_modal = True`)
- `modal_setup.py` - Setup and testing utilities
- `pyproject.toml` - Modal dependency added
