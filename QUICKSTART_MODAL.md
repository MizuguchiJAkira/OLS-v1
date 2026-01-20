# 🚀 Quick Start: Modal Cloud Compute

Get supercomputer performance without melting your Codespace!

## 30-Second Setup

1. **Authenticate with Modal:**
   ```bash
   python modal_setup.py auth
   ```
   - Opens browser → Sign up at modal.com → Free account
   - Copy token back to terminal

2. **Run your audit:**
   ```bash
   python -m land_audit.cli --lat 42.5 --lon -76.8 --radius-m 2000 --cameras-n 6 --output-dir test_zones
   ```

**That's it!** The heavy computation now runs on Modal's cloud (8 cores, 32GB RAM).

## What Changed?

**Before (Local):**
- ❌ 100% CPU usage (Codespace melts)
- ❌ 48 minutes for 6 cameras
- ❌ Risk of timeout/crash

**After (Modal Cloud):**
- ✅ 0% local CPU (just orchestrating)
- ✅ 8-12 minutes for 6 cameras (8 cores parallel)
- ✅ No crashes, no timeouts

## Cost

- **Free tier**: $30/month credits
- **Typical audit**: $0.50-1.00 per run
- **Our test**: $0.60 (6 cameras, 2km radius)

## Verify It's Working

Look for this log message:
```
Running camera placement optimization (using Modal cloud)...
```

If you see `(using local compute)` instead, run authentication again.

## Need Help?

See [MODAL_README.md](MODAL_README.md) for detailed documentation.
