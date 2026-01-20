#!/usr/bin/env python
"""Quick test of Finger Lakes region"""
import sys
sys.path.insert(0, '/workspaces/OLS-v1/src')

from land_audit.audit import LandAudit

print("🌲 Testing Finger Lakes region (hills + flat + water)...")
print("   Location: 42.5°N, -76.8°W")
print("   Radius: 3km, Cameras: 8")

audit = LandAudit(debug=True)
try:
    report = audit.run_point_audit(
        lat=42.5,
        lon=-76.8,
        radius_m=3000.0,
        cameras_n=8,
        dem_resolution_m=10,
        output_dir="finger_lakes"
    )
    print(f"✅ Success! Report: {report}")
    print(f"🗺️  3D Map: finger_lakes/terrain_3d.html")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
