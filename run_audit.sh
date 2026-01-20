#!/bin/bash
cd /workspaces/OLS-v1
echo "Starting audit..."
python -m land_audit.cli --lat 42.3167 --lon -76.4917 --radius-m 3000 --cameras-n 6 --output-dir danby_improved --dem-resolution-m 10
echo "Audit complete!"
echo "Open http://localhost:8000/ to view results"
cd danby_improved && python -m http.server 8001 &
echo "Server started on port 8001"
