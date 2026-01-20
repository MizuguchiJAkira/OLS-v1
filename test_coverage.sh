#!/bin/bash
cd /workspaces/OLS-v1
python -m land_audit.cli --lat 42.3167 --lon -76.4917 --radius-m 3000 --cameras-n 6 --output-dir danby_test_cov --dem-resolution-m 10
