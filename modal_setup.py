#!/usr/bin/env python3
"""
Modal setup and testing script.

Run this to:
1. Authenticate with Modal (if not already authenticated)
2. Test the Modal compute functions
3. Deploy the Modal app

Usage:
    python modal_setup.py auth      # Authenticate with Modal
    python modal_setup.py test      # Test Modal functions
    python modal_setup.py deploy    # Deploy Modal app
"""

import sys


def authenticate():
    """Authenticate with Modal."""
    print("🔐 Setting up Modal authentication...")
    print("Visit: https://modal.com/signup to create a free account")
    print()
    import subprocess
    subprocess.run(["modal", "token", "new"])


def test_modal():
    """Test Modal functions."""
    print("🧪 Testing Modal compute functions...")
    
    from src.land_audit.modal_compute import app
    
    # Simple test
    @app.function()
    def hello():
        return "Hello from Modal cloud! 🚀"
    
    with app.run():
        result = hello.remote()
        print(f"✅ Modal test successful: {result}")


def deploy():
    """Deploy Modal app."""
    print("🚀 Deploying Modal app...")
    import subprocess
    subprocess.run(["modal", "deploy", "src/land_audit/modal_compute.py"])


def show_usage():
    """Show usage instructions."""
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "auth":
        authenticate()
    elif command == "test":
        test_modal()
    elif command == "deploy":
        deploy()
    else:
        print(f"Unknown command: {command}")
        show_usage()
        sys.exit(1)
