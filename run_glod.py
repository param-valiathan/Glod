"""Top-level launcher — run with: python run_glod.py"""
import sys
from pathlib import Path

# Ensure the package directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from thermal_analysis.main import main

if __name__ == "__main__":
    main()
