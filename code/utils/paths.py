"""
Project-wide canonical path definitions.

This module centralizes all important directory paths
to avoid hard-coded strings and improve portability.

Assumption:
- File is located at: code/utils/paths.py
- Project root is two levels above this file
"""

from pathlib import Path


# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
# code/utils/paths.py → parents[2] → project root
ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# Standard directories
# ------------------------------------------------------------------
MODELS = ROOT / "models"
EXTERNAL = ROOT / "external"
RESULTS = ROOT / "results"


# Optional: ensure paths exist where appropriate
# (do NOT auto-create MODELS / EXTERNAL)
RESULTS.mkdir(exist_ok=True)
