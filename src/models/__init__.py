"""
Machine learning model training and evaluation module.

Implements baseline, classical ML, and ensemble models for corn yield prediction.
Includes hyperparameter tuning, evaluation metrics, and error analysis.
"""

from pathlib import Path

# Define model output directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "MODELS_DIR",
    "RESULTS_DIR",
]

