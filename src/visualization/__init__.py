"""
Visualization utilities for model results and data exploration.

Provides plotting functions for model comparison, feature importance,
error analysis, and spatial/temporal patterns.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality plotting defaults
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

FIGURES_DIR = Path(__file__).parent.parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "FIGURES_DIR",
]

