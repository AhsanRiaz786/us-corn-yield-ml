"""
Evaluate trained models with comprehensive error analysis.

Generates:
- Error distribution plots
- Temporal pattern analysis
- Spatial pattern analysis
- Yield level analysis
- Worst predictions analysis

Requires: Trained models from 03_train_models.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import evaluate
from src.visualization import plot_results


def main():
    """Execute model evaluation pipeline."""
    print("=" * 80)
    print("US CORN YIELD PREDICTION - MODEL EVALUATION PIPELINE")
    print("=" * 80)
    print()
    
    print("[1/2] Running comprehensive error analysis...")
    print("-" * 80)
    try:
        evaluate.main()
        print("Error analysis complete.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    print("\n[2/2] Generating result visualizations...")
    print("-" * 80)
    try:
        visualizer = plot_results.ResultsVisualizer()
        visualizer.generate_all_plots()
        print("Visualizations created.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        return
    
    print("\n" + "=" * 80)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - results/figures/")
    print("  - results/tables/")
    print("  - results/error_analysis_summary.txt")
    print("\nFinal step: Launch dashboard with 'streamlit run app/streamlit_app.py'")


if __name__ == "__main__":
    main()

