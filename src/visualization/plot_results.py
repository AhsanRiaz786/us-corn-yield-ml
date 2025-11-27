"""
Visualization utilities for model results and data exploration.

Creates publication-quality figures for model comparison, feature importance,
and error analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple


class ResultsVisualizer:
    """Generate visualizations for model results and analysis."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures. Defaults to results/figures/
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / "results" / "figures"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_model_comparison(self, results_file: Optional[str] = None, 
                            save: bool = True) -> None:
        """
        Create bar chart comparing model performance metrics.
        
        Args:
            results_file: Path to model_comparison.csv
            save: Whether to save figure to disk
        """
        if results_file is None:
            results_file = Path(__file__).parent.parent.parent / "results" / "tables" / "model_comparison.csv"
        
        df = pd.read_csv(results_file)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Test_R2', 'Test_MAE', 'Test_RMSE']
        titles = ['R² Score', 'Mean Absolute Error (BU/ACRE)', 'Root Mean Squared Error (BU/ACRE)']
        
        for ax, metric, title in zip(axes, metrics, titles):
            bars = ax.bar(range(len(df)), df[metric], color=sns.color_palette("husl", len(df)))
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['Model'], rotation=45, ha='right')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, df[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}' if metric == 'Test_R2' else f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "model_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {output_path}")
        
        plt.close()
    
    def plot_feature_importance(self, importance_file: Optional[str] = None,
                               top_n: int = 20, save: bool = True) -> None:
        """
        Create horizontal bar chart of feature importance.
        
        Args:
            importance_file: Path to feature_importance.csv
            top_n: Number of top features to display
            save: Whether to save figure to disk
        """
        if importance_file is None:
            importance_file = Path(__file__).parent.parent.parent / "results" / "tables" / "feature_importance.csv"
        
        df = pd.read_csv(importance_file)
        
        df_top = df.head(top_n).copy()
        df_top['Importance_Pct'] = df_top['Importance'] * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(df_top)), df_top['Importance_Pct'], 
                       color=sns.color_palette("viridis", len(df_top)))
        
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['Feature'])
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'Top {top_n} Feature Importance (XGBoost Model)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, df_top['Importance_Pct'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {val:.2f}%',
                   ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "feature_importance.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {output_path}")
        
        plt.close()
    
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                               title: str = "Predicted vs Actual Yield",
                               save_name: Optional[str] = None) -> None:
        """
        Create scatter plot of predicted vs actual values.
        
        Args:
            y_true: Actual yield values
            y_pred: Predicted yield values
            title: Plot title
            save_name: Filename to save (without extension)
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', lw=2, label='Perfect Prediction')
        
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        textstr = f'R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual Yield (BU/ACRE)', fontsize=12)
        ax.set_ylabel('Predicted Yield (BU/ACRE)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            output_path = self.output_dir / f"{save_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {output_path}")
        
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_name: Optional[str] = None) -> None:
        """
        Create residual plot to assess model fit.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_name: Filename to save (without extension)
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Yield (BU/ACRE)', fontsize=11)
        axes[0].set_ylabel('Residuals (BU/ACRE)', fontsize=11)
        axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals (BU/ACRE)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            output_path = self.output_dir / f"{save_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot saved to: {output_path}")
        
        plt.close()
    
    def generate_all_plots(self):
        """Generate all standard visualization plots."""
        print("Generating all visualization plots...")
        
        try:
            self.plot_model_comparison()
            print("  Model comparison plot created")
        except Exception as e:
            print(f"  Error creating model comparison plot: {e}")
        
        try:
            self.plot_feature_importance()
            print("  Feature importance plot created")
        except Exception as e:
            print(f"  Error creating feature importance plot: {e}")
        
        print("Visualization generation complete!")


def main():
    """Generate all standard plots."""
    visualizer = ResultsVisualizer()
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()

