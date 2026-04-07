import matplotlib.pyplot as plt
import seaborn as sns

def apply_scientific_style():
    """
    Applies a clean, scientific styling to matplotlib and seaborn plots.
    This ensures consistency across all visualizations.
    """
    sns.set_theme(style="whitegrid", context="paper")
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.figsize": (8, 6),
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "axes.formatter.use_mathtext": True
    })
