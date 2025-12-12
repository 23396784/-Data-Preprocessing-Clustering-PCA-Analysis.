"""
Visualization Module

Comprehensive plotting functions for preprocessing, clustering, and PCA analysis:
- Distribution plots
- Silhouette analysis
- PCA variance explained
- 3D cluster visualization
- Comparison charts

Author: Victor Prefa
Course: SIG720 Machine Learning - Deakin University
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import silhouette_samples, confusion_matrix
from typing import Tuple, List, Optional, Dict


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot missing values summary.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    set_style()
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if len(missing) == 0:
        print("No missing values found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(missing)))
    bars = ax.barh(missing.index, missing.values, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, missing.values):
        ax.text(bar.get_width() + max(missing) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
    
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Missing Values by Feature', fontweight='bold')
    ax.set_xlim(0, max(missing) * 1.2)
    
    plt.tight_layout()
    return fig


def plot_distribution_comparison(before: pd.DataFrame, after: pd.DataFrame,
                                features: List[str], figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot feature distributions before and after transformation.
    
    Parameters
    ----------
    before : pd.DataFrame
        Data before transformation
    after : pd.DataFrame
        Data after transformation
    features : list
        Features to plot
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    set_style()
    
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols * 2 > 1 else [axes]
    
    for i, feat in enumerate(features):
        # Before
        ax_before = axes[i * 2]
        ax_before.hist(before[feat].dropna(), bins=30, color='steelblue', 
                      edgecolor='black', alpha=0.7)
        ax_before.set_title(f'{feat}\n(Before)', fontsize=10)
        ax_before.set_ylabel('Frequency')
        
        # After
        ax_after = axes[i * 2 + 1]
        ax_after.hist(after[feat].dropna(), bins=30, color='coral', 
                     edgecolor='black', alpha=0.7)
        ax_after.set_title(f'{feat}\n(After)', fontsize=10)
    
    # Hide unused subplots
    for j in range(n_features * 2, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Feature Distributions: Before vs After Scaling', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_silhouette_curve(scores: Dict[int, float], optimal_k: int,
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot Silhouette score curve for k selection.
    
    Parameters
    ----------
    scores : dict
        Dictionary of {k: silhouette_score}
    optimal_k : int
        Optimal number of clusters
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    k_values = list(scores.keys())
    score_values = list(scores.values())
    
    ax.plot(k_values, score_values, 'b-o', linewidth=2, markersize=10, label='Silhouette Score')
    ax.fill_between(k_values, score_values, alpha=0.2)
    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
               label=f'Optimal k = {optimal_k}')
    
    # Highlight optimal point
    ax.scatter([optimal_k], [scores[optimal_k]], s=200, c='red', zorder=5)
    ax.annotate(f'{scores[optimal_k]:.4f}', 
               (optimal_k, scores[optimal_k]),
               xytext=(10, 10), textcoords='offset points',
               fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis for Optimal k', fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_silhouette_diagram(X: np.ndarray, labels: np.ndarray, n_clusters: int,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot silhouette diagram showing sample-level scores.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    n_clusters : int
        Number of clusters
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    from sklearn.metrics import silhouette_score
    
    set_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_values = sample_silhouette_values[labels == i]
        cluster_values.sort()
        
        size = cluster_values.shape[0]
        y_upper = y_lower + size
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i), fontweight='bold')
        
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
               label=f'Average: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.set_title(f'Silhouette Plot (k={n_clusters})', fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_pca_variance(explained_variance_ratio: np.ndarray, 
                     figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot PCA variance explained.
    
    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Variance ratio per component
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    individual = explained_variance_ratio * 100
    cumulative = np.cumsum(individual)
    n_components = len(individual)
    
    # Individual variance
    ax1 = axes[0]
    bars = ax1.bar(range(1, n_components + 1), individual, 
                   color='steelblue', edgecolor='black', alpha=0.8)
    for bar, var in zip(bars, individual):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{var:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Individual Variance', fontweight='bold')
    ax1.set_xticks(range(1, n_components + 1))
    ax1.set_xticklabels([f'PC{i}' for i in range(1, n_components + 1)])
    
    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(range(1, n_components + 1), cumulative, 'b-o', linewidth=2, markersize=12)
    ax2.fill_between(range(1, n_components + 1), cumulative, alpha=0.3)
    for i, var in enumerate(cumulative):
        ax2.text(i + 1, var + 2, f'{var:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax2.axhline(y=cumulative[-1], color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance (%)')
    ax2.set_title('Cumulative Variance', fontweight='bold')
    ax2.set_xticks(range(1, n_components + 1))
    ax2.set_ylim(0, max(100, cumulative[-1] + 10))
    
    plt.tight_layout()
    return fig


def plot_3d_pca_clusters(X_pca: np.ndarray, labels: np.ndarray,
                        title: str = '3D PCA Clusters',
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot 3D PCA cluster visualization.
    
    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed data (n_samples, 3)
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=labels, cmap='Set1', s=60, alpha=0.7, edgecolors='w')
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Cluster', shrink=0.6)
    plt.tight_layout()
    return fig


def plot_clustering_comparison(results_df: pd.DataFrame,
                              figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot clustering comparison: Original vs PCA.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Comparison results with columns: n_clusters, silhouette_original, silhouette_pca
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    set_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    k_values = results_df['n_clusters'].values
    
    # Silhouette comparison
    ax1 = axes[0]
    ax1.plot(k_values, results_df['silhouette_original'], 'b-o', 
            linewidth=2, markersize=8, label='Original Features')
    ax1.plot(k_values, results_df['silhouette_pca'], 'r-s', 
            linewidth=2, markersize=8, label='PCA Features')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Clustering Quality Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agreement metrics
    ax2 = axes[1]
    ax2.plot(k_values, results_df['ari_score'], 'g-^', 
            linewidth=2, markersize=8, label='ARI')
    ax2.plot(k_values, results_df['nmi_score'], 'm-d', 
            linewidth=2, markersize=8, label='NMI')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Agreement Score')
    ax2.set_title('Clustering Agreement', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Improvement percentage
    ax3 = axes[2]
    bars = ax3.bar(k_values, results_df['improvement_pct'], 
                   color='teal', edgecolor='black', alpha=0.8)
    for bar, imp in zip(bars, results_df['improvement_pct']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{imp:+.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('PCA Improvement over Original', fontweight='bold')
    ax3.set_xticks(k_values)
    
    plt.tight_layout()
    return fig


def plot_cluster_sizes(sizes_original: np.ndarray, sizes_pca: np.ndarray,
                       figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
    """
    Plot cluster size comparison.
    
    Parameters
    ----------
    sizes_original : np.ndarray
        Cluster sizes for original features
    sizes_pca : np.ndarray
        Cluster sizes for PCA features
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original
    ax1 = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes_original)))
    bars1 = ax1.bar(range(len(sizes_original)), sizes_original, color=colors, edgecolor='black')
    for bar, size in zip(bars1, sizes_original):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(size), ha='center', fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Original Features', fontweight='bold')
    ax1.set_xticks(range(len(sizes_original)))
    
    # PCA
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(sizes_pca)), sizes_pca, color=colors[:len(sizes_pca)], edgecolor='black')
    for bar, size in zip(bars2, sizes_pca):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(size), ha='center', fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('PCA Features', fontweight='bold')
    ax2.set_xticks(range(len(sizes_pca)))
    
    plt.suptitle('Cluster Size Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_summary_table(results: Dict) -> str:
    """
    Create formatted summary table as string.
    
    Parameters
    ----------
    results : dict
        Analysis results
        
    Returns
    -------
    str
        Formatted table
    """
    table = """
╔════════════════════════════════════════════════════════════════╗
║                    ANALYSIS SUMMARY                            ║
╠════════════════════════════════════════════════════════════════╣
"""
    for key, value in results.items():
        if isinstance(value, float):
            table += f"║  {key:<30} {value:>25.4f}  ║\n"
        elif isinstance(value, int):
            table += f"║  {key:<30} {value:>25,}  ║\n"
        else:
            table += f"║  {key:<30} {str(value):>25}  ║\n"
    
    table += "╚════════════════════════════════════════════════════════════════╝"
    return table


if __name__ == "__main__":
    # Demo
    print("Visualization Module Demo")
    print("-" * 40)
    
    # Generate sample data
    np.random.seed(42)
    
    # Silhouette scores
    scores = {k: 0.6 - 0.05 * k + np.random.randn() * 0.02 for k in range(2, 8)}
    optimal_k = max(scores, key=scores.get)
    
    fig = plot_silhouette_curve(scores, optimal_k)
    plt.close()
    
    print("Demo plots generated successfully!")
