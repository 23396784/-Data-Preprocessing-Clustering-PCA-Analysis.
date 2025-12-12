"""
PCA Analysis Module

Principal Component Analysis for high-dimensional gene expression data:
- StandardScaler preprocessing
- PCA dimensionality reduction
- Variance explained analysis
- KMeans comparison: Original vs PCA features

Author: Victor Prefa
Course: SIG720 Machine Learning - Deakin University
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, Dict, List, Optional


class GeneExpressionPCA:
    """
    PCA analysis for high-dimensional gene expression data.
    
    Demonstrates:
    - Dimensionality reduction (20,531 → 3 components)
    - Variance explained analysis
    - Clustering performance comparison
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the PCA analyzer.
        
        Parameters
        ----------
        filepath : str, optional
            Path to gene expression CSV file
        """
        self.filepath = filepath
        self.df = None
        self.X_original = None
        self.X_scaled = None
        self.X_pca = None
        self.pca = None
        self.scaler = StandardScaler()
        self.n_components = 3
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the gene expression dataset.
        
        Parameters
        ----------
        filepath : str, optional
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded dataset
        """
        path = filepath or self.filepath
        if path is None:
            raise ValueError("No filepath provided")
            
        self.df = pd.read_csv(path)
        
        print(f"Dataset loaded: {self.df.shape[0]} samples × {self.df.shape[1]} features")
        
        return self.df
    
    def prepare_data(self) -> np.ndarray:
        """
        Prepare data for PCA: handle missing values and standardize.
        
        Returns
        -------
        np.ndarray
            Prepared feature matrix
        """
        print("\n=== Preparing Data for PCA ===")
        
        # Select only numerical columns (genes)
        gene_cols = [col for col in self.df.columns if col.startswith('gene_') or col.isdigit()]
        if not gene_cols:
            gene_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove index column if present
            gene_cols = [c for c in gene_cols if 'unnamed' not in c.lower() and c != 'Unnamed: 0']
        
        print(f"  Gene features: {len(gene_cols)}")
        
        # Handle missing values
        self.X_original = self.df[gene_cols].values
        missing_count = np.isnan(self.X_original).sum()
        
        if missing_count > 0:
            print(f"  Missing values: {missing_count}")
            
            # Check for problematic samples
            missing_per_row = np.isnan(self.X_original).sum(axis=1)
            bad_samples = np.where(missing_per_row > len(gene_cols) * 0.5)[0]
            
            if len(bad_samples) > 0:
                print(f"  Removing {len(bad_samples)} samples with >50% missing")
                mask = missing_per_row <= len(gene_cols) * 0.5
                self.X_original = self.X_original[mask]
            
            # Fill remaining missing with column mean
            col_means = np.nanmean(self.X_original, axis=0)
            inds = np.where(np.isnan(self.X_original))
            self.X_original[inds] = np.take(col_means, inds[1])
        
        print(f"  Final shape: {self.X_original.shape}")
        
        # Standardize
        self.X_scaled = self.scaler.fit_transform(self.X_original)
        print(f"  Standardized: mean={self.X_scaled.mean():.6f}, std={self.X_scaled.std():.6f}")
        
        return self.X_scaled
    
    def apply_pca(self, n_components: int = 3) -> np.ndarray:
        """
        Apply PCA to reduce dimensionality.
        
        Parameters
        ----------
        n_components : int
            Number of principal components to keep
            
        Returns
        -------
        np.ndarray
            Transformed data with reduced dimensions
        """
        self.n_components = n_components
        print(f"\n=== Applying PCA (n_components={n_components}) ===")
        
        if self.X_scaled is None:
            self.prepare_data()
        
        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"  Original dimensions: {self.X_scaled.shape[1]:,}")
        print(f"  Reduced dimensions: {self.X_pca.shape[1]}")
        print(f"  Reduction: {(1 - n_components/self.X_scaled.shape[1])*100:.3f}%")
        
        return self.X_pca
    
    def get_variance_explained(self) -> Dict:
        """
        Get variance explained by principal components.
        
        Returns
        -------
        dict
            Variance explained analysis
        """
        if self.pca is None:
            raise ValueError("Run apply_pca() first")
            
        print("\n=== Variance Explained Analysis ===")
        
        individual_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(individual_var)
        eigenvalues = self.pca.explained_variance_
        
        print("\nIndividual Component Variance:")
        for i, (var, eigval) in enumerate(zip(individual_var, eigenvalues)):
            print(f"  PC{i+1}: {var*100:.2f}% (eigenvalue: {eigval:.2f})")
        
        print(f"\nCumulative Variance Explained:")
        for i, cum_var in enumerate(cumulative_var):
            print(f"  PC1 to PC{i+1}: {cum_var*100:.2f}%")
        
        print(f"\n>>> Total variance retained: {cumulative_var[-1]*100:.2f}%")
        
        return {
            'individual_variance': individual_var,
            'cumulative_variance': cumulative_var,
            'eigenvalues': eigenvalues,
            'total_variance_retained': cumulative_var[-1]
        }
    
    def compare_clustering(self, n_clusters_range: range = range(2, 6)) -> pd.DataFrame:
        """
        Compare KMeans clustering on original vs PCA features.
        
        Parameters
        ----------
        n_clusters_range : range
            Range of cluster numbers to test
            
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        if self.X_pca is None:
            raise ValueError("Run apply_pca() first")
            
        print("\n=== Comparing Clustering: Original vs PCA Features ===")
        print(f"Original features: {self.X_scaled.shape[1]:,} dimensions")
        print(f"PCA features: {self.X_pca.shape[1]} dimensions")
        
        results = []
        
        for n_clusters in n_clusters_range:
            print(f"\nTesting k={n_clusters}...")
            
            # KMeans on original features
            kmeans_orig = KMeans(n_clusters=n_clusters, init='k-means++', 
                                n_init=10, random_state=42)
            labels_orig = kmeans_orig.fit_predict(self.X_scaled)
            sil_orig = silhouette_score(self.X_scaled, labels_orig)
            
            # KMeans on PCA features
            kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', 
                               n_init=10, random_state=42)
            labels_pca = kmeans_pca.fit_predict(self.X_pca)
            sil_pca = silhouette_score(self.X_pca, labels_pca)
            
            # Agreement metrics
            ari = adjusted_rand_score(labels_orig, labels_pca)
            nmi = normalized_mutual_info_score(labels_orig, labels_pca)
            
            results.append({
                'n_clusters': n_clusters,
                'silhouette_original': sil_orig,
                'silhouette_pca': sil_pca,
                'improvement_pct': ((sil_pca - sil_orig) / sil_orig) * 100,
                'ari_score': ari,
                'nmi_score': nmi,
                'inertia_original': kmeans_orig.inertia_,
                'inertia_pca': kmeans_pca.inertia_
            })
            
            print(f"  Original: Silhouette = {sil_orig:.4f}")
            print(f"  PCA:      Silhouette = {sil_pca:.4f} ({(sil_pca-sil_orig)/sil_orig*100:+.1f}%)")
            print(f"  Agreement: ARI = {ari:.4f}, NMI = {nmi:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Find best configuration
        best_pca = results_df.loc[results_df['silhouette_pca'].idxmax()]
        best_orig = results_df.loc[results_df['silhouette_original'].idxmax()]
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\nBest for Original features: k={int(best_orig['n_clusters'])} "
              f"(Silhouette: {best_orig['silhouette_original']:.4f})")
        print(f"Best for PCA features: k={int(best_pca['n_clusters'])} "
              f"(Silhouette: {best_pca['silhouette_pca']:.4f})")
        print(f"\nPCA Improvement: {best_pca['improvement_pct']:+.1f}%")
        
        return results_df
    
    def detailed_comparison(self, k: int = 4) -> Dict:
        """
        Detailed clustering comparison for specific k.
        
        Parameters
        ----------
        k : int
            Number of clusters
            
        Returns
        -------
        dict
            Detailed comparison results
        """
        print(f"\n=== Detailed Comparison (k={k}) ===")
        
        # Clustering on original
        kmeans_orig = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels_orig = kmeans_orig.fit_predict(self.X_scaled)
        
        # Clustering on PCA
        kmeans_pca = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels_pca = kmeans_pca.fit_predict(self.X_pca)
        
        # Metrics
        sil_orig = silhouette_score(self.X_scaled, labels_orig)
        sil_pca = silhouette_score(self.X_pca, labels_pca)
        ari = adjusted_rand_score(labels_orig, labels_pca)
        nmi = normalized_mutual_info_score(labels_orig, labels_pca)
        
        results = {
            'original': {
                'silhouette': sil_orig,
                'inertia': kmeans_orig.inertia_,
                'cluster_sizes': np.bincount(labels_orig),
                'labels': labels_orig
            },
            'pca': {
                'silhouette': sil_pca,
                'inertia': kmeans_pca.inertia_,
                'cluster_sizes': np.bincount(labels_pca),
                'labels': labels_pca
            },
            'agreement': {
                'ari': ari,
                'nmi': nmi
            }
        }
        
        # Print formatted table
        print("\n┌─────────────────────────┬─────────────────┬─────────────────┐")
        print("│ Metric                  │ Original        │ PCA             │")
        print("├─────────────────────────┼─────────────────┼─────────────────┤")
        print(f"│ Dimensions              │ {self.X_scaled.shape[1]:>15,} │ {self.X_pca.shape[1]:>15} │")
        print(f"│ Silhouette Score        │ {sil_orig:>15.4f} │ {sil_pca:>15.4f} │")
        print(f"│ Inertia                 │ {kmeans_orig.inertia_:>15,.0f} │ {kmeans_pca.inertia_:>15,.0f} │")
        print("└─────────────────────────┴─────────────────┴─────────────────┘")
        
        print(f"\nClustering Agreement:")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Info: {nmi:.4f}")
        
        print(f"\nCluster Sizes:")
        print(f"  Original: {list(np.bincount(labels_orig))}")
        print(f"  PCA:      {list(np.bincount(labels_pca))}")
        
        return results
    
    def plot_variance_explained(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot variance explained by principal components.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        """
        if self.pca is None:
            print("Run apply_pca() first")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        individual_var = self.pca.explained_variance_ratio_ * 100
        cumulative_var = np.cumsum(individual_var)
        
        # Individual variance
        ax1 = axes[0]
        bars = ax1.bar(range(1, len(individual_var) + 1), individual_var, 
                       color='steelblue', edgecolor='black', alpha=0.8)
        for bar, var in zip(bars, individual_var):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{var:.1f}%', ha='center', va='bottom', fontsize=11)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained (%)', fontsize=12)
        ax1.set_title('Individual Variance Explained', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, len(individual_var) + 1))
        ax1.set_xticklabels([f'PC{i}' for i in range(1, len(individual_var) + 1)])
        
        # Cumulative variance
        ax2 = axes[1]
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                'b-o', linewidth=2, markersize=10)
        ax2.fill_between(range(1, len(cumulative_var) + 1), cumulative_var, alpha=0.3)
        for i, var in enumerate(cumulative_var):
            ax2.text(i + 1, var + 1, f'{var:.1f}%', ha='center', fontsize=11)
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, len(cumulative_var) + 1))
        ax2.axhline(y=cumulative_var[-1], color='r', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('results/pca_variance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_3d_clusters(self, labels: Optional[np.ndarray] = None, 
                        figsize: Tuple[int, int] = (10, 8)):
        """
        Plot 3D visualization of PCA clusters.
        
        Parameters
        ----------
        labels : array, optional
            Cluster labels
        figsize : tuple
            Figure size
        """
        if self.X_pca is None or self.X_pca.shape[1] < 3:
            print("Need at least 3 PCA components")
            return
        
        if labels is None:
            # Cluster the data
            kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.X_pca)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], self.X_pca[:, 2],
                            c=labels, cmap='Set1', s=50, alpha=0.7, edgecolors='w')
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
        ax.set_title('3D PCA Clusters (Gene Expression Data)', fontsize=14, fontweight='bold')
        
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig('results/pca_3d_clusters.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def full_analysis(self, filepath: Optional[str] = None, n_components: int = 3) -> Dict:
        """
        Run complete PCA analysis pipeline.
        
        Parameters
        ----------
        filepath : str, optional
            Path to data file
        n_components : int
            Number of PCA components
            
        Returns
        -------
        dict
            Complete analysis results
        """
        print("=" * 60)
        print("GENE EXPRESSION PCA ANALYSIS")
        print("=" * 60)
        
        # Load and prepare data
        self.load_data(filepath)
        self.prepare_data()
        
        # Apply PCA
        self.apply_pca(n_components)
        
        # Variance analysis
        variance_results = self.get_variance_explained()
        
        # Clustering comparison
        comparison_df = self.compare_clustering()
        
        # Detailed comparison for optimal k
        detailed = self.detailed_comparison(k=4)
        
        print("\n" + "=" * 60)
        print("PCA ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            'variance': variance_results,
            'comparison': comparison_df,
            'detailed': detailed
        }


def pca_demo():
    """Demonstration of PCA analysis."""
    print("Gene Expression PCA Demo")
    print("-" * 40)
    
    # Create synthetic high-dimensional data
    np.random.seed(42)
    n_samples = 200
    n_features = 1000
    
    # Create data with known structure (3 clusters in 3D subspace)
    # True underlying structure
    true_components = np.random.randn(n_samples, 3)
    true_components[:70] += [3, 0, 0]  # Cluster 1
    true_components[70:140] += [0, 3, 0]  # Cluster 2
    true_components[140:] += [0, 0, 3]  # Cluster 3
    
    # Project to high dimensions with noise
    projection = np.random.randn(3, n_features)
    X = true_components @ projection + np.random.randn(n_samples, n_features) * 0.5
    
    print(f"Synthetic data: {n_samples} samples × {n_features} features")
    
    # Apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA Results:")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    
    # Compare clustering
    print("\nClustering Comparison:")
    for k in [3, 4]:
        kmeans_orig = KMeans(n_clusters=k, random_state=42)
        kmeans_pca = KMeans(n_clusters=k, random_state=42)
        
        labels_orig = kmeans_orig.fit_predict(X_scaled)
        labels_pca = kmeans_pca.fit_predict(X_pca)
        
        sil_orig = silhouette_score(X_scaled, labels_orig)
        sil_pca = silhouette_score(X_pca, labels_pca)
        
        print(f"\n  k={k}:")
        print(f"    Original: Silhouette = {sil_orig:.4f}")
        print(f"    PCA:      Silhouette = {sil_pca:.4f}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    pca_demo()
