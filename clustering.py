"""
Clustering Analysis Module

KMeans clustering with Silhouette analysis for obesity dataset:
- Categorical encoding (Label + One-Hot)
- Silhouette coefficient for optimal k selection
- KMeans and KMeans++ comparison
- Cluster visualization

Author: Victor Prefa
Course: SIG720 Machine Learning - Deakin University
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from typing import Tuple, Dict, List, Optional


class ObesityClusterer:
    """
    Clustering analysis for obesity dataset.
    
    Implements:
    - Categorical feature encoding
    - Silhouette-based optimal k selection
    - KMeans vs KMeans++ comparison
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the clusterer.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the obesity CSV file
        """
        self.filepath = filepath
        self.df = None
        self.df_encoded = None
        self.X = None
        self.silhouette_scores = {}
        self.optimal_k = None
        self.labels = None
        
        # Define feature types
        self.binary_features = ['Gender', 'family_history_with_overweight', 
                               'FAVC', 'SMOKE', 'SCC']
        self.ordinal_features = ['CAEC', 'CALC']
        self.nominal_features = ['MTRANS']
        self.target_column = 'NObeyesdad'
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the obesity dataset.
        
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
        print(f"Dataset loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Check for target column
        if self.target_column in self.df.columns:
            print(f"Target distribution:\n{self.df[self.target_column].value_counts()}")
            
        return self.df
    
    def remove_target(self) -> pd.DataFrame:
        """
        Remove the class label column for unsupervised clustering.
        
        Returns
        -------
        pd.DataFrame
            Dataset without target column
        """
        if self.target_column in self.df.columns:
            self.df = self.df.drop(columns=[self.target_column])
            print(f"Removed target column: {self.target_column}")
            
        return self.df
    
    def encode_features(self) -> pd.DataFrame:
        """
        Encode categorical features using appropriate techniques.
        
        - Binary features: Label Encoding (0/1)
        - Ordinal features: Label Encoding with order
        - Nominal features: One-Hot Encoding
        
        Returns
        -------
        pd.DataFrame
            Encoded dataset
        """
        print("\n=== Encoding Categorical Features ===")
        self.df_encoded = self.df.copy()
        
        # Label encode binary features
        label_encoder = LabelEncoder()
        for col in self.binary_features:
            if col in self.df_encoded.columns:
                self.df_encoded[col] = label_encoder.fit_transform(self.df_encoded[col])
                print(f"  {col}: Label Encoded (binary)")
        
        # Label encode ordinal features (preserving order)
        ordinal_mappings = {
            'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        }
        for col, mapping in ordinal_mappings.items():
            if col in self.df_encoded.columns:
                self.df_encoded[col] = self.df_encoded[col].map(mapping)
                print(f"  {col}: Ordinal Encoded (order preserved)")
        
        # One-hot encode nominal features
        for col in self.nominal_features:
            if col in self.df_encoded.columns:
                dummies = pd.get_dummies(self.df_encoded[col], prefix=col, drop_first=True)
                self.df_encoded = pd.concat([self.df_encoded.drop(columns=[col]), dummies], axis=1)
                print(f"  {col}: One-Hot Encoded ({len(dummies.columns)} new columns)")
        
        print(f"\nEncoded dataset shape: {self.df_encoded.shape}")
        
        # Prepare feature matrix
        self.X = self.df_encoded.select_dtypes(include=[np.number]).values
        
        return self.df_encoded
    
    def find_optimal_k(self, k_range: range = range(2, 11)) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of clusters using Silhouette coefficient.
        
        Parameters
        ----------
        k_range : range
            Range of k values to test
            
        Returns
        -------
        tuple
            (optimal_k, silhouette_scores_dict)
        """
        print("\n=== Finding Optimal k using Silhouette Analysis ===")
        
        if self.X is None:
            raise ValueError("Run encode_features() first")
        
        self.silhouette_scores = {}
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.X)
            score = silhouette_score(self.X, labels)
            self.silhouette_scores[k] = score
            print(f"  k={k}: Silhouette Score = {score:.4f}")
        
        # Find optimal k
        self.optimal_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        best_score = self.silhouette_scores[self.optimal_k]
        
        print(f"\n>>> Optimal k = {self.optimal_k} (Silhouette: {best_score:.4f})")
        
        return self.optimal_k, self.silhouette_scores
    
    def compare_kmeans_methods(self, k: Optional[int] = None) -> Dict[str, Dict]:
        """
        Compare KMeans (random init) vs KMeans++ initialization.
        
        Parameters
        ----------
        k : int, optional
            Number of clusters (uses optimal_k if not provided)
            
        Returns
        -------
        dict
            Comparison results for both methods
        """
        k = k or self.optimal_k
        if k is None:
            raise ValueError("Run find_optimal_k() first or provide k")
            
        print(f"\n=== Comparing KMeans vs KMeans++ (k={k}) ===")
        
        results = {}
        
        # KMeans with random initialization
        kmeans_random = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
        labels_random = kmeans_random.fit_predict(self.X)
        score_random = silhouette_score(self.X, labels_random)
        
        results['KMeans (random)'] = {
            'silhouette_score': score_random,
            'inertia': kmeans_random.inertia_,
            'n_iter': kmeans_random.n_iter_,
            'cluster_sizes': np.bincount(labels_random)
        }
        
        # KMeans++ initialization
        kmeans_pp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels_pp = kmeans_pp.fit_predict(self.X)
        score_pp = silhouette_score(self.X, labels_pp)
        
        results['KMeans++'] = {
            'silhouette_score': score_pp,
            'inertia': kmeans_pp.inertia_,
            'n_iter': kmeans_pp.n_iter_,
            'cluster_sizes': np.bincount(labels_pp)
        }
        
        # Store best labels
        if score_pp >= score_random:
            self.labels = labels_pp
        else:
            self.labels = labels_random
        
        # Print comparison
        print("\n┌─────────────────────┬──────────────────┬──────────────────┐")
        print("│ Metric              │ KMeans (random)  │ KMeans++         │")
        print("├─────────────────────┼──────────────────┼──────────────────┤")
        print(f"│ Silhouette Score    │ {score_random:16.4f} │ {score_pp:16.4f} │")
        print(f"│ Inertia             │ {results['KMeans (random)']['inertia']:16.2f} │ {results['KMeans++']['inertia']:16.2f} │")
        print(f"│ Iterations          │ {results['KMeans (random)']['n_iter']:16d} │ {results['KMeans++']['n_iter']:16d} │")
        print("└─────────────────────┴──────────────────┴──────────────────┘")
        
        # Winner
        if score_pp > score_random:
            print("\n>>> KMeans++ achieves better clustering quality")
        elif score_random > score_pp:
            print("\n>>> Random initialization achieves better clustering quality")
        else:
            print("\n>>> Both methods achieve similar performance")
            
        return results
    
    def plot_silhouette_analysis(self, figsize: Tuple[int, int] = (14, 5)):
        """
        Plot Silhouette score curve and optimal k visualization.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        """
        if not self.silhouette_scores:
            print("Run find_optimal_k() first")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Silhouette scores by k
        ax1 = axes[0]
        k_values = list(self.silhouette_scores.keys())
        scores = list(self.silhouette_scores.values())
        
        ax1.plot(k_values, scores, 'b-o', linewidth=2, markersize=8)
        ax1.axvline(x=self.optimal_k, color='r', linestyle='--', 
                   label=f'Optimal k={self.optimal_k}')
        ax1.fill_between(k_values, scores, alpha=0.3)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Silhouette Score', fontsize=12)
        ax1.set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
        ax1.set_xticks(k_values)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster distribution for optimal k
        ax2 = axes[1]
        if self.labels is not None:
            cluster_sizes = np.bincount(self.labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
            bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes, color=colors, edgecolor='black')
            
            # Add value labels
            for bar, size in zip(bars, cluster_sizes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(size), ha='center', va='bottom', fontweight='bold')
            
            ax2.set_xlabel('Cluster', fontsize=12)
            ax2.set_ylabel('Number of Samples', fontsize=12)
            ax2.set_title(f'Cluster Distribution (k={self.optimal_k})', 
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(cluster_sizes)))
        
        plt.tight_layout()
        plt.savefig('results/silhouette_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_silhouette_samples(self, k: Optional[int] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot silhouette diagram showing sample-level scores.
        
        Parameters
        ----------
        k : int, optional
            Number of clusters
        figsize : tuple
            Figure size
        """
        k = k or self.optimal_k
        
        # Fit model
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(self.X)
        
        # Compute silhouette values
        silhouette_avg = silhouette_score(self.X, labels)
        sample_silhouette_values = silhouette_samples(self.X, labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_lower = 10
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values.sort()
            
            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size
            
            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average: {silhouette_avg:.3f}')
        ax.set_xlabel('Silhouette Coefficient', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_title(f'Silhouette Plot for k={k} Clusters', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/silhouette_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_cluster_profiles(self) -> pd.DataFrame:
        """
        Generate cluster profiles showing mean feature values.
        
        Returns
        -------
        pd.DataFrame
            Cluster profiles
        """
        if self.labels is None:
            print("Run compare_kmeans_methods() first")
            return None
            
        self.df_encoded['Cluster'] = self.labels
        profiles = self.df_encoded.groupby('Cluster').mean()
        
        print("\n=== Cluster Profiles ===")
        print(profiles.round(3).T)
        
        return profiles
    
    def full_analysis(self, filepath: Optional[str] = None, k_range: range = range(2, 11)) -> Dict:
        """
        Run complete clustering analysis pipeline.
        
        Parameters
        ----------
        filepath : str, optional
            Path to data file
        k_range : range
            Range of k values to test
            
        Returns
        -------
        dict
            Complete analysis results
        """
        print("=" * 60)
        print("OBESITY DATA CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Load and prepare data
        self.load_data(filepath)
        self.remove_target()
        self.encode_features()
        
        # Find optimal k
        optimal_k, scores = self.find_optimal_k(k_range)
        
        # Compare methods
        comparison = self.compare_kmeans_methods()
        
        print("\n" + "=" * 60)
        print("CLUSTERING ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            'optimal_k': optimal_k,
            'silhouette_scores': scores,
            'comparison': comparison,
            'labels': self.labels
        }


def clustering_demo():
    """Demonstration of clustering analysis."""
    print("Obesity Clustering Demo")
    print("-" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Generate two distinct clusters
    cluster1 = np.random.randn(n_samples // 2, 4) + np.array([0, 0, 0, 0])
    cluster2 = np.random.randn(n_samples // 2, 4) + np.array([3, 3, 3, 3])
    X = np.vstack([cluster1, cluster2])
    
    # Test different k values
    print("\nSilhouette Analysis:")
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"  k={k}: Silhouette = {score:.4f}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    clustering_demo()
