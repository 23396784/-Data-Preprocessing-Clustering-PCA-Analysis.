"""
Data Preprocessing Module

Comprehensive preprocessing pipeline for microclimate sensor data including:
- Missing value detection and imputation
- Feature engineering (LatLong splitting)
- Min-Max scaling for continuous features
- Distribution visualization

Author: Victor Prefa
Course: SIG720 Machine Learning - Deakin University
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional


class MicroclimatePreprocessor:
    """
    Preprocessor for microclimate sensor data.
    
    Handles missing values, feature engineering, and scaling for
    IoT sensor data with environmental measurements.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the microclimate CSV file
        """
        self.filepath = filepath
        self.df = None
        self.df_original = None
        self.scaler = MinMaxScaler()
        self.replacement_values = {}
        
        # Define feature types
        self.continuous_features = [
            'MinimumWindDirection', 'AverageWindDirection', 'MaximumWindDirection',
            'MinimumWindSpeed', 'AverageWindSpeed', 'GustWindSpeed',
            'AirTemperature', 'RelativeHumidity', 'AtmosphericPressure',
            'PM25', 'PM10', 'Noise'
        ]
        
        self.categorical_features = ['SensorLocation', 'LatLong', 'Device_id']
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the microclimate dataset.
        
        Parameters
        ----------
        filepath : str, optional
            Path to CSV file (uses instance filepath if not provided)
            
        Returns
        -------
        pd.DataFrame
            Loaded dataset
        """
        path = filepath or self.filepath
        if path is None:
            raise ValueError("No filepath provided")
            
        self.df = pd.read_csv(path)
        self.df_original = self.df.copy()
        
        print(f"Dataset loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        return self.df
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Returns
        -------
        pd.DataFrame
            Summary of missing values per feature
        """
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        
        missing_summary = pd.DataFrame({
            'Feature': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_pct.values
        })
        
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
        missing_summary = missing_summary.sort_values('Missing_Count', ascending=False)
        
        print("\n=== Missing Values Analysis ===")
        print(f"Total missing entries: {missing_counts.sum():,}")
        print(f"Features with missing values: {len(missing_summary)}")
        print("\nMissing values by feature:")
        for _, row in missing_summary.iterrows():
            print(f"  {row['Feature']}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.2f}%)")
            
        return missing_summary
    
    def impute_missing_values(self, strategy: str = 'median') -> Dict[str, float]:
        """
        Impute missing values using specified strategy.
        
        Parameters
        ----------
        strategy : str
            Imputation strategy: 'median' or 'mean' for numerical features
            Categorical features always use mode
            
        Returns
        -------
        dict
            Dictionary of replacement values used
        """
        print(f"\n=== Imputing Missing Values (strategy: {strategy}) ===")
        
        # Impute numerical features
        for col in self.continuous_features:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                if strategy == 'median':
                    replacement = self.df[col].median()
                else:
                    replacement = self.df[col].mean()
                    
                self.df[col].fillna(replacement, inplace=True)
                self.replacement_values[col] = replacement
                print(f"  {col}: {replacement:.4f} ({strategy})")
        
        # Impute categorical features with mode
        for col in self.categorical_features:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                self.replacement_values[col] = mode_value
                print(f"  {col}: '{mode_value}' (mode)")
        
        # Verify no missing values remain
        remaining = self.df.isnull().sum().sum()
        print(f"\nRemaining missing values: {remaining}")
        
        return self.replacement_values
    
    def split_latlong(self) -> pd.DataFrame:
        """
        Split LatLong column into separate Latitude and Longitude columns.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with new Latitude and Longitude columns
        """
        if 'LatLong' not in self.df.columns:
            print("LatLong column not found")
            return self.df
            
        print("\n=== Splitting LatLong Column ===")
        
        # Split the LatLong string
        latlong_split = self.df['LatLong'].str.split(',', expand=True)
        
        self.df['Latitude'] = latlong_split[0].astype(float)
        self.df['Longitude'] = latlong_split[1].astype(float)
        
        print(f"  Latitude range: [{self.df['Latitude'].min():.6f}, {self.df['Latitude'].max():.6f}]")
        print(f"  Longitude range: [{self.df['Longitude'].min():.6f}, {self.df['Longitude'].max():.6f}]")
        
        return self.df
    
    def apply_minmax_scaling(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply Min-Max scaling to continuous features.
        
        Parameters
        ----------
        features : list, optional
            List of features to scale (uses continuous_features if not provided)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with scaled features
        """
        features = features or self.continuous_features
        features = [f for f in features if f in self.df.columns]
        
        print(f"\n=== Applying Min-Max Scaling to {len(features)} features ===")
        
        # Store original values for comparison
        self.df_before_scaling = self.df[features].copy()
        
        # Apply scaling
        self.df[features] = self.scaler.fit_transform(self.df[features])
        
        print("Scaled features now in range [0, 1]:")
        for feat in features[:5]:  # Show first 5
            print(f"  {feat}: [{self.df[feat].min():.4f}, {self.df[feat].max():.4f}]")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more features")
            
        return self.df
    
    def plot_distributions(self, features: Optional[List[str]] = None, 
                          n_cols: int = 3, figsize: Tuple[int, int] = (15, 12)):
        """
        Plot feature distributions before and after scaling.
        
        Parameters
        ----------
        features : list, optional
            Features to plot
        n_cols : int
            Number of columns in subplot grid
        figsize : tuple
            Figure size
        """
        if not hasattr(self, 'df_before_scaling'):
            print("Run apply_minmax_scaling() first")
            return
            
        features = features or list(self.df_before_scaling.columns)[:6]
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, feat in enumerate(features):
            # Before scaling
            ax_before = axes[i * 2]
            ax_before.hist(self.df_before_scaling[feat].dropna(), bins=30, 
                          color='steelblue', edgecolor='black', alpha=0.7)
            ax_before.set_title(f'{feat}\n(Before Scaling)', fontsize=10)
            ax_before.set_xlabel('Value')
            ax_before.set_ylabel('Frequency')
            
            # After scaling
            ax_after = axes[i * 2 + 1]
            ax_after.hist(self.df[feat].dropna(), bins=30, 
                         color='coral', edgecolor='black', alpha=0.7)
            ax_after.set_title(f'{feat}\n(After Scaling)', fontsize=10)
            ax_after.set_xlabel('Value')
            ax_after.set_ylabel('Frequency')
        
        # Hide unused subplots
        for j in range(n_features * 2, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Feature Distributions: Before vs After Min-Max Scaling', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/distribution_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n=== Scaling Observations ===")
        print("1. Shape: Distribution shapes are preserved (not changed by scaling)")
        print("2. Range: All values now in [0, 1] range")
        print("3. Central Tendency: Relative positions maintained")
        print("4. Comparability: Features now on same scale for ML algorithms")
    
    def full_pipeline(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Parameters
        ----------
        filepath : str, optional
            Path to data file
            
        Returns
        -------
        pd.DataFrame
            Fully preprocessed dataset
        """
        print("=" * 60)
        print("MICROCLIMATE DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_data(filepath)
        
        # Analyze missing values
        self.analyze_missing_values()
        
        # Impute missing values (median for robustness)
        self.impute_missing_values(strategy='median')
        
        # Split LatLong
        self.split_latlong()
        
        # Apply scaling
        self.apply_minmax_scaling()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print(f"Final dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print("=" * 60)
        
        return self.df
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics of the preprocessed data.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        return self.df.describe()


def preprocess_demo():
    """Demonstration of the preprocessing pipeline."""
    print("Microclimate Data Preprocessing Demo")
    print("-" * 40)
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'Device_id': np.random.choice(['Sensor_A', 'Sensor_B', 'Sensor_C'], n_samples),
        'LatLong': [f'{-37.8 + np.random.randn()*0.01:.6f}, {144.9 + np.random.randn()*0.01:.6f}' 
                   for _ in range(n_samples)],
        'AirTemperature': np.random.uniform(10, 35, n_samples),
        'RelativeHumidity': np.random.uniform(30, 90, n_samples),
        'PM25': np.random.exponential(15, n_samples),
        'AverageWindSpeed': np.random.weibull(2, n_samples) * 5
    })
    
    # Add some missing values
    for col in ['AirTemperature', 'PM25']:
        mask = np.random.random(n_samples) < 0.05
        sample_data.loc[mask, col] = np.nan
    
    # Initialize preprocessor
    preprocessor = MicroclimatePreprocessor()
    preprocessor.df = sample_data
    preprocessor.continuous_features = ['AirTemperature', 'RelativeHumidity', 'PM25', 'AverageWindSpeed']
    
    # Run pipeline steps
    preprocessor.analyze_missing_values()
    preprocessor.impute_missing_values(strategy='median')
    preprocessor.split_latlong()
    preprocessor.apply_minmax_scaling()
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    preprocess_demo()
