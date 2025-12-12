"""
Data Preprocessing, Clustering & PCA Analysis

A comprehensive machine learning toolkit for:
- Data preprocessing pipelines
- K-Means clustering with Silhouette analysis
- PCA dimensionality reduction
- Visualization utilities

Author: Victor Prefa
Course: SIG720 Machine Learning - Deakin University
"""

from .preprocessing import MicroclimatePreprocessor
from .clustering import ObesityClusterer
from .pca_analysis import GeneExpressionPCA

__all__ = [
    'MicroclimatePreprocessor',
    'ObesityClusterer', 
    'GeneExpressionPCA'
]

__version__ = '1.0.0'
__author__ = 'Victor Prefa'
