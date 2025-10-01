"""
Core module for the data mining framework.

This package provides the abstract base classes (interfaces) that define
the contracts for all components in the framework, following the Strategy Pattern.

The framework is designed around four main interfaces:
- Dataset: For data source abstraction
- DistanceMeasure: For distance/similarity calculations
- ClusteringAlgorithm: For clustering implementations
- QualityMeasure: For clustering quality evaluation

Example usage:
    from core import Dataset, DistanceMeasure, ClusteringAlgorithm, QualityMeasure
"""

from .dataset import Dataset
from .distance_measure import DistanceMeasure
from .clustering_algorithm import ClusteringAlgorithm
from .quality_measure import QualityMeasure

__all__ = [
    'Dataset',
    'DistanceMeasure', 
    'ClusteringAlgorithm',
    'QualityMeasure'
]

__version__ = '0.1.0'
__author__ = 'Data Mining Framework Team'