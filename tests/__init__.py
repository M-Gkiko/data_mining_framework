"""
Test package initialization file.
"""

import sys
import os

# Add the parent directory to the Python path to allow importing the core module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

__all__ = [
    'test_dataset',
    'test_distance_measure', 
    'test_clustering_algorithm',
    'test_quality_measure'
]