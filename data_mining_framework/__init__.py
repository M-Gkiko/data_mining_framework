"""
Data Mining Framework - A comprehensive framework for data mining algorithm benchmarking and evaluation.

This framework provides a unified interface for:
- Dimensionality Reduction algorithms (PCA, MDS, t-SNE, Sammon Mapping)
- Clustering algorithms (Hierarchical, DBSCAN, K-Means)
- Network Analysis (Community Detection, Node Measures, Edge Measures)
- Quality measures for DR and clustering
- Distance measures (Manhattan, Euclidean, Cosine)
- Benchmarking pipelines and result analysis

Example usage:
    # Run a benchmark
    from data_mining_framework import run_benchmark
    results = run_benchmark('configs/sample_configs/clustering_benchmark.yaml')

    # Use individual components
    from data_mining_framework.core import Dataset
    from data_mining_framework.implementations.clustering import HierarchicalClustering
    from data_mining_framework.implementations.distance import ManhattanDistance

    dataset = CSVDataset('data/iris.csv')
    distance = ManhattanDistance()
    clustering = HierarchicalClustering(distance_measure=distance, n_clusters=3)
    clustering.fit(dataset)
    labels = clustering.get_labels()

    # Network analysis
    from data_mining_framework import NetworkXWrapper, LouvainCommunityDetection
    network = NetworkXWrapper.from_edgelist('data/network.edgelist')
    community_detection = LouvainCommunityDetection()
    community_detection.fit(network)
    communities = community_detection.get_communities()
"""

__version__ = "0.1.0"
__author__ = "Mario"
__email__ = "your.email@example.com"

# Core interfaces
from .core.dataset import Dataset
from .core.distance_measure import DistanceMeasure
from .core.clustering import Clustering
from .core.dimensionality_reduction import DimensionalityReduction
from .core.clustering_quality_measure import ClusteringQualityMeasure
from .core.dimensionality_reduction_quality_measure import DRQualityMeasure
from .core.pipeline import Pipeline, PipelineComponent
from .core.network import Network
from .core.community_detection import CommunityDetection
from .core.node_measure import NodeMeasure
from .core.edge_measure import EdgeMeasure

# Dataset implementations
from .implementations.datasets import CSVDataset, NumpyDataset

# Distance measure implementations
from .implementations.distance.manhattan import ManhattanDistance
from .implementations.distance.euclidean import EuclideanDistance
from .implementations.distance.cosine import CosineDistance  

# Clustering implementations
from .implementations.clustering.hierarchical import HierarchicalClustering
from .implementations.clustering.dbscan import DBSCANClustering
from .implementations.clustering.kmeans import KMeansClustering  

# DR implementations
from .implementations.dr.pca_projection import PCAProjection
from .implementations.dr.mds_projection import MDSProjection
from .implementations.dr.tsne_projection import TSNEProjection
from .implementations.dr.sammon_mapping import SammonMapping

# Network implementations
from .implementations.networks.edgelist import EdgeListNetwork
from .implementations.networks.adjacency_matrix import AdjacencyMatrixNetwork
from .implementations.networks.networkx_wrapper import NetworkXWrapper

# Community detection implementations
from .implementations.community_detection.louvain import LouvainCommunityDetection
from .implementations.community_detection.label_propagation import LabelPropagationCommunity
from .implementations.community_detection.girvan_newman import GirvanNewmanCommunity

# Node measure implementations
from .implementations.node_measures.pagerank import PageRankMeasure
from .implementations.node_measures.degree_centrality import DegreeCentralityMeasure
from .implementations.node_measures.closeness_centrality import ClosenessCentralityMeasure

# Edge measure implementations
from .implementations.edge_measures.betweenness import EdgeBetweennessMeasure
from .implementations.edge_measures.weight import EdgeWeightMeasure
from .implementations.edge_measures.jaccard import JaccardCoefficientMeasure

# Quality measure implementations
from .implementations.clustering.quality.calinski_harabasz import CalinskiHarabaszIndex
from .implementations.clustering.quality.davies_bouldin import DaviesBouldinIndex
from .implementations.dr.quality.trustworthiness import Trustworthiness
from .implementations.dr.quality.continuity import Continuity
from .implementations.dr.quality.reconstruction_error import ReconstructionError

# Benchmarking
from .benchmarks.core import BenchmarkConfig, SimpleBenchmark
from .benchmarks.utils import load_benchmark_config, export_benchmark_results

# Pipeline adapters
from .implementations.pipelines.clustering_adapter import ClusteringAdapter
from .implementations.pipelines.dr_adapter import DRAdapter
from .implementations.pipelines.clustering_quality_adapter import ClusteringQualityAdapter
from .implementations.pipelines.dr_quality_adapter import DRQualityAdapter
from .implementations.pipelines.network_adapter import NetworkAdapter
from .implementations.pipelines.community_adapter import CommunityDetectionAdapter
from .implementations.pipelines.node_measure_adapter import NodeMeasureAdapter
from .implementations.pipelines.edge_measure_adapter import EdgeMeasureAdapter

# Convenience function
def run_benchmark(config_path: str, verbose: bool = False):
    """
    Run a benchmark from a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        verbose: Whether to print verbose output
        
    Returns:
        BenchmarkResults object with timing and quality data
    """
    config = load_benchmark_config(config_path)
    benchmark = SimpleBenchmark(config)
    dataset = CSVDataset(config.dataset_path)
    return benchmark.run(dataset)

# Export main classes for easy access
__all__ = [
    # Core interfaces
    'Dataset', 'DistanceMeasure', 'Clustering', 'DimensionalityReduction',
    'ClusteringQualityMeasure', 'DRQualityMeasure', 'Pipeline', 'PipelineComponent',
    'Network', 'CommunityDetection', 'NodeMeasure', 'EdgeMeasure',

    # Dataset implementations
    'CSVDataset', 'NumpyDataset',

    # Distance measures
    'ManhattanDistance', 'EuclideanDistance', 'CosineDistance',

    # Clustering algorithms
    'HierarchicalClustering', 'DBSCANClustering', 'KMeansClustering',

    # DR algorithms
    'PCAProjection', 'MDSProjection', 'TSNEProjection', 'SammonMapping',

    # Network implementations
    'EdgeListNetwork', 'AdjacencyMatrixNetwork', 'NetworkXWrapper',

    # Community detection algorithms
    'LouvainCommunityDetection', 'LabelPropagationCommunity', 'GirvanNewmanCommunity',

    # Node measures
    'PageRankMeasure', 'DegreeCentralityMeasure', 'ClosenessCentralityMeasure',

    # Edge measures
    'EdgeBetweennessMeasure', 'EdgeWeightMeasure', 'JaccardCoefficientMeasure',

    # Quality measures
    'CalinskiHarabaszIndex', 'DaviesBouldinIndex',
    'Trustworthiness', 'Continuity', 'ReconstructionError',

    # Benchmarking
    'BenchmarkConfig', 'SimpleBenchmark', 'run_benchmark',
    'load_benchmark_config', 'export_benchmark_results',

    # Adapters
    'ClusteringAdapter', 'DRAdapter', 'ClusteringQualityAdapter', 'DRQualityAdapter',
    'NetworkAdapter', 'CommunityDetectionAdapter', 'NodeMeasureAdapter', 'EdgeMeasureAdapter',
]