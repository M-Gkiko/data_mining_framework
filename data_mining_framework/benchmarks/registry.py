"""
Registry for algorithms and adapters.
"""
from typing import Dict, Any, Optional

from ..core.dataset import Dataset
from ..core.distance_measure import DistanceMeasure
from ..core.network import Network
from ..implementations.pipelines import (
    DRAdapter, ClusteringAdapter, DRQualityAdapter, ClusteringQualityAdapter,
    NetworkAdapter, CommunityDetectionAdapter, NodeMeasureAdapter, EdgeMeasureAdapter
)


# Simple lookup tables - easy to understand and extend
ALGORITHMS = {
    "clustering": {
        "Hierarchical": None,  # Lazy loaded
        "DBSCAN": None,
        "KMeans": None,
    },
    "dimensionality_reduction": {
        "PCA": None,
        "MDS": None,
        "TSNE": None,
    },
    "community_detection": {
        "Louvain": None,
        "GirvanNewman": None,
        "LabelPropagation": None,
    },
    "node_measures": {
        "PageRank": None,
        "DegreeCentrality": None,
        "ClosenessCentrality": None,
    },
    "edge_measures": {
        "EdgeBetweenness": None,
        "EdgeWeight": None,
        "JaccardCoefficient": None,
    }
}

# Network implementations
NETWORKS = {
    "NetworkX": None,
    "EdgeList": None,
    "AdjacencyMatrix": None,
}

QUALITY_MEASURES = {
    "Calinski_Harabasz": None,  # Lazy loaded
    "Davies_Bouldin": None,
    "Silhouette": None,
    "Trustworthiness": None,
    "Continuity": None,
    "Reconstruction_Error": None,
}

# Distance measures
DISTANCE_MEASURES = {
    "Manhattan": None,
    "Euclidean": None,
    "Cosine": None,
}

# Simple adapter mapping
ADAPTERS = {
    "clustering": ClusteringAdapter,
    "dimensionality_reduction": DRAdapter,
    "clustering_quality": ClusteringQualityAdapter,
    "dr_quality": DRQualityAdapter,
    "network": NetworkAdapter,
    "community_detection": CommunityDetectionAdapter,
    "node_measures": NodeMeasureAdapter,
    "edge_measures": EdgeMeasureAdapter,
}

_loaded = False


def _load_algorithms():
    """Load algorithms only when first needed."""
    global _loaded
    if _loaded:
        return

    # Import clustering algorithms
    from ..implementations.clustering.hierarchical import HierarchicalClustering
    from ..implementations.clustering.dbscan import DBSCANClustering
    from ..implementations.clustering.kmeans import KMeansClustering

    # Import DR algorithms
    from ..implementations.dr.pca_projection import PCAProjection
    from ..implementations.dr.mds_projection import MDSProjection
    from ..implementations.dr.tsne_projection import TSNEProjection

    # Import clustering quality measures
    from ..implementations.clustering.quality.calinski_harabasz import CalinskiHarabaszIndex
    from ..implementations.clustering.quality.davies_bouldin import DaviesBouldinIndex
    from ..implementations.clustering.quality.silhouette import Silhouette

    # Import DR quality measures
    from ..implementations.dr.quality.trustworthiness import Trustworthiness
    from ..implementations.dr.quality.continuity import Continuity
    from ..implementations.dr.quality.reconstruction_error import ReconstructionError

    # Import distance measures
    from ..implementations.distance.manhattan import ManhattanDistance
    from ..implementations.distance.euclidean import EuclideanDistance
    from ..implementations.distance.cosine import CosineDistance

    # Import network implementations
    from ..implementations.networks.networkx_wrapper import NetworkXWrapper
    from ..implementations.networks.edgelist import EdgeListNetwork
    from ..implementations.networks.adjacency_matrix import AdjacencyMatrixNetwork

    # Import community detection algorithms
    from ..implementations.community_detection.louvain import LouvainCommunityDetection
    from ..implementations.community_detection.girvan_newman import GirvanNewmanCommunity
    from ..implementations.community_detection.label_propagation import LabelPropagationCommunity

    # Import node measures
    from ..implementations.node_measures.pagerank import PageRankMeasure
    from ..implementations.node_measures.degree_centrality import DegreeCentralityMeasure
    from ..implementations.node_measures.closeness_centrality import ClosenessCentralityMeasure

    # Import edge measures
    from ..implementations.edge_measures.betweenness import EdgeBetweennessMeasure
    from ..implementations.edge_measures.weight import EdgeWeightMeasure
    from ..implementations.edge_measures.jaccard import JaccardCoefficientMeasure

    # Populate clustering algorithms
    ALGORITHMS["clustering"]["Hierarchical"] = HierarchicalClustering
    ALGORITHMS["clustering"]["DBSCAN"] = DBSCANClustering
    ALGORITHMS["clustering"]["KMeans"] = KMeansClustering

    # Populate DR algorithms
    ALGORITHMS["dimensionality_reduction"]["PCA"] = PCAProjection
    ALGORITHMS["dimensionality_reduction"]["MDS"] = MDSProjection
    ALGORITHMS["dimensionality_reduction"]["TSNE"] = TSNEProjection

    # Populate community detection algorithms
    ALGORITHMS["community_detection"]["Louvain"] = LouvainCommunityDetection
    ALGORITHMS["community_detection"]["GirvanNewman"] = GirvanNewmanCommunity
    ALGORITHMS["community_detection"]["LabelPropagation"] = LabelPropagationCommunity

    # Populate node measures
    ALGORITHMS["node_measures"]["PageRank"] = PageRankMeasure
    ALGORITHMS["node_measures"]["DegreeCentrality"] = DegreeCentralityMeasure
    ALGORITHMS["node_measures"]["ClosenessCentrality"] = ClosenessCentralityMeasure

    # Populate edge measures
    ALGORITHMS["edge_measures"]["EdgeBetweenness"] = EdgeBetweennessMeasure
    ALGORITHMS["edge_measures"]["EdgeWeight"] = EdgeWeightMeasure
    ALGORITHMS["edge_measures"]["JaccardCoefficient"] = JaccardCoefficientMeasure

    # Populate quality measures
    QUALITY_MEASURES["Calinski_Harabasz"] = CalinskiHarabaszIndex
    QUALITY_MEASURES["Davies_Bouldin"] = DaviesBouldinIndex
    QUALITY_MEASURES["Silhouette"] = Silhouette
    QUALITY_MEASURES["Trustworthiness"] = Trustworthiness
    QUALITY_MEASURES["Continuity"] = Continuity
    QUALITY_MEASURES["Reconstruction_Error"] = ReconstructionError

    # Populate distance measures
    DISTANCE_MEASURES["Manhattan"] = ManhattanDistance
    DISTANCE_MEASURES["Euclidean"] = EuclideanDistance
    DISTANCE_MEASURES["Cosine"] = CosineDistance

    # Populate network implementations
    NETWORKS["NetworkX"] = NetworkXWrapper
    NETWORKS["EdgeList"] = EdgeListNetwork
    NETWORKS["AdjacencyMatrix"] = AdjacencyMatrixNetwork

    _loaded = True


def create_algorithm(step_type: str, algorithm_name: str, **params) -> Any:
    """Create algorithm - dead simple."""
    _load_algorithms()
    
    # Handle quality measures
    if step_type in ["clustering_quality", "dr_quality"]:
        if algorithm_name not in QUALITY_MEASURES:
            available = list(QUALITY_MEASURES.keys())
            raise ValueError(f"Unknown quality measure: {algorithm_name}. Available: {available}")
        return QUALITY_MEASURES[algorithm_name](**params)
    
    # Handle regular algorithms
    if step_type not in ALGORITHMS:
        available_types = list(ALGORITHMS.keys())
        raise ValueError(f"Unknown step type: {step_type}. Available: {available_types}")
    
    if algorithm_name not in ALGORITHMS[step_type]:
        available = list(ALGORITHMS[step_type].keys())
        raise ValueError(f"Unknown {step_type} algorithm: {algorithm_name}. Available: {available}")
    
    algorithm_class = ALGORITHMS[step_type][algorithm_name]
    return algorithm_class(**params)


def create_distance_measure(distance_name: str) -> DistanceMeasure:
    """Create distance measure - same pattern as algorithms."""
    _load_algorithms()
    
    if distance_name not in DISTANCE_MEASURES:
        available = list(DISTANCE_MEASURES.keys())
        raise ValueError(f"Unknown distance measure: {distance_name}. Available: {available}")
    
    return DISTANCE_MEASURES[distance_name]()


def create_network(network_type: str, **params) -> Network:
    """Create network instance."""
    _load_algorithms()

    if network_type not in NETWORKS:
        available = list(NETWORKS.keys())
        raise ValueError(f"Unknown network type: {network_type}. Available: {available}")

    network_class = NETWORKS[network_type]
    return network_class(**params)


def create_adapter(step_type: str, algorithm, distance_measure: Optional[DistanceMeasure] = None,
                  dataset: Optional[Dataset] = None):
    """Create adapter - dead simple."""
    if step_type not in ADAPTERS:
        available = list(ADAPTERS.keys())
        raise ValueError(f"Unknown step type: {step_type}. Available: {available}")

    adapter_class = ADAPTERS[step_type]

    # Handle network adapters (different constructor signature)
    if step_type == "network":
        return adapter_class(algorithm, name=f"{step_type}_{algorithm.__class__.__name__}")
    # Handle community detection, node measure, edge measure adapters
    elif step_type in ["community_detection", "node_measure", "edge_measure"]:
        return adapter_class(algorithm, name=f"{step_type}_{algorithm.__class__.__name__}")
    # Handle dr_quality adapters which have different constructor signature
    elif step_type == "dr_quality":
        return adapter_class(algorithm, dataset, distance_measure=distance_measure,
                           name=f"{step_type}_{algorithm.__class__.__name__}")
    else:
        return adapter_class(algorithm, distance_measure=distance_measure,
                           name=f"{step_type}_{algorithm.__class__.__name__}")


def get_available_algorithms() -> Dict[str, list]:
    """Get what's available."""
    _load_algorithms()
    result = {}
    for step_type, algos in ALGORITHMS.items():
        result[step_type] = list(algos.keys())
    result["quality_measures"] = list(QUALITY_MEASURES.keys())
    result["distance_measures"] = list(DISTANCE_MEASURES.keys())
    result["networks"] = list(NETWORKS.keys())
    return result