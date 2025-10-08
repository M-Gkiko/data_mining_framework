"""
Registry for algorithms and adapters.
"""
from typing import Dict, Any, Optional

from core.dataset import Dataset
from core.distance_measure import DistanceMeasure
from implementations.pipelines import DRAdapter, ClusteringAdapter, DRQualityAdapter, ClusteringQualityAdapter


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
    }
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
}

_loaded = False


def _load_algorithms():
    """Load algorithms only when first needed."""
    global _loaded
    if _loaded:
        return
    
    # Import clustering algorithms
    from implementations.clustering.hierarchical import HierarchicalClustering
    from implementations.clustering.dbscan import DBSCANClustering
    #from implementations.clustering.kmeans import KMeansClustering
    
    # Import DR algorithms
    from implementations.dr.pca_projection import PCAProjection
    from implementations.dr.mds_projection import MDSProjection
    from implementations.dr.tsne_projection import TSNEProjection
    
    # Import clustering quality measures
    from implementations.clustering.quality.calinski_harabasz import CalinskiHarabaszIndex
    from implementations.clustering.quality.davies_bouldin import DaviesBouldinIndex
    #from implementations.clustering.quality.silhouette import SilhouetteScore
    
    # Import DR quality measures
    from implementations.dr.quality.trustworthiness import Trustworthiness
    from implementations.dr.quality.continuity import Continuity
    from implementations.dr.quality.reconstruction_error import ReconstructionError
    
    # Import distance measures
    from implementations.distance.manhattan import ManhattanDistance
    #from implementations.distance.euclidean import EuclideanDistance
    from implementations.distance.cosine import CosineDistance
    
    # Populate clustering algorithms
    ALGORITHMS["clustering"]["Hierarchical"] = HierarchicalClustering
    ALGORITHMS["clustering"]["DBSCAN"] = DBSCANClustering
    #ALGORITHMS["clustering"]["KMeans"] = KMeansClustering
    
    # Populate DR algorithms
    ALGORITHMS["dimensionality_reduction"]["PCA"] = PCAProjection
    ALGORITHMS["dimensionality_reduction"]["MDS"] = MDSProjection
    ALGORITHMS["dimensionality_reduction"]["TSNE"] = TSNEProjection
    
    # Populate quality measures
    QUALITY_MEASURES["Calinski_Harabasz"] = CalinskiHarabaszIndex
    QUALITY_MEASURES["Davies_Bouldin"] = DaviesBouldinIndex
    #QUALITY_MEASURES["Silhouette"] = SilhouetteScore
    QUALITY_MEASURES["Trustworthiness"] = Trustworthiness
    QUALITY_MEASURES["Continuity"] = Continuity
    QUALITY_MEASURES["Reconstruction_Error"] = ReconstructionError
    
    # Populate distance measures
    DISTANCE_MEASURES["Manhattan"] = ManhattanDistance
    #DISTANCE_MEASURES["Euclidean"] = EuclideanDistance
    DISTANCE_MEASURES["Cosine"] = CosineDistance
    
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


def create_adapter(step_type: str, algorithm, distance_measure: Optional[DistanceMeasure] = None, 
                  dataset: Optional[Dataset] = None):
    """Create adapter - dead simple."""
    if step_type not in ADAPTERS:
        available = list(ADAPTERS.keys())
        raise ValueError(f"Unknown step type: {step_type}. Available: {available}")
    
    adapter_class = ADAPTERS[step_type]
    
    # Handle dr_quality adapters which have different constructor signature
    if step_type == "dr_quality":
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
    return result