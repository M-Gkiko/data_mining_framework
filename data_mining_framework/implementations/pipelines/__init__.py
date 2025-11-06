"""
Pipeline adapters for wrapping algorithms into pipeline components.
"""

from .dr_adapter import DRAdapter
from .clustering_adapter import ClusteringAdapter
from .dr_quality_adapter import DRQualityAdapter
from .clustering_quality_adapter import ClusteringQualityAdapter
from .network_adapter import NetworkAdapter
from .community_adapter import CommunityDetectionAdapter
from .node_measure_adapter import NodeMeasureAdapter
from .edge_measure_adapter import EdgeMeasureAdapter

__all__ = [
    'DRAdapter',
    'ClusteringAdapter',
    'DRQualityAdapter',
    'ClusteringQualityAdapter',
    'NetworkAdapter',
    'CommunityDetectionAdapter',
    'NodeMeasureAdapter',
    'EdgeMeasureAdapter'
]