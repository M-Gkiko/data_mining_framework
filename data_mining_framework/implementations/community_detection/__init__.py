"""Community detection algorithm implementations."""

from .louvain import LouvainCommunityDetection
from .label_propagation import LabelPropagationCommunity
from .girvan_newman import GirvanNewmanCommunity

__all__ = [
    'LouvainCommunityDetection',
    'LabelPropagationCommunity',
    'GirvanNewmanCommunity'
]
