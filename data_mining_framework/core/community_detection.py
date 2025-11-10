from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from .network import Network


class CommunityDetection(ABC):
    """
    Abstract base class for community detection algorithms.

    This interface defines the contract that all community detection implementations
    must follow, enabling the Strategy Pattern for community detection operations.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the community detection algorithm with configuration parameters.

        Args:
            **kwargs: Algorithm-specific hyperparameters
        """
        pass

    @abstractmethod
    def fit(self, network: Network, **kwargs: Any) -> None:
        """
        Detect communities in the given network.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional algorithm-specific parameters

        Raises:
            ValueError: If network is invalid or empty
        """
        pass

    @abstractmethod
    def get_communities(self) -> Union[List[List[Any]], Dict[Any, int]]:
        """
        Get the detected communities.

        Returns:
            Union[List[List[Any]], Dict[Any, int]]:
                Either a list of communities (each community is a list of nodes)
                or a dictionary mapping nodes to community labels
        """
        pass

    @abstractmethod
    def get_modularity(self) -> Optional[float]:
        """
        Get the modularity score of the detected communities.

        Returns:
            Optional[float]: Modularity score, or None if not applicable
        """
        pass
