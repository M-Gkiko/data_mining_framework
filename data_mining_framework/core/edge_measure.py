from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from .network import Network


class EdgeMeasure(ABC):
    """
    Abstract base class for edge-level measures in networks.

    This interface defines the contract that all edge measure implementations
    must follow, enabling the Strategy Pattern for edge analysis operations.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the edge measure with configuration parameters.

        Args:
            **kwargs: Algorithm-specific hyperparameters
        """
        pass

    @abstractmethod
    def calculate(self, network: Network, **kwargs: Any) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate the measure for all edges in the network.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dict[Tuple[Any, Any], float]: Dictionary mapping edge tuples to their measure values

        Raises:
            ValueError: If network is invalid or empty
        """
        pass
