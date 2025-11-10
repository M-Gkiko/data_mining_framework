"""Pipeline adapter for Community Detection components."""

from typing import Any, Dict, Union, List
from ...core.pipeline import PipelineComponent
from ...core.community_detection import CommunityDetection
from ...core.network import Network


class CommunityDetectionAdapter(PipelineComponent):
    """
    Adapter that wraps Community Detection algorithms for pipeline execution.

    Handles Network inputs and returns community detection results.
    """

    def __init__(self, community_algorithm: CommunityDetection, name: str = None, **kwargs):
        """
        Initialize community detection adapter.

        Args:
            community_algorithm: The community detection algorithm to wrap
            name: Name for this component in the pipeline
            **kwargs: Algorithm-specific parameters
        """
        self.community_algorithm = community_algorithm
        self.algorithm_params = kwargs

        component_name = name or f"CommunityDetection_{community_algorithm.__class__.__name__}"
        super().__init__(component_name)

    def execute(self, input_data: Network) -> Dict[str, Any]:
        """
        Execute community detection algorithm.

        Args:
            input_data: Network object

        Returns:
            Dict containing:
                - 'communities': Detected communities (list or dict)
                - 'modularity': Modularity score (if available)
                - 'algorithm': Reference to fitted algorithm
                - 'network': Network object used

        Raises:
            ValueError: If input data is not a Network
            RuntimeError: If algorithm execution fails
        """
        # Validate input
        if not isinstance(input_data, Network):
            raise ValueError(f"Expected Network input, got {type(input_data)}")

        # Run community detection
        try:
            self.community_algorithm.fit(input_data, **self.algorithm_params)
        except Exception as e:
            raise RuntimeError(f"Community detection failed: {str(e)}") from e

        # Get results
        communities = self.community_algorithm.get_communities()
        modularity = self.community_algorithm.get_modularity()

        return {
            'communities': communities,
            'modularity': modularity,
            'algorithm': self.community_algorithm,
            'network': input_data
        }
