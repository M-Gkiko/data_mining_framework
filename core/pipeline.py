"""
Core pipeline interface for data mining components.

This module provides the fundamental pipeline abstractions for chaining
algorithm components together with timing measurement.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict
from utils.timer import Timer


class PipelineComponent(ABC):
    """
    Abstract base class for all pipeline components.
    
    Each component represents a single algorithm wrapped for pipeline execution.
    """
    
    def __init__(self, name: str):
        """
        Initialize component with a name.
        
        Args:
            name: Human-readable name for this component
        """
        self.name = name
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Execute this component with the provided input data.
        
        Args:
            input_data: Input data for this component
            
        Returns:
            Output data from this component
        """
        pass
    
    def get_name(self) -> str:
        """Get the component name."""
        return self.name


class Pipeline:
    """
    Base pipeline class that chains components together with timing.
    
    Executes components sequentially, measuring execution time for each
    component and the overall pipeline.
    """
    
    def __init__(self, name: str):
        """
        Initialize pipeline.
        
        Args:
            name: Name for this pipeline configuration
        """
        self.name = name
        self.components: List[PipelineComponent] = []
        self.execution_times: Dict[str, float] = {}
        self.total_execution_time: float = 0.0
    
    def add_component(self, component: PipelineComponent) -> 'Pipeline':
        """
        Add a component to the pipeline.
        
        Args:
            component: Component to add to the pipeline
            
        Returns:
            Self for method chaining
        """
        self.components.append(component)
        return self
    
    def execute(self, input_data: Any) -> Any:
        """
        Execute the entire pipeline with timing measurement.
        
        Args:
            input_data: Initial input data
            
        Returns:
            Final output from the last component in the pipeline
        """
        if not self.components:
            raise ValueError(f"Pipeline '{self.name}' is empty - add components before executing")
        
        # Reset timing data
        self.execution_times.clear()
        self.total_execution_time = 0.0
        
        current_data = input_data
        overall_timer = Timer("total_pipeline")
        overall_timer.start()
        
        # Execute each component sequentially
        for component in self.components:
            component_timer = Timer(component.get_name())
            component_timer.start()
            
            try:
                current_data = component.execute(current_data)
            except Exception as e:
                raise RuntimeError(
                    f"Error executing component '{component.get_name()}' in pipeline '{self.name}': {str(e)}"
                ) from e
            
            elapsed = component_timer.stop()
            self.execution_times[component.get_name()] = elapsed
        
        self.total_execution_time = overall_timer.stop()
        return current_data
    
    def get_execution_times(self) -> Dict[str, float]:
        """
        Get execution times for each component.
        
        Returns:
            Dictionary mapping component names to execution times in seconds
        """
        return self.execution_times.copy()
    
    def get_total_time(self) -> float:
        """
        Get total pipeline execution time.
        
        Returns:
            Total execution time in seconds
        """
        return self.total_execution_time
    
    def get_component_names(self) -> List[str]:
        """
        Get list of component names in execution order.
        
        Returns:
            List of component names
        """
        return [component.get_name() for component in self.components]
    
    def clear(self) -> None:
        """
        Clear all components and timing data from the pipeline.
        """
        self.components.clear()
        self.execution_times.clear()
        self.total_execution_time = 0.0
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        component_names = " → ".join(self.get_component_names())
        return f"Pipeline('{self.name}': {component_names})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Pipeline(name='{self.name}', components={len(self.components)})"
