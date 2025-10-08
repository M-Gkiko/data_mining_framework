"""
Core benchmark orchestration system for automated algorithm comparison.

This module provides the main benchmarking infrastructure for systematically
testing different algorithm combinations in pipelines.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from itertools import product
import time
from pathlib import Path
import traceback

from core.pipeline import Pipeline
from implementations.pipelines import DRAdapter, ClusteringAdapter, DRQualityAdapter, ClusteringQualityAdapter
from core.dataset import Dataset
from utils.timer import Timer


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    pipeline_name: str
    algorithm_combination: Dict[str, str]
    execution_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    iteration: int = 0
    
    def add_timing(self, component_name: str, time_seconds: float) -> None:
        """Add timing result for a component."""
        self.execution_times[component_name] = time_seconds
        self.total_time = sum(self.execution_times.values())
    
    def add_quality_score(self, metric_name: str, score: float) -> None:
        """Add quality score."""
        self.quality_scores[metric_name] = score
    
    def set_error(self, error_message: str) -> None:
        """Mark benchmark as failed."""
        self.success = False
        self.error_message = error_message


@dataclass
class BenchmarkConfiguration:
    """Configuration for a benchmark run."""
    
    name: str
    dataset_path: str
    pipeline_steps: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 1
    output_directory: str = "benchmark_results"
    output_formats: List[str] = field(default_factory=lambda: ["csv"])
    distance_measure: str = "Manhattan"  # Default distance measure
    
    def add_pipeline_step(self, step_type: str, algorithms: List[str], 
                         params: Dict[str, Dict[str, Any]] = None) -> None:
        """Add a step to the pipeline configuration."""
        self.pipeline_steps.append({
            'type': step_type,
            'algorithms': algorithms,
            'params': params or {}
        })
    
    def generate_combinations(self) -> List[Dict[str, str]]:
        """Generate all possible algorithm combinations."""
        if not self.pipeline_steps:
            return [{}]
        
        # Extract algorithm options for each step type
        step_types = []
        algorithm_options = []
        
        for step in self.pipeline_steps:
            step_types.append(step['type'])
            algorithm_options.append(step['algorithms'])
        
        # Generate all combinations using itertools.product
        combinations = []
        for combo in product(*algorithm_options):
            combination_dict = dict(zip(step_types, combo))
            combinations.append(combination_dict)
        
        return combinations


class AlgorithmRegistry:
    """Registry for available algorithms by type."""
    
    def __init__(self):
        self._algorithms: Dict[str, Dict[str, Any]] = {}
    
    def register_algorithm(self, algorithm_type: str, name: str, 
                          algorithm_class: Any, default_params: Dict[str, Any] = None) -> None:
        """Register an algorithm."""
        if algorithm_type not in self._algorithms:
            self._algorithms[algorithm_type] = {}
        
        self._algorithms[algorithm_type][name] = {
            'class': algorithm_class,
            'default_params': default_params or {}
        }
    
    def get_algorithm(self, algorithm_type: str, name: str, **override_params) -> Any:
        """Create an algorithm instance with parameters."""
        if algorithm_type not in self._algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        if name not in self._algorithms[algorithm_type]:
            raise ValueError(f"Unknown algorithm '{name}' for type '{algorithm_type}'")
        
        algo_info = self._algorithms[algorithm_type][name]
        params = algo_info['default_params'].copy()
        params.update(override_params)
        
        return algo_info['class'](**params)
    
    def get_available_algorithms(self, algorithm_type: str) -> List[str]:
        """Get list of available algorithms for a type."""
        return list(self._algorithms.get(algorithm_type, {}).keys())


class PipelineBenchmark:
    """
    Main benchmark orchestrator for systematic algorithm comparison.
    
    Coordinates the execution of multiple pipeline configurations,
    measures performance, and collects results.
    """
    
    def __init__(self, registry: AlgorithmRegistry):
        """
        Initialize benchmark with algorithm registry.
        
        Args:
            registry: Registry containing available algorithms
        """
        self.registry = registry
        self.results: List[BenchmarkResult] = []
        self.distance_measure = None  # Will be set by run_benchmark.py
    
    def run_benchmark(self, config: BenchmarkConfiguration, 
                     dataset: Dataset) -> List[BenchmarkResult]:
        """
        Run complete benchmark according to configuration.
        
        Args:
            config: Benchmark configuration
            dataset: Dataset to run benchmarks on
            
        Returns:
            List of benchmark results for all combinations and iterations
        """
        print(f"Starting benchmark: {config.name}")
        print(f"Dataset: {dataset.name}")
        
        self.results.clear()
        combinations = config.generate_combinations()
        
        print(f"Testing {len(combinations)} algorithm combinations")
        print(f"Running {config.iterations} iterations each")
        print(f"Total benchmark runs: {len(combinations) * config.iterations}")
        
        for i, combination in enumerate(combinations, 1):
            print(f"\n--- Combination {i}/{len(combinations)}: {combination} ---")
            
            for iteration in range(config.iterations):
                print(f"  Iteration {iteration + 1}/{config.iterations}")
                
                result = self._run_single_benchmark(
                    config, combination, dataset, iteration
                )
                self.results.append(result)
                
                if result.success:
                    print(f"    ✓ Success - Total time: {result.total_time:.3f}s")
                    if result.quality_scores:
                        quality_str = ", ".join([f"{k}: {v:.3f}" for k, v in result.quality_scores.items()])
                        print(f"    ✓ Quality scores: {quality_str}")
                else:
                    print(f"    ✗ Failed: {result.error_message}")
        
        print(f"\nBenchmark completed: {len([r for r in self.results if r.success])}/{len(self.results)} successful")
        return self.results
    
    def _run_single_benchmark(self, config: BenchmarkConfiguration,
                            combination: Dict[str, str], dataset: Dataset,
                            iteration: int) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        pipeline_name = f"{config.name}_{iteration}"
        result = BenchmarkResult(pipeline_name, combination, iteration=iteration)
        
        try:
            # Build pipeline from combination
            pipeline = self._build_pipeline(config, combination, dataset)
            
            # Execute pipeline with timing
            overall_timer = Timer("benchmark_execution")
            overall_timer.start()
            
            output = pipeline.execute(dataset)
            
            total_time = overall_timer.stop()
            
            # Record component timings
            for comp_name, comp_time in pipeline.get_execution_times().items():
                result.add_timing(comp_name, comp_time)
            
            # Extract quality scores if present
            if isinstance(output, dict) and any(isinstance(v, (int, float)) for v in output.values()):
                for metric_name, score in output.items():
                    if isinstance(score, (int, float)):
                        result.add_quality_score(metric_name, score)
            
        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}\n{traceback.format_exc()}"
            result.set_error(error_msg)
        
        return result
    
    def _build_pipeline(self, config: BenchmarkConfiguration, 
                       combination: Dict[str, str], dataset: Dataset) -> Pipeline:
        """Build a pipeline from algorithm combination."""
        pipeline_name = "_".join(combination.values())
        pipeline = Pipeline(pipeline_name)
        
        for step in config.pipeline_steps:
            step_type = step['type']
            algorithm_name = combination[step_type]
            step_params = step['params'].get(algorithm_name, {})
            
            # Create algorithm instance
            algorithm = self.registry.get_algorithm(step_type, algorithm_name, **step_params)
            
            # Wrap in appropriate adapter with distance measure
            if step_type == "dimensionality_reduction":
                adapter = DRAdapter(algorithm, distance_measure=self.distance_measure, name=f"DR_{algorithm.__class__.__name__}")
                
            elif step_type == "clustering":
                adapter = ClusteringAdapter(algorithm, distance_measure=self.distance_measure, name=f"Clustering_{algorithm.__class__.__name__}")
                
            elif step_type == "dr_quality":
                adapter = DRQualityAdapter(algorithm, dataset, distance_measure=self.distance_measure, name=f"DRQuality_{algorithm.__class__.__name__}")
                
            elif step_type == "clustering_quality":
                adapter = ClusteringQualityAdapter(algorithm, distance_measure=self.distance_measure, name=f"ClusteringQuality_{algorithm.__class__.__name__}")
                
            else:
                raise ValueError(f"Unknown pipeline step type: {step_type}")
            
            pipeline.add_component(adapter)
        
        return pipeline
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        
        summary = {
            'total_runs': len(self.results),
            'successful_runs': len(successful_results),
            'failed_runs': len(self.results) - len(successful_results),
            'success_rate': len(successful_results) / len(self.results) * 100
        }
        
        if successful_results:
            times = [r.total_time for r in successful_results]
            summary.update({
                'avg_execution_time': sum(times) / len(times),
                'min_execution_time': min(times),
                'max_execution_time': max(times),
                'total_benchmark_time': sum(times)
            })
            
            # Find best performing combination by speed
            fastest = min(successful_results, key=lambda x: x.total_time)
            summary['fastest_combination'] = {
                'algorithms': fastest.algorithm_combination,
                'time': fastest.total_time,
                'quality_scores': fastest.quality_scores
            }
            
            # Find best performing combination by quality (if available)
            quality_results = [r for r in successful_results if r.quality_scores]
            if quality_results:
                # Use first quality metric for comparison
                first_metric = list(quality_results[0].quality_scores.keys())[0]
                best_quality = max(quality_results, 
                                 key=lambda x: x.quality_scores.get(first_metric, 0))
                
                summary['best_quality_combination'] = {
                    'algorithms': best_quality.algorithm_combination,
                    'time': best_quality.total_time,
                    'quality_scores': best_quality.quality_scores,
                    'metric_used': first_metric
                }
        
        return summary