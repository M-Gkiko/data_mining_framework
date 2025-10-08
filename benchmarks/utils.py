"""
Utilities for benchmark I/O operations.
"""

import yaml
import json
import csv
from typing import Dict, Any, List
from pathlib import Path

from .core import BenchmarkConfig, BenchmarkResult


def load_benchmark_config(config_path: str) -> BenchmarkConfig:
    """
    Load and parse benchmark configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        BenchmarkConfig object with automatic validation
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If configuration is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # Extract benchmark section and merge with top-level settings
    benchmark_data = yaml_data.get('benchmark', {})
    
    # Add other top-level settings to benchmark data
    for key in ['iterations', 'distance_measure', 'output']:
        if key in yaml_data:
            if key == 'output':
                # Handle output section
                output_section = yaml_data['output']
                benchmark_data.setdefault('output_directory', output_section.get('directory', 'benchmark_results'))
                benchmark_data.setdefault('output_formats', output_section.get('format', ['csv']))
            else:
                benchmark_data[key] = yaml_data[key]
    
    # Add pipeline template
    if 'pipeline_template' in yaml_data:
        benchmark_data['pipeline_steps'] = yaml_data['pipeline_template']
    
    return BenchmarkConfig(**benchmark_data)


def export_results_csv(results: List[BenchmarkResult], output_path: str) -> None:
    """Export benchmark results to CSV format."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if not results:
            f.write("No results to export\n")
            return
        
        # Determine all possible columns
        all_quality_keys = set()
        all_combination_keys = set()
        
        for result in results:
            if result.quality_scores:
                all_quality_keys.update(result.quality_scores.keys())
            all_combination_keys.update(result.combination.keys())
        
        # Create headers
        headers = ['iteration', 'success', 'execution_time_seconds', 'error_message']
        headers.extend(sorted(all_combination_keys))
        headers.extend(sorted(all_quality_keys))
        
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write data rows
        for result in results:
            row = [
                result.iteration,
                result.success,
                f"{result.execution_time:.6f}" if result.success else "0.000000",
                result.error_message or ""
            ]
            
            # Add combination values
            for key in sorted(all_combination_keys):
                row.append(result.combination.get(key, ""))
            
            # Add quality scores
            for key in sorted(all_quality_keys):
                score = result.quality_scores.get(key, "") if result.quality_scores else ""
                row.append(f"{score:.6f}" if isinstance(score, (int, float)) else score)
            
            writer.writerow(row)
    
    print(f"Results exported to CSV: {output_path}")


def export_benchmark_results(results: List[BenchmarkResult], config: BenchmarkConfig, 
                            timestamp_suffix: str = "") -> None:
    """
    Export benchmark results in all configured formats.
    
    Args:
        results: List of benchmark results to export
        config: Benchmark configuration with output settings
        timestamp_suffix: Optional timestamp suffix for filenames
    """
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"{config.name.replace(' ', '_')}{timestamp_suffix}"
    
    export_functions = {
        'csv': export_results_csv,
    }
    
    for format_name in config.output_formats:
        if format_name in export_functions:
            output_path = output_dir / f"{base_filename}.{format_name}"
            export_functions[format_name](results, str(output_path))
        else:
            print(f"Warning: Unknown export format '{format_name}'. Skipping.")


def validate_benchmark_config(config: BenchmarkConfig) -> List[str]:
    """
    Validate benchmark configuration and return list of issues.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    # Check dataset path (already validated by Pydantic, but double-check)
    if not Path(config.dataset).exists():
        issues.append(f"Dataset file not found: {config.dataset}")
    
    # Check pipeline steps
    if not config.pipeline_steps:
        issues.append("No pipeline steps defined")
    
    # Check iterations
    if config.iterations < 1:
        issues.append("Iterations must be >= 1")
    
    # Check output formats
    valid_formats = {'csv', 'json', 'yaml'}
    invalid_formats = set(config.output_formats) - valid_formats
    if invalid_formats:
        issues.append(f"Unsupported output formats: {invalid_formats}")
    
    return issues


def print_benchmark_summary(results: List[BenchmarkResult], config: BenchmarkConfig) -> None:
    """Print a formatted summary of benchmark results."""
    if not results:
        print("No benchmark results to summarize.")
        return
    
    successful_results = [r for r in results if r.success]
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {len(successful_results)}")
    print(f"Failed runs: {len(results) - len(successful_results)}")
    print(f"Success rate: {(len(successful_results) / len(results) * 100):.1f}%")
    
    if successful_results:
        times = [r.execution_time for r in successful_results]
        print(f"\nTiming Results:")
        print(f"Average execution time: {sum(times) / len(times):.3f}s")
        print(f"Fastest execution time: {min(times):.3f}s")
        print(f"Slowest execution time: {max(times):.3f}s")
        print(f"Total benchmark time: {sum(times):.3f}s")
        
        # Find fastest combination
        fastest = min(successful_results, key=lambda x: x.execution_time)
        combo_str = " + ".join([f"{k}:{v}" for k, v in fastest.combination.items()])
        print(f"\nFastest Algorithm Combination:")
        print(f"  {combo_str}")
        print(f"  Time: {fastest.execution_time:.3f}s")
        if fastest.quality_scores:
            for metric, score in fastest.quality_scores.items():
                print(f"  Quality: {metric}: {score:.3f}")
        
        # Find best quality combination (Calinski-Harabasz higher is better)
        best_quality = None
        best_metric = None
        for result in successful_results:
            if result.quality_scores:
                # Look for Calinski-Harabasz first (higher is better)
                calinski_scores = {k: v for k, v in result.quality_scores.items() if 'Calinski' in k}
                if calinski_scores:
                    max_score = max(calinski_scores.values())
                    if best_quality is None or max_score > best_quality:
                        best_quality = max_score
                        best_metric = max(calinski_scores.keys(), key=lambda k: calinski_scores[k])
                        best_result = result
        
        if best_quality and 'best_result' in locals():
            combo_str = " + ".join([f"{k}:{v}" for k, v in best_result.combination.items()])
            print(f"\nBest Quality Algorithm Combination:")
            print(f"  {combo_str}")
            print(f"  Time: {best_result.execution_time:.3f}s")
            print(f"  Quality: {best_metric}: {best_quality:.3f}")
            print(f"  Metric used: {best_metric}")