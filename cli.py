#!/usr/bin/env python3
"""
Command-line interface for the Data Mining Framework.

Provides easy access to benchmark functionality from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .benchmarks.utils import load_benchmark_config
from .benchmarks.core import run_benchmark_from_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Mining Framework - Benchmark data mining algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a benchmark from config file
  dm-benchmark configs/sample_configs/clustering_benchmark.yaml
  
  # Run with verbose output
  dm-benchmark configs/sample_configs/dr_cl_quality.yaml --verbose
  
  # List available sample configs
  dm-benchmark --list-configs
        """
    )
    
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to benchmark configuration YAML file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output during benchmark execution"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available sample configuration files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory for results"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Data Mining Framework 0.1.0"
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_sample_configs()
        return 0
    
    if not args.config:
        parser.error("Configuration file is required (or use --list-configs)")
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found")
        return 1
    
    try:
        print("="*60)
        print("DATA MINING FRAMEWORK BENCHMARK")
        print("="*60)
        print()
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        config = load_benchmark_config(str(config_path))
        print(f"✓ Configuration loaded: {config.benchmark.name}")
        print()
        
        # Override output directory if specified
        if args.output_dir:
            config.output.directory = args.output_dir
            print(f"✓ Output directory overridden: {args.output_dir}")
        
        # Run benchmark
        results = run_benchmark_from_config(config, verbose=args.verbose)
        
        print()
        print("✓ Benchmark completed successfully!")
        print(f"✓ Results exported to: {config.output.directory}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def list_sample_configs():
    """List available sample configuration files."""
    print("Available sample configurations:")
    print()
    
    # Try to find example configs relative to package
    try:
        import data_mining_framework
        package_dir = Path(data_mining_framework.__file__).parent
        examples_dir = package_dir / "examples"
    except:
        # Fallback to current directory
        examples_dir = Path("examples")
    
    if examples_dir.exists():
        yaml_files = list(examples_dir.glob("*.yaml")) + list(examples_dir.glob("*.yml"))
        
        if yaml_files:
            for config_file in sorted(yaml_files):
                rel_path = config_file.relative_to(examples_dir.parent)
                print(f"  {rel_path}")
                
                # Try to read the config description
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        if 'name:' in content:
                            # Extract name from YAML
                            for line in content.split('\n'):
                                if 'name:' in line and 'benchmark:' not in line:
                                    name = line.split('name:')[1].strip().strip('"\'')
                                    print(f"    Description: {name}")
                                    break
                except:
                    pass
                print()
        else:
            print("  No example configuration files found")
    else:
        print("  Examples directory not found")
        print(f"  Looked in: {examples_dir}")


if __name__ == "__main__":
    sys.exit(main())