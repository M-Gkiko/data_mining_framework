"""
Pipeline Example: Dimensionality Reduction + Clustering + Quality

This example shows how to chain DR, clustering, and quality evaluation.
"""

from data_mining_framework import (
    CSVDataset,
    PCAProjection,
    HierarchicalClustering,
    CalinskiHarabaszIndex,
    ManhattanDistance,
    Pipeline,
    DRAdapter,
    ClusteringAdapter,
    ClusteringQualityAdapter
)


def pipeline_example():
    """Run a complete DR -> Clustering -> Quality pipeline."""
    print("=== Pipeline Example: PCA -> Hierarchical -> Quality ===")
    
    try:
        # Load your dataset (replace with your own data file)
        dataset = CSVDataset('your_data.csv')
        print(f"Loaded dataset: {dataset.get_rows()} rows, {len(dataset.get_features())} features")
        
        # Create distance measure
        distance_measure = ManhattanDistance()
        
        # Create pipeline
        pipeline = Pipeline("PCA_Hierarchical_Quality")
        
        # Step 1: Dimensionality Reduction
        pca = PCAProjection(n_components=2)
        dr_adapter = DRAdapter(pca)
        pipeline.add_component(dr_adapter)
        
        # Step 2: Clustering
        clustering = HierarchicalClustering(
            distance_measure=distance_measure,
            n_clusters=3,
            linkage='complete'
        )
        clustering_adapter = ClusteringAdapter(clustering, distance_measure)
        pipeline.add_component(clustering_adapter)
        
        # Step 3: Quality Evaluation
        quality_measure = CalinskiHarabaszIndex()
        quality_adapter = ClusteringQualityAdapter(quality_measure)
        pipeline.add_component(quality_adapter)
        
        # Execute the pipeline
        print("Executing pipeline...")
        results = pipeline.execute(dataset)
        
        # Display results
        print("Pipeline completed!")
        print(f"Execution times: {pipeline.get_execution_times()}")
        print(f"Quality results: {results}")
        
    except FileNotFoundError:
        print("Please replace 'your_data.csv' with the path to your actual dataset")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    pipeline_example()