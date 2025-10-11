"""
Basic Clustering Example

This example demonstrates how to use the framework for simple clustering tasks.
"""

from data_mining_framework import (
    CSVDataset, 
    HierarchicalClustering, 
    ManhattanDistance,
    CalinskiHarabaszIndex
)


def basic_clustering_example():
    """Run a basic clustering example."""
    print("=== Basic Clustering Example ===")
    
    # Note: You'll need to provide your own dataset
    # This is just an example of how to use the framework
    
    try:
        # Load your dataset (replace with your own data file)
        dataset = CSVDataset('your_data.csv')
        print(f"Loaded dataset: {dataset.get_rows()} rows, {len(dataset.get_features())} features")
        
        # Create distance measure
        distance_measure = ManhattanDistance()
        
        # Create clustering algorithm
        clustering = HierarchicalClustering(
            distance_measure=distance_measure,
            n_clusters=3,
            linkage='complete'
        )
        
        # Fit the clustering algorithm
        print("Fitting clustering algorithm...")
        clustering.fit(dataset)
        
        # Get cluster labels
        labels = clustering.get_labels()
        print(f"Clustering completed! Found {len(set(labels))} clusters")
        
        # Evaluate clustering quality
        quality_measure = CalinskiHarabaszIndex()
        score = quality_measure.evaluate(dataset, labels)
        print(f"Calinski-Harabasz Score: {score:.3f}")
        
    except FileNotFoundError:
        print("Please replace 'your_data.csv' with the path to your actual dataset")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    basic_clustering_example()