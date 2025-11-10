"""
Basic Component Usage Example: Direct Usage Without Pipelines or Benchmarks

This example demonstrates using framework components directly without pipelines:
1. Clustering: DR + Clustering + Quality evaluation
2. Network Analysis: Community Detection + Node/Edge measures

This is useful when you need fine-grained control over each step.
"""

import networkx as nx
from data_mining_framework import (
    # Datasets
    CSVDataset,
    NumpyDataset,

    # DR algorithms
    PCAProjection,

    # Clustering algorithms
    HierarchicalClustering,

    # Quality measures
    CalinskiHarabaszIndex,

    # Distance measures
    ManhattanDistance,

    # Network
    NetworkXWrapper,

    # Community detection
    LouvainCommunityDetection,

    # Node measures
    PageRankMeasure,
    DegreeCentralityMeasure,

    # Edge measures
    EdgeBetweennessMeasure
)


def clustering_example():
    """Example 1: Clustering workflow using direct component calls"""
    print("Example 1: Clustering with Direct Component Usage\n")

    # Load dataset
    dataset = CSVDataset('data/iris.csv')
    print(f"Dataset: {dataset.get_rows()} samples, {len(dataset.get_features())} features\n")

    # Step 1: Apply dimensionality reduction
    print("Step 1: Dimensionality Reduction (PCA)")
    pca = PCAProjection(n_components=2)
    reduced_array = pca.fit_transform(dataset)
    reduced_data = NumpyDataset(reduced_array)  # Wrap back into Dataset
    print(f"  Reduced to 2 dimensions\n")

    # Step 2: Apply clustering
    print("Step 2: Clustering (Hierarchical)")
    distance_measure = ManhattanDistance()
    clustering = HierarchicalClustering(
        distance_measure=distance_measure,
        n_clusters=3,
        linkage='complete'
    )
    clustering.fit(reduced_data)
    labels = clustering.get_labels()

    # Count clusters
    unique_clusters = len(set(labels))
    print(f"  Clusters found: {unique_clusters}\n")

    # Step 3: Evaluate quality
    print("Step 3: Quality Evaluation (Calinski-Harabasz)")
    quality_measure = CalinskiHarabaszIndex()
    score = quality_measure.evaluate(reduced_data, labels)
    print(f"  Calinski-Harabasz Score: {score:.4f} (higher is better)\n")


def network_example():
    """Example 2: Network analysis using direct component calls"""
    print("Example 2: Network Analysis with Direct Component Usage\n")

    # Load network
    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)
    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges\n")

    # Step 1: Community detection
    print("Step 1: Community Detection (Louvain)")
    louvain = LouvainCommunityDetection(resolution=1.0)
    louvain.fit(network)
    communities = louvain.get_communities()
    modularity = louvain.get_modularity()

    num_communities = len(set(communities.values()))
    print(f"  Communities detected: {num_communities}")
    print(f"  Modularity: {modularity:.4f}\n")

    # Step 2: Calculate edge measures
    print("Step 2: Edge Measures (Betweenness)")
    edge_betweenness = EdgeBetweennessMeasure()
    edge_scores = edge_betweenness.calculate(network)

    top_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3 edges by betweenness:")
    for edge, score in top_edges:
        print(f"    Edge {edge}: {score:.4f}")
    print()

    # Step 3: Calculate node measures
    print("Step 3: Node Measures")

    # PageRank
    pagerank = PageRankMeasure(alpha=0.85)
    pr_scores = pagerank.calculate(network)
    top_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3 nodes by PageRank:")
    for node, score in top_pr:
        print(f"    Node {node}: {score:.4f}")

    # Degree Centrality
    degree_centrality = DegreeCentralityMeasure(normalized=True)
    deg_scores = degree_centrality.calculate(network)
    top_deg = sorted(deg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3 nodes by Degree Centrality:")
    for node, score in top_deg:
        print(f"    Node {node}: {score:.4f}")
    print()


def main():
    print("=" * 80)
    print("Basic Component Usage Examples")
    print("=" * 80)
    print()

    clustering_example()

    print("-" * 80)
    print()

    network_example()

    print("=" * 80)


if __name__ == "__main__":
    main()
