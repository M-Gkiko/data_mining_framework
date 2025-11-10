"""
Executable examples demonstrating framework pipelines with real data.
"""

from data_mining_framework import (
    CSVDataset,
    PCAProjection, TSNEProjection,
    HierarchicalClustering, DBSCANClustering,
    CalinskiHarabaszIndex, DaviesBouldinIndex,
    ManhattanDistance, EuclideanDistance,
    Pipeline,
    DRAdapter, ClusteringAdapter, ClusteringQualityAdapter,
    NetworkXWrapper, LouvainCommunityDetection, GirvanNewmanCommunity,
    PageRankMeasure, DegreeCentralityMeasure
)


def example_pca_clustering():
    print("Example 1: PCA -> Hierarchical Clustering -> Quality\n")

    dataset = CSVDataset('../data/iris.csv')
    print(f"Dataset: {dataset.get_rows()} samples, {len(dataset.get_features())} features")

    distance_measure = ManhattanDistance()
    pipeline = Pipeline("PCA_Hierarchical_Quality")

    pipeline.add_component(DRAdapter(PCAProjection(n_components=2)))
    pipeline.add_component(ClusteringAdapter(
        HierarchicalClustering(distance_measure=distance_measure, n_clusters=3, linkage='complete'),
        distance_measure
    ))
    pipeline.add_component(ClusteringQualityAdapter(CalinskiHarabaszIndex()))

    result = pipeline.execute(dataset)
    times = pipeline.get_execution_times()

    print(f"\nResults:")
    for measure, score in result.items():
        print(f"  {measure}: {score:.4f}")
    print(f"\nExecution Times:")
    for component, time in times.items():
        print(f"  {component}: {time:.4f}s")
    print(f"  Total: {pipeline.get_total_time():.4f}s\n")


def example_tsne_dbscan():
    print("Example 2: t-SNE -> DBSCAN -> Quality\n")

    dataset = CSVDataset('../data/iris.csv')

    pipeline = Pipeline("TSNE_DBSCAN_Quality")
    pipeline.add_component(DRAdapter(TSNEProjection(n_components=2, perplexity=30)))
    pipeline.add_component(ClusteringAdapter(
        DBSCANClustering(eps=0.6, min_samples=4, distance_measure=EuclideanDistance()),
        EuclideanDistance()
    ))
    pipeline.add_component(ClusteringQualityAdapter(DaviesBouldinIndex()))

    result = pipeline.execute(dataset)
    times = pipeline.get_execution_times()

    print(f"Results:")
    for measure, score in result.items():
        print(f"  {measure}: {score:.4f}")
    print(f"\nExecution Times:")
    for component, time in times.items():
        print(f"  {component}: {time:.4f}s")
    print(f"  Total: {pipeline.get_total_time():.4f}s\n")


def example_network_analysis():
    print("Example 3: Network Community Detection + Node Measures\n")

    network = NetworkXWrapper(filepath='../data/karate.edgelist', format='edgelist')
    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges")

    louvain = LouvainCommunityDetection(resolution=1.0)
    louvain.fit(network)
    communities = louvain.get_communities()
    modularity = louvain.get_modularity()

    print(f"\nCommunity Detection (Louvain):")
    print(f"  Communities: {len(set(communities.values()))}")
    print(f"  Modularity: {modularity:.4f}")

    pagerank = PageRankMeasure(alpha=0.85)
    pr_scores = pagerank.calculate(network)

    degree_centrality = DegreeCentralityMeasure(normalized=True)
    deg_scores = degree_centrality.calculate(network)

    print(f"\nTop 5 nodes by PageRank:")
    top_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_pr:
        print(f"  Node {node}: {score:.4f}")

    print(f"\nTop 5 nodes by Degree Centrality:")
    top_deg = sorted(deg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_deg:
        print(f"  Node {node}: {score:.4f}")
    print()


def compare_community_detection():
    print("Example 4: Comparing Community Detection Algorithms\n")

    network = NetworkXWrapper(filepath='../data/karate.edgelist', format='edgelist')

    algorithms = {
        "Louvain": LouvainCommunityDetection(resolution=1.0),
        "Girvan-Newman": GirvanNewmanCommunity(k=2)
    }

    for name, algo in algorithms.items():
        algo.fit(network)
        communities = algo.get_communities()
        modularity = algo.get_modularity()

        if isinstance(communities, dict):
            num_communities = len(set(communities.values()))
        else:
            num_communities = len(communities)

        print(f"{name}:")
        print(f"  Communities: {num_communities}")
        print(f"  Modularity: {modularity:.4f}\n")


if __name__ == "__main__":
    example_pca_clustering()
    print("-" * 70 + "\n")
    example_tsne_dbscan()
    print("-" * 70 + "\n")
    example_network_analysis()
    print("-" * 70 + "\n")
    compare_community_detection()
