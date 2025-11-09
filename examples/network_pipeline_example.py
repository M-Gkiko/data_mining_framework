"""
Network Analysis Pipeline Example

This example demonstrates how to:
1. Load a network from an edge list file
2. Detect communities using Louvain algorithm
3. Calculate node centrality measures (PageRank, Degree)
4. Calculate edge measures (Betweenness)
"""

from data_mining_framework import (
    NetworkXWrapper,
    EdgeListNetwork,
    LouvainCommunityDetection,
    PageRankMeasure,
    DegreeCentralityMeasure,
    EdgeBetweennessMeasure,
    Pipeline,
    NetworkAdapter,
    CommunityDetectionAdapter,
    NodeMeasureAdapter,
    EdgeMeasureAdapter
)


def simple_network_example():
    """Simple example: Load network and run community detection."""
    print("=== Simple Network Example ===\n")

    # Create a small example network using NetworkX
    import networkx as nx

    # Create Karate Club network (a classic network analysis dataset)
    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)

    print(f"Network loaded: {network.node_count()} nodes, {network.edge_count()} edges\n")

    # Detect communities
    print("Running Louvain community detection...")
    louvain = LouvainCommunityDetection(resolution=1.0)
    louvain.fit(network)
    communities = louvain.get_communities()
    modularity = louvain.get_modularity()

    print(f"Communities detected: {len(set(communities.values()))}")
    print(f"Modularity: {modularity:.4f}\n")

    # Calculate PageRank
    print("Calculating PageRank...")
    pagerank = PageRankMeasure(alpha=0.85)
    pr_scores = pagerank.calculate(network)

    # Get top 5 nodes by PageRank
    top_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 nodes by PageRank:")
    for node, score in top_nodes:
        print(f"  Node {node}: {score:.4f}")
    print()


def pipeline_network_example():
    """Complete pipeline: Network -> Community Detection -> Node/Edge Measures."""
    print("\n=== Network Analysis Pipeline ===\n")

    # Create a small example network
    import networkx as nx
    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)

    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges\n")

    # Create pipeline
    pipeline = Pipeline("Network_Analysis_Pipeline")

    # Step 1: Load network
    network_adapter = NetworkAdapter(network, name="KarateClub")
    pipeline.add_component(network_adapter)

    # Step 2: Community detection
    louvain = LouvainCommunityDetection(resolution=1.0)
    community_adapter = CommunityDetectionAdapter(louvain, name="Louvain")
    pipeline.add_component(community_adapter)

    # Step 3: Node measure (PageRank)
    pagerank = PageRankMeasure(alpha=0.85)
    pagerank_adapter = NodeMeasureAdapter(pagerank, name="PageRank")
    pipeline.add_component(pagerank_adapter)

    # Step 4: Node measure (Degree Centrality)
    degree = DegreeCentralityMeasure(normalized=True)
    degree_adapter = NodeMeasureAdapter(degree, name="DegreeCentrality")
    pipeline.add_component(degree_adapter)

    # Step 5: Edge measure (Edge Betweenness)
    edge_betweenness = EdgeBetweennessMeasure(normalized=True)
    edge_adapter = EdgeMeasureAdapter(edge_betweenness, name="EdgeBetweenness")
    pipeline.add_component(edge_adapter)

    # Execute pipeline
    print("Executing pipeline...")
    results = pipeline.execute()

    # Display results
    print("\nPipeline completed!")
    print(f"Execution times: {pipeline.get_execution_times()}")

    # Show degree centrality results
    if isinstance(results, dict) and 'node_scores' in results:
        top_nodes = sorted(results['node_scores'].items(),
                          key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 nodes by Degree Centrality:")
        for node, score in top_nodes:
            print(f"  Node {node}: {score:.4f}")


def load_from_file_example():
    """Example: Load network from edge list file."""
    print("\n=== Load Network from File ===\n")

    # This example assumes you have an edge list file
    # Format: Each line should be "node1 node2" or "node1,node2"

    try:
        network = EdgeListNetwork(
            filepath='path/to/your/network.edgelist',
            directed=False
        )

        print(f"Network loaded: {network.node_count()} nodes, {network.edge_count()} edges")

        # Run analysis
        louvain = LouvainCommunityDetection()
        louvain.fit(network)
        print(f"Communities: {len(set(louvain.get_communities().values()))}")
        print(f"Modularity: {louvain.get_modularity():.4f}")

    except FileNotFoundError:
        print("File not found. Please provide a valid edge list file path.")
        print("Example file format:")
        print("  0 1")
        print("  1 2")
        print("  2 3")


def compare_algorithms_example():
    """Example: Compare different community detection algorithms."""
    print("\n=== Compare Community Detection Algorithms ===\n")

    # Create network
    import networkx as nx
    from data_mining_framework import GirvanNewmanCommunity, LabelPropagationCommunity

    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)

    algorithms = {
        "Louvain": LouvainCommunityDetection(resolution=1.0),
        "Girvan-Newman": GirvanNewmanCommunity(k=2),
        "Label Propagation": LabelPropagationCommunity(max_iterations=100)
    }

    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges\n")

    for name, algo in algorithms.items():
        algo.fit(network)
        communities = algo.get_communities()
        modularity = algo.get_modularity()

        # Count communities
        if isinstance(communities, dict):
            num_communities = len(set(communities.values()))
        else:
            num_communities = len(communities)

        print(f"{name}:")
        print(f"  Communities: {num_communities}")
        print(f"  Modularity: {modularity:.4f}\n")


if __name__ == "__main__":
    # Run all examples
    simple_network_example()
    pipeline_network_example()
    load_from_file_example()
    compare_algorithms_example()
