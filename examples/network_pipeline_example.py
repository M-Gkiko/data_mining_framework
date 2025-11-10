"""
Network Analysis Pipeline Example: Community Detection + Edge Measures + Node Measures

This example demonstrates a comprehensive network analysis pipeline using the framework:
1. Load a network (Karate Club)
2. Detect communities (Louvain)
3. Calculate edge measures (Betweenness)
4. Calculate node measures (PageRank)

The pipeline chains these operations together and all results are accumulated.
"""

import networkx as nx
from data_mining_framework import (
    NetworkXWrapper,
    LouvainCommunityDetection,
    PageRankMeasure,
    EdgeBetweennessMeasure,
    Pipeline,
    NetworkAdapter,
    CommunityDetectionAdapter,
    NodeMeasureAdapter,
    EdgeMeasureAdapter
)


def main():
    print("Network Analysis Pipeline Example: Community + Edge + Node Analysis\n")

    # Load network
    karate_graph = nx.karate_club_graph()
    network = NetworkXWrapper(graph=karate_graph)
    print(f"Network: {network.node_count()} nodes, {network.edge_count()} edges\n")

    # Create pipeline with four stages
    pipeline = Pipeline("Comprehensive_Network_Analysis")

    # Stage 1: Load Network
    pipeline.add_component(NetworkAdapter(network))

    # Stage 2: Community Detection
    pipeline.add_component(CommunityDetectionAdapter(LouvainCommunityDetection(resolution=1.0)))

    # Stage 3: Edge Measures
    pipeline.add_component(EdgeMeasureAdapter(EdgeBetweennessMeasure()))

    # Stage 4: Node Measures
    pipeline.add_component(NodeMeasureAdapter(PageRankMeasure(alpha=0.85)))

    # Execute pipeline
    results = pipeline.execute(None)

    # Display results - all accumulated from each stage
    print("Pipeline Results:\n")

    print(f"Community Detection:")
    print(f"  Communities detected: {len(set(results['communities'].values()))}")
    print(f"  Modularity: {results['modularity']:.4f}\n")

    print("Edge Measures (Top 3 by Betweenness):")
    top_edges = sorted(results['edge_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
    for edge, score in top_edges:
        print(f"  Edge {edge}: {score:.4f}")

    print("\nNode Measures (Top 5 by PageRank):")
    top_nodes = sorted(results['node_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_nodes:
        print(f"  Node {node}: {score:.4f}")

    print(f"\nExecution Times:")
    for component, time in pipeline.get_execution_times().items():
        print(f"  {component}: {time:.4f}s")
    print(f"  Total: {pipeline.get_total_time():.4f}s")


if __name__ == "__main__":
    main()
