import pandas as pd
import numpy as np
import json
from collections import defaultdict
from itertools import combinations
from networkx.algorithms import isomorphism
from math import comb
from multiprocessing import Pool

def build_graph(edges, vertices):
    print("Building graph...")
    graph = defaultdict(list)
    
    # Ensure unique indices in the vertices DataFrame
    vertices = vertices.drop_duplicates(subset='id').set_index('id')
    
    # Add all edges to the graph
    for _, row in edges.iterrows():
        source, target = row['source_id'], row['target_id']
        graph[source].append((target, row['amt'], row['strategy_name'], row['buscode']))
    
    print(f"Total nodes: {len(graph)}, Total edges: {sum(len(v) for v in graph.values())}\n")
    return graph

def hash_edge(source, target, amt, strategy_name, buscode):
    # I think we can go without some of these values, but thought they might be nice to have anyways :-)
    return f"{min(source, target)}-{max(source, target)}-{amt}-{strategy_name}-{buscode}"

def mine_frequent_subgraphs(graph, pattern_size, support_threshold, output_file):
    print("Mining frequent subgraphs...")
    subgraph_counts = defaultdict(int)
    subgraph_patterns = defaultdict(list)
    
    # Iterate over node combinations
    for nodes in combinations(graph.keys(), pattern_size):
        edges = []
        for u, v in combinations(nodes, 2):
            if v in [neighbor[0] for neighbor in graph[u]]:
                edge_details = next(
                    (neighbor for neighbor in graph[u] if neighbor[0] == v), None
                )
                if edge_details:
                    edges.append(
                        hash_edge(u, v, *edge_details[1:])
                    )
        
        if len(edges) == pattern_size:
            subgraph_key = "_".join(sorted(edges))
            subgraph_counts[subgraph_key] += 1
            subgraph_patterns[subgraph_key].append(edges)
    
    # Filter frequent subgraphs
    frequent_subgraphs = {
        k: v for k, v in subgraph_counts.items() if v >= support_threshold
    }
    
    save_results(frequent_subgraphs, subgraph_patterns, output_file)
    print("Frequent subgraph mining completed.")

def save_results(frequent_subgraphs, subgraph_patterns, output_file):
    result = []
    for subgraph, frequency in frequent_subgraphs.items():
        edges = subgraph_patterns[subgraph]
        result.append({
            "frequency": frequency,
            "edges": [{"source": edge.split("-")[0], "target": edge.split("-")[1], "details": edge} for edge in edges]
        })
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    print("Reading data...")
    header1 = ['id', 'name', 'timestamp', 'black']
    header2 = ['source_id', 'target_id', 'timestamp', 'amt', 'strategy_name', 'trade_no', 'buscode', 'other']

    account = pd.read_csv('data/account', names=header1, sep=',')
    card = pd.read_csv('data/card', names=header1, sep=',')
    account_to_account = pd.read_csv('data/account_to_account', names=header2, sep=',', usecols=range(len(header2)))
    account_to_card = pd.read_csv('data/account_to_card', names=header2, sep=',', usecols=range(len(header2)))

    vertices = pd.concat([account, card])
    edges = pd.concat([account_to_account, account_to_card])
    edges['amt'] = edges['amt'].round()

    graph = build_graph(edges, vertices)

    # Start mining frequent subgraphs
    mine_frequent_subgraphs(
        graph,
        pattern_size=3,
        support_threshold=10000,
        output_file=f"results/bdci_data.json"
    )