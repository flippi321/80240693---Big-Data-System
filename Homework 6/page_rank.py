import sys
from pyspark import SparkConf, SparkContext
import time

if __name__ == '__main__':
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR") # Added to avoid Warnings cluttering the terminal
    file_path = sys.argv[1]
    lines = sc.textFile(file_path)

    # Parameters
    damping = 0.8
    num_iterations = 50
    top_i_nodes = 5
    dynamic = sys.argv[2] 

    first = time.time()

    # Parse the lines into (source, destination) pairs and remove duplicates
    edges = lines.map(lambda line: tuple(map(int, line.split()))).distinct()

    # We estimate the amount of nodes
    if(dynamic):
        # This method takes aprox. 0.5s longer but is dynamic
        nodes = edges.flatMap(lambda edge: edge).distinct()
        n = nodes.count()
    else: 
        # This method is faster but not dynamic
        n = 100 if "small" in file_path else 1000 if "full" in file_path else None
    
    # Create an adjacency list as (node, [neighbors])
    adj_list = edges.groupByKey().mapValues(list).cache()
    
    # Initialize each node's PageRank value
    page_ranks = adj_list.mapValues(lambda _: 1.0 / n)
    
    for i in range(num_iterations):
        # Broadcast the adjacency list for efficient access
        adjacency_broadcast = sc.broadcast(adj_list.collectAsMap())
        
        # Compute contributions for each node's neighbors
        contributions = page_ranks.flatMap(lambda node_rank: [
            (neighbor, node_rank[1] / len(adjacency_broadcast.value.get(node_rank[0], [])))
            for neighbor in adjacency_broadcast.value.get(node_rank[0], [])
        ])
        
        # Aggregate contributions and calculate new PageRank values
        page_ranks = contributions.reduceByKey(lambda a, b: a + b).mapValues(
            lambda rank: (1 - damping) / n + damping * rank
        )
    
    # Get the top 5 nodes with the highest PageRank scores
    highest = page_ranks.takeOrdered(top_i_nodes, key=lambda x: -x[1])
    
    # Print the top 5 nodes
    print("Top 5 nodes with highest PageRank scores:")
    for node, score in highest:
        print(f"Node {node}: {score}")

    last = time.time()
    print("Total program time: %.2f seconds" % (last - first))
    sc.stop()