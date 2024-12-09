import sys
import struct
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class GraphPartitioner:
    def __init__(self, file_path, num_partitions, show_details):
        self.file_path = file_path
        self.num_partitions = num_partitions
        self.show_details = show_details

    # A (Slightly modified) replication of the C code already existing
    # Reads the data and saves all edges in the graph
    def read_graph(self):
        edges = []
        with open(self.file_path, "rb") as file:
            data = file.read(8)     # Read first 8 bytes
            while data:
                src, dst = struct.unpack("ii", data) # (4 byte src, 4 byte dst)
                edges.append((src, dst))
                data = file.read(8) # Read the next 8 bytes
        return edges

    # Show graphs in subplots for each partition
    def show_graph(self, partitions, title="Graph Partitions"):
        height = 4
        width = len(partitions) * height
        fig, axes = plt.subplots(1, len(partitions), figsize=(width, height))

        if len(partitions) == 1:
            axes = [axes]  # Ensure axes is always iterable for a single partition

        for i, (partition_id, partition) in enumerate(sorted(partitions.items())):
            ax = axes[i]
            G = nx.DiGraph()  # Use DiGraph for directed edges
            G.add_edges_from(partition['edges'])

            # Differentiate master vertices by color, or color all nodes lightgreen if none exist
            master_vertices = partition.get('master_vertices', set())
            # If there is no master_vertices value, then we are on the Full graph and all are master vertices
            if not master_vertices:
                node_colors = ["lightgreen" for node in G.nodes()]
            # If only some are master vertices, color them lightgreen and the rest skyblue
            else:
                node_colors = ["lightgreen" if node in master_vertices else "skyblue" for node in G.nodes()]
            
            pos = nx.spring_layout(G)  # Use spring layout for better visualization
            nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight="bold",
                    node_size=500, edge_color="gray", ax=ax, arrows=True)
            ax.set_title(f"Partition {partition_id}")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Balanced p-way edge-cut partitioning
    def edge_cut_partition(self, edges):
        partitions = defaultdict(
            lambda: {'master_vertices': set(), 'total_vertices': set(), 'replicated_edges': 0, 'edges': []}
        )
        vertex_to_partition = {}
        
        for src, dst in edges:
            # Ensure the partition for each vertex is hashed consistently
            src_partition = vertex_to_partition.get(src, hash(src) % self.num_partitions)
            dst_partition = vertex_to_partition.get(dst, hash(dst) % self.num_partitions)
            
            # Assign the vertex to the partition
            vertex_to_partition[src] = src_partition
            vertex_to_partition[dst] = dst_partition
            
            # Assign edges and count replicated edges
            if src_partition != dst_partition:
                partitions[src_partition]['replicated_edges'] += 1
                partitions[dst_partition]['replicated_edges'] += 1

            partitions[src_partition]['edges'].append((src, dst))
            partitions[dst_partition]['edges'].append((src, dst))

            # Update master and total vertices
            partitions[src_partition]['master_vertices'].add(src)
            partitions[dst_partition]['master_vertices'].add(dst)
            partitions[src_partition]['total_vertices'].update([src, dst])
            partitions[dst_partition]['total_vertices'].update([src, dst])

        # Print partition statistics
        for i, partition in sorted(partitions.items()):
            print(f"Partition {i}")
            print(partition['master_vertices'])
            print(partition['total_vertices'])
            print(partition['replicated_edges'])
            print(partition['edges'])

        # Show the partitions side by side
        if self.show_details:
            self.show_graph(partitions, title="Edge-Cut Partitioning")

    def vertex_cut_partition(self, edges):
        partitions = defaultdict(
            lambda: {'master_vertices': set(), 'total_vertices': set(), 'edges': []}
        )
        vertex_partitions = defaultdict(set)  # Store all partitions a vertex is assigned to

        # Assign edges to partitions using hash
        for src, dst in edges:
            # Assign edge to a partition using hash
            edge_partition = hash((src, dst)) % self.num_partitions
            partitions[edge_partition]['edges'].append((src, dst))

            # Update vertex partitions
            vertex_partitions[src].add(edge_partition)
            vertex_partitions[dst].add(edge_partition)

        # Calculate master and total vertices for each partition
        for vertex, assigned_partitions in vertex_partitions.items():
            # Choose one partition as master
            master_partition = random.choice(list(assigned_partitions))  # We can still randomly choose master
            for partition_id in assigned_partitions:
                partitions[partition_id]['total_vertices'].add(vertex)
                if partition_id == master_partition:
                    partitions[partition_id]['master_vertices'].add(vertex)

        # Print partition statistics
        for i, partition in sorted(partitions.items()):
            print(f"Partition {i}")
            print(partition['master_vertices'])
            print(partition['total_vertices'])
            print(partition['edges'])

        # Show the partitions side by side
        if self.show_details:
            self.show_graph(partitions, title="Vertex-Cut Partitioning")

if __name__ == "__main__":
    file_path = "small-5.graph"
    num_partitions = 3
    show_details = True
    partitioner = GraphPartitioner(file_path, num_partitions, show_details)

    # Read the graph
    edges = partitioner.read_graph()

    # Display the graph if applicable
    if show_details:
        partitioner.show_graph({0: {'edges': edges}}, title="Full Graph")

    # Partition the graph
    print("Edge-Cut Partitioning:")
    partitioner.edge_cut_partition(edges)
    print("\nVertex-Cut Partitioning:")
    partitioner.vertex_cut_partition(edges)
