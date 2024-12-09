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
        self.edges = self.read_graph()
        self.directed = self.isDirected()

    def isDirected(self):
        # Some of the graphs are undirected
        return not any(file_name in file_path for file_name in ["small-5", "synthesized-1b"])

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
        print("Done reading edges\n")
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
            G = nx.DiGraph() if self.directed else nx.Graph()
            G.add_edges_from(partition['edges'])

            # Differentiate master vertices by color, or color all nodes lightgreen if none exist
            master_vertices = partition.get('master_vertices', set())
            # If there is no master_vertices value, then we are on the Full graph and all are master vertices
            if not master_vertices:
                node_colors = ["lightgreen" for _ in G.nodes()]
            # If only some are master vertices, color them lightgreen and the rest skyblue
            else:
                node_colors = ["lightgreen" if node in master_vertices else "skyblue" for node in G.nodes()]
            
            pos = nx.spring_layout(G)  # Use spring layout for better visualization
            nx.draw(G, pos, with_labels=True, node_color=node_colors, font_weight="bold",
                    node_size=500, edge_color="gray", ax=ax, arrows=self.directed)
            ax.set_title(f"Partition {partition_id}")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Balanced p-way edge-cut partitioning
    def edge_cut_partition(self):
        partitions = defaultdict(
            lambda: {
                'master_vertices': set(), 
                'total_vertices': set(), 
                'replicated_edges': 0, 
                'edges': []
            }
        )
        vertex_to_partition = {}

        for src, dst in self.edges:
            # Hash vertices to assign them to partitions
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
            print(len(partition['master_vertices']))
            print(len(partition['total_vertices']))
            print(partition['replicated_edges'])
            print(len(partition['edges']))

        # Show the partitions side by side
        if self.show_details:
            self.show_graph(partitions, title="Edge-Cut Partitioning")

    # Balanced p-way vertex-cut partitioning
    def vertex_cut_partition(self):
        partitions = defaultdict(
            lambda: {
                'master_vertices': set(), 
                'total_vertices': set(), 
                'edges': []
            }
        )
        vertex_partitions = defaultdict(set)  # Store all partitions a vertex is assigned to

        for src, dst in self.edges:
            # Hash the edge to assign it to a partition
            edge_partition = hash((src, dst)) % self.num_partitions
            partitions[edge_partition]['edges'].append((src, dst))

            # Update vertex partitions
            vertex_partitions[src].add(edge_partition)
            vertex_partitions[dst].add(edge_partition)

        # Calculate master and total vertices for each partition
        for vertex, assigned_partitions in vertex_partitions.items():
            # Choose one partition as master
            master_partition = random.choice(list(assigned_partitions))
            for partition_id in assigned_partitions:
                partitions[partition_id]['total_vertices'].add(vertex)
                if partition_id == master_partition:
                    partitions[partition_id]['master_vertices'].add(vertex)

        # Print partition statistics
        for i, partition in sorted(partitions.items()):
            print(f"Partition {i}")
            print(f"{len(partition['master_vertices'])}")
            print(f"{len(partition['total_vertices'])}")
            print(f"{len(partition['edges'])}")

        # Show the partitions side by side
        if self.show_details:
            self.show_graph(partitions, title="Vertex-Cut Partitioning")

    # Greedy heuristic vertex-cut partitioning
    def greedy_heuristic_partition(self):
        partitions = defaultdict(
            lambda: {
                'master_vertices': set(),
                'total_vertices': set(),
                'edges': []
            }
        )
        vertex_degrees = defaultdict(int)

        # Calculate degrees for all vertices
        for src, dst in self.edges:
            vertex_degrees[src] += 1
            vertex_degrees[dst] += 1

        # Assign vertices to partitions using greedy heuristic
        for src, dst in self.edges:
            src_partition = vertex_degrees[src] % self.num_partitions
            dst_partition = vertex_degrees[dst] % self.num_partitions

            partitions[src_partition]['edges'].append((src, dst))
            partitions[dst_partition]['edges'].append((src, dst))

            partitions[src_partition]['total_vertices'].add(src)
            partitions[dst_partition]['total_vertices'].add(dst)

            master_partition = random.choice([src_partition, dst_partition])
            if master_partition == src_partition:
                partitions[src_partition]['master_vertices'].add(src)
            else:
                partitions[dst_partition]['master_vertices'].add(dst)

        # Ensure every partition gets at least one vertex
        for i in range(self.num_partitions):
            if not partitions[i]['total_vertices']:
                random_vertex = random.choice(list(vertex_degrees.keys()))
                partitions[i]['total_vertices'].add(random_vertex)
                partitions[i]['master_vertices'].add(random_vertex)

        # Print partition statistics
        for i, partition in sorted(partitions.items()):
            print(f"Partition {i}")
            print(len(partition['master_vertices']))
            print(len(partition['total_vertices']))
            print(len(partition['edges']))

        # Show the partitions side by side
        if self.show_details:
            self.show_graph(partitions, title="Greedy Vertex-Cut Partitioning")

    # Helper function to get the neighbors of a vertex
    def get_neighbors(self, vertex):
        neighbors = set()
        for src, dst in self.edges:
            if src == vertex:
                neighbors.add(dst)
            elif dst == vertex:
                neighbors.add(src)
        return neighbors

    # Helper function to get partition for a vertex
    def get_partition_for_vertex(self, vertex, partitions):
        for partition_id, partition in partitions.items():
            if vertex in partition['total_vertices']:
                return partition_id
        return None

    # Balanced p-way Hybrid-Cut based on PowerLyra
    def hybrid_cut_partition(self, theta=10):
        partitions = defaultdict(
            lambda: {
                'master_vertices': set(),
                'total_vertices': set(),
                'edges': []
            }
        )
        vertex_degree = defaultdict(int)  # Store vertex degrees
        vertex_partitions = defaultdict(set)  # Tracks the partitions each vertex is assigned to

        # Calculate degrees for all vertices
        for src, dst in self.edges:
            vertex_degree[src] += 1
            vertex_degree[dst] += 1

        for src, dst in self.edges:
            if vertex_degree[src] > theta or vertex_degree[dst] > theta:
                # High-degree vertices: use edge-cut strategy
                src_partition = hash(src) % self.num_partitions
                dst_partition = hash(dst) % self.num_partitions
                partitions[src_partition]['edges'].append((src, dst))
                partitions[dst_partition]['edges'].append((src, dst))
                partitions[src_partition]['master_vertices'].add(src)
                partitions[dst_partition]['master_vertices'].add(dst)
                partitions[src_partition]['total_vertices'].update([src, dst])
                partitions[dst_partition]['total_vertices'].update([src, dst])
            else:
                # Low-degree vertices: use vertex-cut (greedy heuristic)
                min_partition = None
                min_replication = float('inf')

                for partition_id in range(self.num_partitions):
                    replication_cost = (
                        int(src not in partitions[partition_id]['master_vertices']) +
                        int(dst not in partitions[partition_id]['master_vertices'])
                    )
                    if replication_cost < min_replication:
                        min_replication = replication_cost
                        min_partition = partition_id

                partitions[min_partition]['edges'].append((src, dst))
                partitions[min_partition]['master_vertices'].update([src, dst])
                partitions[min_partition]['total_vertices'].update([src, dst])
                vertex_partitions[src].add(min_partition)
                vertex_partitions[dst].add(min_partition)

        # Print partition statistics
        for i, partition in sorted(partitions.items()):
            print(f"Partition {i}")
            print(f"{len(partition['master_vertices'])}")
            print(f"{len(partition['total_vertices'])}")
            print(f"{len(partition['edges'])}")

        if self.show_details:
            self.show_graph(partitions, title=f"Hybrid-Cut Partitioning (Theta={theta})")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "small-5.graph"
    num_partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    show_details = (sys.argv[3]!="False") if len(sys.argv) > 3 else False
    partitioner = GraphPartitioner(file_path, num_partitions, show_details)

    # Display the graph if applicable
    if show_details:
        partitioner.show_graph({0: {'edges': partitioner.edges}}, title="Full Graph")
        
    # Partition the graph
    print("\nEdge-Cut Partitioning:")
    partitioner.edge_cut_partition()
    print("\nVertex-Cut Partitioning:")
    partitioner.vertex_cut_partition()
    print("\nHybrid-Cut Partitioning:")
    partitioner.hybrid_cut_partition(theta=2)
    print("\nGreedy-Cut Partitioning:")
    partitioner.greedy_heuristic_partition()
            