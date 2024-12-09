import struct

# Define the edges as (source, destination) tuples
edges = [
    (1, 4), (1, 6), (2, 1), (2, 5), (2, 6),
    (3, 1), (3, 4), (4, 1), (5, 1)
]

# Write the edges to a binary file
with open("example_graph.graph", "wb") as f:
    for src, dst in edges:
        # Write each edge as two 4-byte integers (source and destination)
        f.write(struct.pack("ii", src, dst))

print("Binary file 'sample_graph.bin' created with edges in binary format.")
