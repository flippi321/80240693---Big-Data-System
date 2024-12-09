#include "core/graph.hpp"

int main(int argc, char ** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: pagerank_delta [path] [iterations] [memory budget in GB]\n");
		exit(-1);
	}
	std::string path = argv[1];
	int iterations = atoi(argv[2]);
	long memory_bytes = (argc >= 4) ? atol(argv[3]) * 1024l * 1024l * 1024l : 8l * 1024l * 1024l * 1024l;

	Graph graph(path);
	graph.set_memory_bytes(memory_bytes);

	BigVector<VertexId> degree(graph.path + "/degree", graph.vertices);
	BigVector<float> pagerank(graph.path + "/pagerank", graph.vertices);
	BigVector<float> delta(graph.path + "/delta", graph.vertices);
	BigVector<float> new_delta(graph.path + "/new_delta", graph.vertices);

	long vertex_data_bytes = (long)graph.vertices * (sizeof(VertexId) + sizeof(float) * 3);
	graph.set_vertex_data_bytes(vertex_data_bytes);

	double begin_time = get_time();

	// Initialize degrees
	degree.fill(0);
	graph.stream_edges<VertexId>(
		[&](Edge & e) {
			write_add(&degree[e.source], 1);
			return 0;
		}, nullptr, 0, 0
	);

	// Initialize Pagerank and Delta
	graph.hint(pagerank, delta, new_delta);
	graph.stream_vertices<VertexId>(
		[&](VertexId i) {
			pagerank[i] = 0.15f;  // Initial PageRank value
			delta[i] = 1.0f / degree[i];  // Initial delta
			new_delta[i] = 0;
			return 0;
		}, nullptr, 0,
		[&](std::pair<VertexId, VertexId> vid_range) {
			pagerank.load(vid_range.first, vid_range.second);
			delta.load(vid_range.first, vid_range.second);
			new_delta.load(vid_range.first, vid_range.second);
		},
		[&](std::pair<VertexId, VertexId> vid_range) {
			pagerank.save();
			delta.save();
			new_delta.save();
		}
	);

	// PageRank Delta Iterations
	for (int iter = 0; iter < iterations; iter++) {
		graph.hint(delta, new_delta);
		graph.stream_edges<VertexId>(
			[&](Edge & e) {
				write_add(&new_delta[e.target], 0.85f * delta[e.source]);
				return 0;
			}, nullptr, 0, 1,
			[&](std::pair<VertexId, VertexId> source_vid_range) {
				delta.lock(source_vid_range.first, source_vid_range.second);
			},
			[&](std::pair<VertexId, VertexId> source_vid_range) {
				delta.unlock(source_vid_range.first, source_vid_range.second);
			}
		);

		graph.hint(pagerank, delta, new_delta);
		graph.stream_vertices<float>(
			[&](VertexId i) {
				pagerank[i] += new_delta[i];
				delta[i] = new_delta[i] / degree[i];
				new_delta[i] = 0;  // Reset for next iteration
				return 0;
			}, nullptr, 0,
			[&](std::pair<VertexId, VertexId> vid_range) {
				pagerank.load(vid_range.first, vid_range.second);
				delta.load(vid_range.first, vid_range.second);
				new_delta.load(vid_range.first, vid_range.second);
			},
			[&](std::pair<VertexId, VertexId> vid_range) {
				pagerank.save();
				delta.save();
				new_delta.save();
			}
		);
	}

	double end_time = get_time();
	printf("%d iterations of pagerank delta took %.2f seconds\n", iterations, end_time - begin_time);

	return 0;
}
