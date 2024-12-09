#include "core/graph.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: kcores [path] [k] [memory budget in GB]\n");
        exit(-1);
    }

    std::string path = argv[1];
    int k = atoi(argv[2]);
    long memory_bytes = (argc >= 4) ? atol(argv[3]) * 1024l * 1024l * 1024l : 8l * 1024l * 1024l * 1024l;

    Graph graph(path);
    graph.set_memory_bytes(memory_bytes);

    Bitmap *active_in = graph.alloc_bitmap();
    Bitmap *active_out = graph.alloc_bitmap();
    BigVector<int> degree(graph.path + "/degree", graph.vertices);
    BigVector<int> core(graph.path + "/core", graph.vertices);

    long vertex_data_bytes = (long)graph.vertices * (sizeof(int) + sizeof(int));
    graph.set_vertex_data_bytes(vertex_data_bytes);

    // Initialize degree and active vertices
    active_out->fill();
    degree.fill(0);
    graph.stream_edges<VertexId>(
        [&](Edge &e) {
            write_add(&degree[e.source], 1);
            return 0;
        },
        nullptr, 0, 0);

    // Initialize core and set initial active vertices
    int active_vertices = graph.stream_vertices<VertexId>(
        [&](VertexId i) {
            core[i] = (degree[i] >= k) ? 1 : 0;
            return core[i];
        });

    printf("Initialization complete: %d active vertices\n", active_vertices);

    // K-core decomposition iterations
    int iteration = 0;
    while (active_vertices > 0) {
        iteration++;
        printf("Iteration %d: %d active vertices\n", iteration, active_vertices);

        std::swap(active_in, active_out);
        active_out->clear();

        graph.hint(degree, core);
        active_vertices = graph.stream_edges<VertexId>(
            [&](Edge &e) {
                if (core[e.source] == 1 && core[e.target] == 0) {
                    write_add(&degree[e.target], -1);
                    if (degree[e.target] < k) {
                        core[e.target] = 0;
                        active_out->set_bit(e.target);
                        return 1;
                    }
                }
                return 0;
            },
            active_in);
    }

    // Count k-core vertices
    int kcore_vertices = graph.stream_vertices<VertexId>(
        [&](VertexId i) {
            return core[i] == 1;
        });

    printf("K-core (%d-core) decomposition complete: %d vertices remain\n", k, kcore_vertices);

    return 0;
}
