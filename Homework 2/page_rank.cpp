#include "page_rank.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <utility>

#include "./common/CycleTimer.h"
#include "./common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    double *solution_new = new double[numNodes];
    double *score_old = solution;
    double *score_new = solution_new;
    bool converged = false;
    double broadcastScore = 0.0;
    double globalDiff = 0.0;
    int iter = 0;

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    while (!converged && iter < MAXITER) {
        iter++;
        broadcastScore = 0.0;
        globalDiff = 0.0;

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < numNodes; i++) {
            score_new[i] = 0.0;

            const Vertex *in_begin = incoming_begin(g, i);
            const Vertex *in_end = incoming_end(g, i);
            for (const Vertex *v = in_begin; v < in_end; ++v) {
                score_new[i] += score_old[*v] / outgoing_size(g, *v);
            }
            score_new[i] = damping * score_new[i] + (1.0 - damping) * equal_prob;
        }
        
        #pragma omp parallel for reduction(+:broadcastScore) schedule(guided)
        for (int i = 0; i < numNodes; i++) {
            if (outgoing_size(g, i) == 0) {
                broadcastScore += score_old[i];
            }
        }
  
        #pragma omp parallel for reduction(+:globalDiff)
        for (int i = 0; i < numNodes; i++) {
            score_new[i] += damping * broadcastScore * equal_prob;
            globalDiff += std::abs(score_new[i] - score_old[i]);
        }

        converged = (globalDiff < convergence);
        std::swap(score_new, score_old);
    }
    if (score_new != solution) {
        memcpy(solution, score_new, sizeof(double) * numNodes);
    }
    delete[] solution_new;
}
