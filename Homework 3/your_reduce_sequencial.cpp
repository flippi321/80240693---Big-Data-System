#include <mpi.h>
#include <cstring>
#include <stdio.h>
#include <omp.h> // Include OpenMP header

void YOUR_Reduce(const int *sendbuf, int *recvbuf, int count) {
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize recvbuf with the values from sendbuf
    memcpy(recvbuf, sendbuf, count * sizeof(int));

    // Allocate temp_buffer once
    int* temp_buffer = new int[count];

    // Binary tree reduction
    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) == 0) {
            // Root process collects data
            if (rank + step < size) {
                MPI_Request request;

                // Start non-blocking receive
                MPI_Irecv(temp_buffer, count, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &request);
                
                // Wait for the receive to complete
                MPI_Wait(&request, MPI_STATUS_IGNORE);

                // Combine results in parallel
                #pragma omp parallel for
                for (int i = 0; i < count; i++) {
                    recvbuf[i] += temp_buffer[i];
                }
            }
        } else if (rank % step == 0) {
            // Send to parent using non-blocking send
            MPI_Request request;
            MPI_Isend(recvbuf, count, MPI_INT, rank - step, 0, MPI_COMM_WORLD, &request);
            break; 
        }
    }

    // Clean up
    delete[] temp_buffer;
}
