#include <mpi.h>
#include <cstring>
#include <stdio.h>

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
                MPI_Recv(temp_buffer, count, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Combine results using a loop
                for (int i = 0; i < count; i++) {
                    recvbuf[i] += temp_buffer[i];
                }
            }
        } else if (rank % step == 0) {
            // Send to parent
            MPI_Send(recvbuf, count, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break; 
        }
    }

    // Clean up
    delete[] temp_buffer;
}