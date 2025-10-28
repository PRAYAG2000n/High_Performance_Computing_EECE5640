#include <mpi.h>
#include <iostream>
#include <unistd.h>     // For gethostname() on many systems
#include <cstring>

// We define a small struct-like container to hold both the integer value
// and a "phase" flag indicating incrementing or decrementing.
struct Token {
    int value;  // The integer being passed
    int phase;  // 0 => increment phase; 1 => decrement phase
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // We want at least 64 processes total
    if (world_size < 64) {
        if (world_rank == 0) {
            std::cerr << "Error: At least 64 MPI processes are required.\n";
        }
        MPI_Finalize();
        return -1;
    }

    // Retrieve hostname for printing
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    hostname[sizeof(hostname) - 1] = '\0'; // Safely null-terminate

    // For a ring: next rank is (rank + 1) mod world_size
    //             previous rank is (rank - 1 + world_size) mod world_size
    int next_rank = (world_rank + 1) % world_size;
    int prev_rank = (world_rank - 1 + world_size) % world_size;

    // Define tags for sending/receiving (just use 0 and 1, or the same tag with multiple calls)
    const int TAG_TOKEN = 0;
    const int TAG_PHASE = 1;

    // Initialize token only on rank 0
    Token token;
    if (world_rank == 0) {
        // Start with value=0, in "increment" phase
        token.value = 0;
        token.phase = 0;  // 0 => increment

        // Send to the next rank to kick things off
        MPI_Send(&token, 2, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    while (true) {
        // Each rank receives the token from its previous neighbor
        MPI_Recv(&token, 2, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // If token.value is negative, it means "no more work"; break out of loop
        // We'll use a negative token to indicate the ring is shutting down.
        if (token.value < 0) {
            break;
        }

        // Print the received value, rank, and node
        // The *received* token.value is what we display
        std::cout << "Process " << world_rank << " on node " << hostname
                  << " received value: " << token.value << std::endl;

        // Now update the token for the next send
        if (token.phase == 0) {
            // Increment phase
            token.value++;
            // If we've just reached 64, switch to decrement phase
            if (token.value == 64) {
                token.phase = 1; // Switch to decrement
                // Immediately decrement by 2 in the same step
                token.value -= 2;
            }
        } else {
            // Decrement phase
            token.value -= 2;
        }

        // If token <= 0, we mark it as negative to indicate the ring is done
        if (token.value <= 0) {
            // Set token.value to -1 so the entire ring stops
            token.value = -1;
        }

        // Forward token to next rank, so everyone stays in sync
        MPI_Send(&token, 2, MPI_INT, next_rank, 0, MPI_COMM_WORLD);

        // If token.value < 0, we're done. This rank stops receiving/sending.
        if (token.value < 0) {
            break;
        }
    }

    MPI_Finalize();
    return 0;
}
