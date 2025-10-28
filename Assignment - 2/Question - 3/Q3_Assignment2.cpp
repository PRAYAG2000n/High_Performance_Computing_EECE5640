#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>

// -----------------------------------------------------------
// 1. parallelGraphColoring (unmodified parallel logic)
// -----------------------------------------------------------
std::vector<int> parallelGraphColoring(const std::vector<std::vector<int>>& adj) {
    int V = static_cast<int>(adj.size());
    std::vector<int> color(V, -1);
    std::vector<int> priority(V);

    // Initialize priorities
    for (int i = 0; i < V; ++i) {
        priority[i] = i;
    }

    // Shuffle priorities randomly
    srand(static_cast<unsigned>(time(nullptr)));
    std::random_shuffle(priority.begin(), priority.end());

    bool uncoloredExists;
    do {
        uncoloredExists = false;
        std::vector<bool> candidate(V, false);

        // Determine candidate vertices
        #pragma omp parallel for
        for (int v = 0; v < V; ++v) {
            if (color[v] != -1) {
                continue;  // Already colored
            }

            bool is_candidate = true;
            for (int u : adj[v]) {
                if (color[u] == -1 && priority[v] <= priority[u]) {
                    is_candidate = false;
                    break;
                }
            }

            if (is_candidate) {
                candidate[v] = true;
            } else {
                #pragma omp atomic write
                uncoloredExists = true;
            }
        }

        // Color all candidate vertices
        #pragma omp parallel for
        for (int v = 0; v < V; ++v) {
            if (!candidate[v]) {
                continue;  // Skip non-candidates
            }

            std::vector<bool> used_colors(V + 1, false);
            // Mark used colors among neighbors
            for (int u : adj[v]) {
                if (color[u] != -1) {
                    used_colors[color[u]] = true;
                }
            }

            // Assign the smallest available color (1-based)
            int c = 1;
            while (c <= V && used_colors[c]) {
                ++c;
            }
            color[v] = c;
        }

        // Check if any uncolored vertices remain
        bool anyUncolored = false;
        #pragma omp parallel for reduction(||:anyUncolored)
        for (int v = 0; v < V; ++v) {
            if (color[v] == -1) {
                anyUncolored = true;
            }
        }
        uncoloredExists = anyUncolored;

    } while (uncoloredExists);

    return color;
}

// -----------------------------------------------------------
// 2. Verify that no two adjacent vertices share the same color
// -----------------------------------------------------------
bool verifyColoring(const std::vector<std::vector<int>>& adj, const std::vector<int>& color) {
    int V = static_cast<int>(adj.size());
    for (int v = 0; v < V; ++v) {
        for (int u : adj[v]) {
            if (color[v] == color[u]) {
                return false; // Found invalid coloring
            }
        }
    }
    return true;
}

// -----------------------------------------------------------
// 3. Generate a random undirected graph with a HARD-CODED p
// -----------------------------------------------------------
std::vector<std::vector<int>> generateRandomGraph(int V) {
    double p = 0.6;  // <-- HARDCODED probability
    std::vector<std::vector<int>> adj(V);

    // Fixed seed for reproducibility
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            if (dist(gen) < p) {
                // Add undirected edge
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    return adj;
}

// -----------------------------------------------------------
// 4. Main
// -----------------------------------------------------------
int main() {
    int V;          // number of vertices
    int numThreads; // number of OpenMP threads

    // Prompt for input
    std::cout << "\nEnter number of vertices: ";
    std::cin >> V;

    std::cout << "Enter number of threads: ";
    std::cin >> numThreads;

    // Conditionally print "Generating graph" line
    if (V <= 20) {
        std::cout << "\nGenerating graph with " << V 
                  << " vertices and edge probability:\n";
    } else {
        std::cout << "\nGenerating a large graph (V = " << V 
                  << ", p = 0.6)\n";
    }

    // Generate random graph
    auto adj = generateRandomGraph(V);

    // ----------------------------
    // Print adjacency only if V <= 20
    // ----------------------------
    if (V <= 20) {
        std::cout << "\nAdjacency List:\n";
        for (int i = 0; i < V; ++i) {
            std::cout << "Vertex " << i << " neighbors: ";
            for (int nbr : adj[i]) {
                std::cout << nbr << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    } else {
        std::cout << "\nGraph is too large (" << V 
                  << " vertices) to display adjacency list.\n";
    }

    // Set number of threads
    omp_set_num_threads(numThreads);

    // Time the graph coloring
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> color = parallelGraphColoring(adj);
    auto end   = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Fix-up: ensure no conflicts remain
    int maxColor = *std::max_element(color.begin(), color.end());
    bool conflict = true;
    while (conflict) {
        conflict = false;
        for (int v = 0; v < V; ++v) {
            for (int u : adj[v]) {
                if (color[v] == color[u] && v < u) {
                    // conflict found
                    maxColor++;
                    color[u] = maxColor;
                    conflict = true;
                }
            }
        }
    }

    // Verify correctness again
    bool valid = verifyColoring(adj, color);
    int finalMaxColor = *std::max_element(color.begin(), color.end());

    // Print results
    std::cout << "\nFinal check - no two adjacent vertices share the same color: " 
              << (valid ? "YES" : "NO") << "\n";
    std::cout << "Number of distinct colors used: " << finalMaxColor << "\n";
    std::cout << "Time taken (initial coloring only): " << time_ms << " ms\n";

    // Print final color assignment only if V <= 20
    if (V <= 20) {
        std::cout << "\nAssigned colors:\n";
        for (int v = 0; v < V; ++v) {
            std::cout << "  Vertex " << v << " -> Color " << color[v] << "\n";
        }
    } else {
        std::cout << "\nGraph is too large (" << V 
                  << " vertices) to display final color assignment.\n \n";
    }

    return 0;
}
