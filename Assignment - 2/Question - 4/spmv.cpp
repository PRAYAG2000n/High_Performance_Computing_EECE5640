#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <queue>
#include <cassert>
#include <omp.h>  // optional parallel

// A struct for storing a matrix in CSR format
struct CSR {
    int nrows, ncols;
    std::vector<int> row_ptr;    // row_ptr[i] = index of first nonzero in row i
    std::vector<int> col_idx;    // column indices of each nonzero
    std::vector<double> values;  // values of each nonzero
};

//--------------------------------------------------------------------------
// readMatrixMarket: Reads a matrix from Matrix Market (.mtx) into CSR
// with basic integrity checks.
//--------------------------------------------------------------------------
CSR readMatrixMarket(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Skip comment lines
    while (file.peek() == '%') {
        file.ignore(2048, '\n');
    }

    int M, N, nnz;
    file >> M >> N >> nnz;

    if (M <= 0 || N <= 0 || nnz < 0) {
        std::cerr << "Error: invalid matrix dimensions or nnz.\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<int> row_coo(nnz), col_coo(nnz);
    std::vector<double> val_coo(nnz);

    for(int i = 0; i < nnz; i++){
        int r, c;
        double val;
        file >> r >> c >> val;
        if (file.fail()) {
            std::cerr << "Error reading matrix entry " << i << "\n";
            std::exit(EXIT_FAILURE);
        }
        // Convert from 1-based to 0-based indexing
        r -= 1;
        c -= 1;
        if(r < 0 || r >= M || c < 0 || c >= N) {
            std::cerr << "Error: out-of-range index in .mtx at entry " << i << "\n";
            std::exit(EXIT_FAILURE);
        }
        row_coo[i] = r;
        col_coo[i] = c;
        val_coo[i] = val;
    }

    CSR csr;
    csr.nrows = M;
    csr.ncols = N;
    csr.row_ptr.resize(M+1, 0);

    // Build row_ptr
    for(int i = 0; i < nnz; i++){
        csr.row_ptr[row_coo[i] + 1]++;
    }
    for(int i = 1; i <= M; i++){
        csr.row_ptr[i] += csr.row_ptr[i-1];
    }

    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);

    // Fill col_idx, values
    std::vector<int> offset(M, 0);
    for(int i = 0; i < nnz; i++){
        int r = row_coo[i];
        int dest = csr.row_ptr[r] + offset[r]++;
        csr.col_idx[dest] = col_coo[i];
        csr.values[dest]  = val_coo[i];
    }

    // Basic row_ptr integrity checks
    for(int i = 0; i < M; i++){
        if (csr.row_ptr[i] > csr.row_ptr[i+1]) {
            std::cerr << "Error: row_ptr not sorted at i=" << i << "\n";
            std::exit(EXIT_FAILURE);
        }
    }
    if (csr.row_ptr[M] != (int)csr.col_idx.size()) {
        std::cerr << "Error: row_ptr[nrows] != col_idx.size() mismatch.\n";
        std::exit(EXIT_FAILURE);
    }

    return csr;
}

//--------------------------------------------------------------------------
// buildUndirectedAdj: build an undirected adjacency list from CSR
// for RCM. If the matrix is not structurally symmetric, this function
// "symmetrizes" by adding edges i->col and col->i whenever col<nrows.
//--------------------------------------------------------------------------
std::vector<std::vector<int>> buildUndirectedAdj(const CSR &csr) {
    int n = csr.nrows;
    // adjacency[i] = list of neighbors of i
    std::vector<std::vector<int>> adjacency(n);

    for(int i = 0; i < n; i++){
        int start = csr.row_ptr[i];
        int end   = csr.row_ptr[i+1];
        for(int k = start; k < end; k++){
            int c = csr.col_idx[k];
            // We'll only consider c < n to ensure we treat row indices as vertices
            // (Otherwise, if c >= n, it doesn't correspond to a row-vertex.)
            if(c < n && c >= 0) {
                // add i->c
                adjacency[i].push_back(c);
                // add c->i
                adjacency[c].push_back(i);
            }
        }
    }

    // Remove duplicates from adjacency
    for(int i = 0; i < n; i++){
        auto &nbrs = adjacency[i];
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }

    return adjacency;
}

//--------------------------------------------------------------------------
// RCM: Reverse Cuthill–McKee over all connected components
// 1. For each unvisited node, do BFS, sorting neighbors by ascending degree
// 2. Add BFS result to overall order
// 3. Reverse final BFS result
//--------------------------------------------------------------------------
// This ensures that we do not skip disconnected components (which can cause
// partial BFS that leads to out-of-range permutations).
//--------------------------------------------------------------------------
std::vector<int> RCM(const std::vector<std::vector<int>> &adj) {
    int n = (int)adj.size();
    std::vector<bool> visited(n, false);
    std::vector<int> order; 
    order.reserve(n);

    auto bfs_component = [&](int start) {
        std::queue<int> Q;
        Q.push(start);
        visited[start] = true;

        std::vector<int> local_bfs;
        local_bfs.reserve(n);

        while(!Q.empty()){
            int u = Q.front();
            Q.pop();
            local_bfs.push_back(u);

            // Sort neighbors by ascending degree
            std::vector<int> neighs = adj[u];
            std::sort(neighs.begin(), neighs.end(), [&](int a, int b){
                return (int)adj[a].size() < (int)adj[b].size();
            });

            // push unvisited
            for (auto &v : neighs) {
                if (!visited[v]) {
                    visited[v] = true;
                    Q.push(v);
                }
            }
        }
        // Append BFS of this component to global order
        // but do NOT reverse it here; we do a single global reverse at the end.
        order.insert(order.end(), local_bfs.begin(), local_bfs.end());
    };

    // BFS from each unvisited node => covers all components
    for(int i = 0; i < n; i++){
        if(!visited[i]) {
            bfs_component(i);
        }
    }
    // Now do a single global reverse to get RCM
    std::reverse(order.begin(), order.end());
    // Check we visited all
    if((int)order.size() != n) {
        std::cerr << "Warning: BFS covered " << order.size() 
                  << " out of " << n << " vertices.\n";
        // For safety, but normally we should always match n
    }
    return order;
}

//--------------------------------------------------------------------------
// applySymPermutation: apply "p[new_r] = old_r" to rows AND columns
// with extra index checks. p must be size n. 
// This ensures we won't write out of bounds if p is malformed.
//--------------------------------------------------------------------------
CSR applySymPermutation(const CSR &csr, const std::vector<int> &p) {
    int n = csr.nrows;
    if ((int)p.size() != n) {
        std::cerr << "Error: permutation size != matrix dimension.\n";
        std::exit(EXIT_FAILURE);
    }

    // Build pinv such that pinv[old_r] = new_r
    std::vector<int> pinv(n, -1);
    for(int newr = 0; newr < n; newr++){
        int oldr = p[newr];
        if(oldr < 0 || oldr >= n) {
            std::cerr << "Error: permutation out-of-range at index "
                      << newr << " => oldr=" << oldr << "\n";
            std::exit(EXIT_FAILURE);
        }
        pinv[oldr] = newr;
    }

    // check if any pinv remains -1 => invalid p
    for(int i = 0; i < n; i++){
        if(pinv[i] < 0) {
            std::cerr << "Error: permutation does not map row "
                      << i << " properly.\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // Gather new matrix in COO
    std::vector<int> new_rows, new_cols;
    std::vector<double> new_vals;
    new_rows.reserve(csr.col_idx.size());
    new_cols.reserve(csr.col_idx.size());
    new_vals.reserve(csr.col_idx.size());

    for(int old_r = 0; old_r < n; old_r++){
        int start = csr.row_ptr[old_r];
        int end   = csr.row_ptr[old_r+1];
        int new_r = pinv[old_r];
        for(int k = start; k < end; k++){
            int old_c = csr.col_idx[k];
            if(old_c < 0 || old_c >= csr.ncols) {
                // If old_c is out-of-bounds, skip or handle error
                continue;
            }
            double val = csr.values[k];

            // new column = pinv[old_c] if old_c < n
            // If old_c >= n, it's outside row domain => skip
            if(old_c < n) {
                int new_c = pinv[old_c];
                new_rows.push_back(new_r);
                new_cols.push_back(new_c);
                new_vals.push_back(val);
            }
        }
    }

    // Sort by (new_rows, new_cols)
    std::vector<int> idx(new_rows.size());
    for(size_t i = 0; i < idx.size(); i++){
        idx[i] = (int)i;
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        if(new_rows[a] == new_rows[b])
            return new_cols[a] < new_cols[b];
        return new_rows[a] < new_rows[b];
    });

    CSR out;
    out.nrows = n;
    out.ncols = n;
    out.row_ptr.resize(n+1, 0);

    for(auto i : idx){
        // new_rows[i] might be out-of-range => check
        int rr = new_rows[i];
        if(rr < 0 || rr >= n){
            std::cerr << "Error: new_rows["<<i<<"]="<<rr<<" out-of-range.\n";
            std::exit(EXIT_FAILURE);
        }
        out.row_ptr[rr + 1]++;
    }
    // prefix sum
    for(int i = 1; i <= n; i++){
        out.row_ptr[i] += out.row_ptr[i-1];
    }

    out.col_idx.resize(idx.size());
    out.values.resize(idx.size());

    std::vector<int> offset(n, 0);
    for(auto i : idx){
        int r = new_rows[i];
        int c = new_cols[i];
        double v = new_vals[i];
        if(r < 0 || r >= n || c < 0 || c >= n) {
            std::cerr << "Error: (r,c)=("<<r<<","<<c<<") out-of-range.\n";
            std::exit(EXIT_FAILURE);
        }
        int dest = out.row_ptr[r] + offset[r]++;
        out.col_idx[dest] = c;
        out.values[dest]  = v;
    }

    // Final sanity check
    if(!out.col_idx.empty()) {
        if((int)out.col_idx.size() != (int)out.values.size()) {
            std::cerr << "Error: mismatch in col_idx vs values size.\n";
            std::exit(EXIT_FAILURE);
        }
    }
    if(out.row_ptr[n] != (int)out.col_idx.size()){
        std::cerr << "Error: row_ptr[n] != number of new nonzeros.\n";
        std::exit(EXIT_FAILURE);
    }

    return out;
}

//--------------------------------------------------------------------------
// spmv_csr: safe version with bounds checking for demonstration
// (You can remove bounds checks for performance in production.)
//--------------------------------------------------------------------------
std::vector<double> spmv_csr(const CSR &A, const std::vector<double> &x) {
    if((int)x.size() < A.ncols) {
        std::cerr << "Error: x.size() < A.ncols.\n";
        std::exit(EXIT_FAILURE);
    }
    std::vector<double> y(A.nrows, 0.0);

    #pragma omp parallel for
    for(int i = 0; i < A.nrows; i++){
        double sum = 0.0;
        int start = A.row_ptr[i];
        int end   = A.row_ptr[i+1];
        for(int k = start; k < end; k++){
            int c = A.col_idx[k];
            if(c < 0 || c >= A.ncols) {
                std::cerr << "Error: c=" << c 
                          << " out-of-bounds during spmv.\n";
                std::exit(EXIT_FAILURE);
            }
            sum += A.values[k] * x[c];
        }
        y[i] = sum;
    }
    return y;
}

//--------------------------------------------------------------------------
// dense_matvec: naive NxN multiply
//--------------------------------------------------------------------------
std::vector<double> dense_matvec(const std::vector<double> &A_dense,
                                 const std::vector<double> &x, int N) {
    if((int)A_dense.size() < N*N) {
        std::cerr << "Error: A_dense too small for NxN.\n";
        std::exit(EXIT_FAILURE);
    }
    if((int)x.size() < N) {
        std::cerr << "Error: x.size() < N in dense_matvec.\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<double> y(N, 0.0);
    #pragma omp parallel for
    for(int i = 0; i < N; i++){
        double sum = 0.0;
        for(int j = 0; j < N; j++){
            sum += A_dense[i*N + j] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

//--------------------------------------------------------------------------
// main: Tie everything together with extra checks
//--------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx>\n";
        return 1;
    }

    std::string mtx_file = argv[1];
    // 1) Read the matrix
    CSR A = readMatrixMarket(mtx_file);
    std::cout << "Matrix loaded: " << A.nrows << " x " << A.ncols 
              << ", nnz=" << A.row_ptr[A.nrows] << "\n";

    // 2) Build undirected adjacency
    auto adjacency = buildUndirectedAdj(A);

    // 3) RCM reordering over all connected components
    double t0 = omp_get_wtime();
    std::vector<int> perm = RCM(adjacency);
    double t1 = omp_get_wtime();
    double rcmTime = t1 - t0;
    std::cout << "RCM ordering took: " << rcmTime << " seconds.\n";

    // 4) Apply the permutation
    double t2 = omp_get_wtime();
    CSR A_rcm = applySymPermutation(A, perm);
    double t3 = omp_get_wtime();
    double permTime = t3 - t2;
    std::cout << "Applying permutation took: " << permTime << " seconds.\n\n";

    // 5) Build a dense version for naive comparison (if feasible)
    if((long long)A.nrows * (long long)A.ncols > 10000000) {
        std::cout << "Matrix too large for naive dense array; skipping.\n";
    }

    // If it's small enough, do the naive dense
    bool doDense = ((long long)A.nrows * (long long)A.ncols <= 10000000LL);

    std::vector<double> A_dense;
    if(doDense) {
        A_dense.resize((size_t)A.nrows * (size_t)A.ncols, 0.0);
        for(int i = 0; i < A.nrows; i++){
            for(int k = A.row_ptr[i]; k < A.row_ptr[i+1]; k++){
                int c = A.col_idx[k];
                A_dense[(long long)i*A.ncols + c] = A.values[k];
            }
        }
    }

    // Prepare x
    std::vector<double> x(A.ncols, 1.0);

    // Warm-up 
    auto y_orig = spmv_csr(A, x);
    auto y_rcm  = spmv_csr(A_rcm, x);

    // Timings
    const int reps = 5;
    double start, end;

    // Original CSR
    start = omp_get_wtime();
    for(int r = 0; r < reps; r++){
        y_orig = spmv_csr(A, x);
    }
    end = omp_get_wtime();
    double origTime = (end - start)/reps;

    // RCM CSR
    start = omp_get_wtime();
    for(int r = 0; r < reps; r++){
        y_rcm = spmv_csr(A_rcm, x);
    }
    end = omp_get_wtime();
    double rcmSpmvTime = (end - start)/reps;

    // Dense if feasible
    double denseTime = 0.0;
    if(doDense) {
        // Warm-up
        auto y_dense = dense_matvec(A_dense, x, A.nrows);
        start = omp_get_wtime();
        for(int r = 0; r < reps; r++){
            y_dense = dense_matvec(A_dense, x, A.nrows);
        }
        end = omp_get_wtime();
        denseTime = (end - start)/reps;
    }

    // Results
    std::cout << "CSR original SpMV time (avg): " << origTime << " s\n"
              << "CSR RCM SpMV time (avg):      " << rcmSpmvTime << " s\n";

    if(doDense) {
        std::cout << "Naive dense matvec time (avg): " << denseTime << " s\n";
        double spdDense = (rcmSpmvTime > 0.0) ? (denseTime / rcmSpmvTime) : 0.0;
        std::cout << "Speedup vs dense = " << spdDense << "\n";
    }
    double spdOrig = (rcmSpmvTime > 0.0) ? (origTime / rcmSpmvTime) : 0.0;
    std::cout << "Speedup vs original CSR = " << spdOrig << "\n";

    return 0;
}
