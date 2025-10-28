#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <limits>

void reverseCuthillMcKee(int n,
                         const std::vector<int> &rowPtr,
                         const std::vector<int> &colIdx,
                         std::vector<int> &rcmOrder)
{
    rcmOrder.resize(n, -1);
    int start = 0;
    int minDeg = std::numeric_limits<int>::max();
    for (int i = 0; i < n; i++) {
        int deg = rowPtr[i+1] - rowPtr[i];
        if (deg < minDeg) {
            minDeg = deg;
            start = i;
        }
    }
    std::vector<bool> visited(n, false);
    visited[start] = true;
    std::queue<int> Q;
    Q.push(start);
    int idx = 0;
    while (!Q.empty()) {
        int node = Q.front();
        Q.pop();
        rcmOrder[idx++] = node;
        std::vector<int> neighbors;
        for (int k = rowPtr[node]; k < rowPtr[node+1]; k++) {
            int nbr = colIdx[k];
            if (!visited[nbr]) {
                neighbors.push_back(nbr);
                visited[nbr] = true;
            }
        }
        std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
            int degA = rowPtr[a+1] - rowPtr[a];
            int degB = rowPtr[b+1] - rowPtr[b];
            return degA < degB;
        });
        for (int nbr : neighbors) {
            Q.push(nbr);
        }
    }
    std::reverse(rcmOrder.begin(), rcmOrder.end());
}

int main()
{
    int n = 5;
    std::vector<int> rowPtr = {0, 2, 5, 8, 10, 12};
    std::vector<int> colIdx = {
        1, 3,
        0, 2, 4,
        1, 3, 4,
        0, 2,
        1, 2
    };
    std::vector<int> rcmOrder;
    reverseCuthillMcKee(n, rowPtr, colIdx, rcmOrder);
    std::cout << "RCM order: ";
    for (auto &r : rcmOrder) {
        std::cout << r << " ";
    }
    std::cout << std::endl;
    return 0;
}






