#include <bits/stdc++.h>
using namespace std;

struct SubGraph {
    vector<int> nodes;
    vector<vector<int>> adj;
};

vector<SubGraph> buildSubgraphsAfterSeparator(
    const vector<vector<int>> &G,
    const unordered_set<int> &separator,
    const vector<int> &allNodes
) {
    unordered_set<int> validNodes;
    for (int v : allNodes) {
        if (!separator.count(v)) {
            validNodes.insert(v);
        }
    }

    vector<SubGraph> comps;
    unordered_map<int,int> visited;

    for (int start : allNodes) {
        if (validNodes.count(start) && visited.find(start) == visited.end()) {
            comps.push_back(SubGraph());
            int cidx = (int) comps.size() - 1;
            queue<int>q;
            q.push(start);
            visited[start] = cidx;
            comps[cidx].nodes.push_back(start);

            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (auto &nbr : G[u]) {
                    if (validNodes.count(nbr) && visited.find(nbr) == visited.end()) {
                        visited[nbr] = cidx;
                        comps[cidx].nodes.push_back(nbr);
                        q.push(nbr);
                    }
                }
            }
        }
    }

    for (auto &sg : comps) {
        unordered_map<int,int> localIdx;
        for (int i = 0; i < (int)sg.nodes.size(); i++) {
            localIdx[sg.nodes[i]] = i;
        }
        sg.adj.resize(sg.nodes.size());
        for (int i = 0; i < (int)sg.nodes.size(); i++) {
            int gv = sg.nodes[i];
            for (auto &nbr : G[gv]) {
                if (localIdx.find(nbr) != localIdx.end()) {
                    int li = localIdx[gv];
                    int lj = localIdx[nbr];
                    sg.adj[li].push_back(lj);
                }
            }
        }
    }

    return comps;
}


bool findSeparator(
    const vector<vector<int>> &G,
    const vector<int> &subNodes,
    unordered_set<int> &separator,
    vector<SubGraph> &subgraphs
) {
    if ((int)subNodes.size() < 4) {
        return false;
    }

    static std::mt19937 rng(12345);
    uniform_int_distribution<int> dist(0,(int)subNodes.size()-1);
    int startIdx = dist(rng);
    int start = subNodes[startIdx];
    vector<bool> visited(G.size(), false);
    visited[start] = true;
    queue<int>q;
    q.push(start);
    int count = 1;
    int halfGoal = (int)subNodes.size()/2;
    vector<int> frontier = {start};
    bool done = false;
    while (!q.empty() && !done) {
        int s = (int)q.size();
        vector<int> levelNodes;
        for (int i=0; i<s; i++){
            int u = q.front(); q.pop();
            for (auto &nbr : G[u]) {
                if (!visited[nbr]) {
                    visited[nbr] = true;
                    q.push(nbr);
                    levelNodes.push_back(nbr);
                }
            }
        }
        if (!levelNodes.empty()) {
            count += (int)levelNodes.size();
            frontier = levelNodes;
            if (count >= halfGoal) {
                done = true;
            }
        }
    }

    separator.clear();
    for (auto &v : frontier) {
        separator.insert(v);
    }

    subgraphs = buildSubgraphsAfterSeparator(G, separator, subNodes);

    if (subgraphs.size() <= 1) {
        return false;
    }
    return true;
}


void nestedDissection(
    const vector<vector<int>> &G,
    const vector<int> &subNodes,
    vector<int> &order
) {
    if ((int)subNodes.size() <= 3) {
        order.insert(order.end(), subNodes.begin(), subNodes.end());
        return;
    }

    unordered_set<int> separator;
    vector<SubGraph> parts;
    bool ok = findSeparator(G, subNodes, separator, parts);

    if (!ok) {
        order.insert(order.end(), subNodes.begin(), subNodes.end());
        return;
    }

    for (auto &sg : parts) {
        vector<int> subOrder;
        nestedDissection(sg.adj, sg.nodes, subOrder);
        order.insert(order.end(), subOrder.begin(), subOrder.end());
    }

    for (auto &v : separator) {
        order.push_back(v);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<vector<int>> graph(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    vector<int> allNodes(n);
    iota(allNodes.begin(), allNodes.end(), 0);

    vector<int> NDorder;
    nestedDissection(graph, allNodes, NDorder);

    cout << "Nested Dissection ordering:\n";
    for (auto &v : NDorder) {
        cout << v << " ";
    }
    cout << "\n";

    return 0;
}
