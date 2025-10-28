#include <bits/stdc++.h>
using namespace std;

static int compute_cut(
    const vector<vector<int>> &adj,
    const vector<bool> &inA
) {
    int cut = 0;
    int n = (int)adj.size();
    for(int u=0; u<n; u++){
        for(auto &v : adj[u]){
            if(u < v){
                bool Au = inA[u];
                bool Av = inA[v];
                if(Au != Av){
                    cut++;
                }
            }
        }
    }
    return cut;
}

static int compute_D(
    int u,
    const vector<vector<int>> &adj,
    const vector<bool> &inA
) {
    int ext = 0, inr = 0;
    for(auto &nbr : adj[u]){
        if(inA[u] == inA[nbr]){
            inr++;
        } else {
            ext++;
        }
    }
    return (ext - inr);
}

static void recompute_gains(
    const vector<vector<int>> &adj,
    const vector<bool> &inA,
    const vector<bool> &locked,
    vector<int> &D
){
    int n = (int)adj.size();
    for(int u=0; u<n; u++){
        if(!locked[u]){
            D[u] = compute_D(u, adj, inA);
        }
    }
}

static int KL_pass(
    const vector<vector<int>> &adj,
    vector<bool> &inA
) {
    int n = (int)adj.size();
    int sizeA = 0; 
    for(int u=0; u<n; u++){
        if(inA[u]) sizeA++;
    }
    int sizeB = n - sizeA;
    vector<int> D(n);
    vector<bool> locked(n, false);
    recompute_gains(adj, inA, locked, D);
    vector<pair<int,int>> chosenPairs; 
    vector<int> gains;
    chosenPairs.reserve(n/2);
    gains.reserve(n/2);
    int maxSwaps = min(sizeA, sizeB);
    for(int step = 0; step < maxSwaps; step++){
        int bestU = -1, bestW = -1;
        int bestGain = -9999999;
        for(int u=0; u<n; u++){
            if(!locked[u] && inA[u]){
                for(auto &w : adj[u]){
                    if(!locked[w] && !inA[w]) {
                        int curGain = D[u] + D[w];
                        if(curGain > bestGain){
                            bestGain = curGain;
                            bestU = u;
                            bestW = w;
                        }
                    }
                }
            }
        }
        if(bestU == -1 || bestW == -1){
            break;
        }
        chosenPairs.push_back({bestU, bestW});
        gains.push_back(bestGain);
        locked[bestU] = true;
        locked[bestW] = true;
        inA[bestU] = false;
        inA[bestW] = true;
        for(auto &nbr : adj[bestU]){
            if(!locked[nbr]){
                D[nbr] = compute_D(nbr, adj, inA);
            }
        }
        for(auto &nbr : adj[bestW]){
            if(!locked[nbr]){
                D[nbr] = compute_D(nbr, adj, inA);
            }
        }
    }
    int bestPrefixIndex = -1;
    int runningSum = 0;
    int bestSum = -1;
    for(int i=0; i<(int)gains.size(); i++){
        runningSum += gains[i];
        if(runningSum > bestSum){
            bestSum = runningSum;
            bestPrefixIndex = i;
        }
    }
    if(bestSum <= 0 || bestPrefixIndex < 0){
        for(int i=0; i<(int)chosenPairs.size(); i++){
            auto &pp = chosenPairs[i];
            inA[pp.first] = true;
            inA[pp.second] = false;
        }
        return 0;
    } else {
        for(int i=bestPrefixIndex+1; i<(int)chosenPairs.size(); i++){
            auto &pp = chosenPairs[i];
            inA[pp.first] = true;
            inA[pp.second] = false;
        }
        return bestSum;
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    int n, m;
    cin >> n >> m;
    vector<vector<int>> adj(n);
    for(int i=0; i<m; i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 0);
    random_shuffle(nodes.begin(), nodes.end());
    vector<bool> inA(n,false);
    for(int i=0; i<n/2; i++){
        inA[nodes[i]] = true;
    }
    int initialCut = compute_cut(adj, inA);
    cerr << "Initial cut: " << initialCut << "\n";
    int maxIter = 10; 
    int bestCut = initialCut;
    for(int iter=0; iter<maxIter; iter++){
        int improvement = KL_pass(adj, inA);
        if(improvement <= 0){
            break;
        }
        int newCut = compute_cut(adj, inA);
        if(newCut < bestCut){
            bestCut = newCut;
        } else {
            break;
        }
        cerr << "Iteration " << iter << " cut: " << bestCut << " (improved by " << (initialCut - bestCut) << " from initial)\n";
    }
    cout << "Final cut cost: " << bestCut << "\n";
    cout << "Partition (vertex -> A/B):\n";
    for(int v=0; v<n; v++){
        cout << v << " " << (inA[v] ? "A" : "B") << "\n";
    }
    return 0;
}

