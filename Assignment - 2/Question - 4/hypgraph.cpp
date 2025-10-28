#include <bits/stdc++.h>
using namespace std;

struct Hypergraph {
    int n;
    int m;
    vector<vector<int>> edges;
    vector<vector<int>> v2edges;
};

Hypergraph readHypergraph() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    Hypergraph H;
    cin >> H.n >> H.m;
    H.edges.resize(H.m);
    H.v2edges.resize(H.n);

    for(int e = 0; e < H.m; e++){
        int k;
        cin >> k;
        H.edges[e].resize(k);
        for(int i=0; i<k; i++){
            int v;
            cin >> v;
            H.edges[e][i] = v;
            H.v2edges[v].push_back(e);
        }
    }
    return H;
}

int hyperedgeCutCost(const Hypergraph &H, const vector<bool> &inA)
{
    int cut = 0;
    for(int e = 0; e < H.m; e++){
        bool hasA = false, hasB = false;
        for(auto v : H.edges[e]){
            if(inA[v]) {
                hasA = true;
            } else {
                hasB = true;
            }
            if(hasA && hasB) {
                cut++;
                break;
            }
        }
    }
    return cut;
}

int computeGain(const Hypergraph &H, int v,
                const vector<bool> &inA)
{
    bool oldSide = inA[v];
    bool newSide = !oldSide;

    int fixCount = 0;
    int breakCount = 0;

    for(auto e : H.v2edges[v]){
        bool hasA = false, hasB = false;
        for(auto w : H.edges[e]){
            if(w == v) continue;
            if(inA[w]) hasA = true;
            else       hasB = true;
            if(hasA && hasB) break;
        }
        bool oldCut = (hasA || oldSide) && (hasB || !oldSide);
        bool newCut = (hasA || newSide) && (hasB || !newSide);

        if(oldCut && !newCut) fixCount++;
        if(!oldCut && newCut) breakCount++;
    }

    return fixCount - breakCount;
}

void recomputeAllGains(const Hypergraph &H,
                       const vector<bool> &locked,
                       const vector<bool> &inA,
                       vector<int> &gain)
{
    int n = H.n;
    for(int v = 0; v < n; v++){
        if(!locked[v]){
            gain[v] = computeGain(H, v, inA);
        }
    }
}

int fm_pass(Hypergraph &H, vector<bool> &inA)
{
    int n = H.n;
    vector<int> gain(n, 0);
    vector<bool> locked(n, false);
    recomputeAllGains(H, locked, inA, gain);

    vector<int> movedVertices; 
    vector<int> partialSums;

    for(int step = 0; step < n; step++){
        int bestV = -1;
        int bestG = -99999999;
        for(int v = 0; v < n; v++){
            if(!locked[v]){
                if(gain[v] > bestG){
                    bestG = gain[v];
                    bestV = v;
                }
            }
        }
        if(bestV == -1){
            break;
        }

        locked[bestV] = true;
        inA[bestV] = !inA[bestV];

        if(step == 0) partialSums.push_back(bestG);
        else partialSums.push_back(partialSums.back() + bestG);
        movedVertices.push_back(bestV);

        for(auto e : H.v2edges[bestV]){
            for(auto w : H.edges[e]){
                if(!locked[w] && w != bestV){
                    gain[w] = computeGain(H, w, inA);
                }
            }
        }
    }

    int bestSum = -1;
    int bestIndex = -1;
    for(int i=0; i<(int)partialSums.size(); i++){
        if(partialSums[i] > bestSum){
            bestSum = partialSums[i];
            bestIndex = i;
        }
    }

    if(bestSum <= 0){
        for(auto v : movedVertices){
            inA[v] = !inA[v];
        }
        return 0;
    } else {
        for(int i=bestIndex+1; i<(int)movedVertices.size(); i++){
            int v = movedVertices[i];
            inA[v] = !inA[v];
        }
        return bestSum;
    }
}

int main(){
    Hypergraph H = readHypergraph();
    int n = H.n;

    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 0);
    random_shuffle(nodes.begin(), nodes.end());

    vector<bool> inA(n, false);
    for(int i=0; i<n/2; i++){
        inA[nodes[i]] = true;
    }

    int curCut = hyperedgeCutCost(H, inA);
    cerr << "Initial cut: " << curCut << "\n";

    int maxIters = 10;
    for(int iter = 0; iter < maxIters; iter++){
        int improvement = fm_pass(H, inA);
        if(improvement <= 0) {
            break;
        }
        int newCut = hyperedgeCutCost(H, inA);
        if(newCut < curCut){
            curCut = newCut;
            cerr << "Iteration " << iter << ", cut => " << curCut << "\n";
        } else {
            break;
        }
    }

    cout << "Final cut cost: " << curCut << "\n";
    cout << "Partition:\n";
    for(int v=0; v<n; v++){
        cout << v << " " << (inA[v] ? "A" : "B") << "\n";
    }
    return 0;
}
