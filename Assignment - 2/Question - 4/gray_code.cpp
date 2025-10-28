#include <bits/stdc++.h>
using namespace std;

static inline uint64_t toGrayCode(uint64_t x)
{
    return x ^ (x >> 1);
}

struct RowData {
    int rowID;
    int nnz;
    vector<int> cols;
    uint64_t bitmask;
    uint64_t grayval;
};

static const int DENSE_THRESHOLD = 20;

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, c;
    cin >> n >> c;

    vector<RowData> rows(n);

    for(int i=0; i<n; i++){
        RowData &rd = rows[i];
        rd.rowID = i;
        int rnnz;
        cin >> rnnz;
        rd.nnz = rnnz;
        rd.cols.resize(rnnz);
        for(int j=0; j<rnnz; j++){
            cin >> rd.cols[j];
        }
        sort(rd.cols.begin(), rd.cols.end());

        uint64_t mask = 0ULL;
        for(auto colIndex : rd.cols){
            int bitpos = (c <= 64) ? colIndex : (colIndex % 64);
            if(bitpos < 64 && bitpos >= 0){
                mask |= (1ULL << bitpos);
            }
        }
        rd.bitmask = mask;
        rd.grayval = toGrayCode(mask);
    }

    vector<RowData> sparseRows, denseRows;
    for(int i=0; i<n; i++){
        if(rows[i].nnz >= DENSE_THRESHOLD){
            denseRows.push_back(rows[i]);
        } else {
            sparseRows.push_back(rows[i]);
        }
    }

    sort(sparseRows.begin(), sparseRows.end(),
        [](const RowData &a, const RowData &b){
            return a.grayval < b.grayval || (a.grayval == b.grayval && a.rowID < b.rowID);
        }
    );

    sort(denseRows.begin(), denseRows.end(),
        [](const RowData &a, const RowData &b){
            if(a.nnz == 0 && b.nnz == 0) {
                return a.rowID < b.rowID;
            } else if(a.nnz == 0) {
                return true;
            } else if(b.nnz == 0) {
                return false;
            } else {
                return a.cols[0] < b.cols[0];
            }
        }
    );

    vector<int> finalOrder;
    finalOrder.reserve(n);
    for(auto &r : sparseRows){
        finalOrder.push_back(r.rowID);
    }
    for(auto &r : denseRows){
        finalOrder.push_back(r.rowID);
    }

    cout << "Gray code ordering (threshold=" << DENSE_THRESHOLD << "):\n";
    for(auto &row : finalOrder){
        cout << row << " ";
    }
    cout << "\n";

    return 0;
}






