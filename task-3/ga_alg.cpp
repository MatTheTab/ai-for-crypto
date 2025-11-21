#include <bits/stdc++.h>
using namespace std;

uint8_t POPCOUNT[256];

void init_popcount() {
    for (int i = 0; i < 256; i++)
        POPCOUNT[i] = __builtin_popcount(i);
}

inline uint8_t dot_mask(uint8_t a, uint8_t b) {
    return POPCOUNT[a & b] & 1;
}

struct WHSMetric {
    int R;
    vector<uint8_t> all_masks, all_x;

    WHSMetric(int R_=12) : R(R_) {
        all_masks.resize(256);
        all_x.resize(256);
        for (int i = 0; i < 256; i++) {
            all_masks[i] = uint8_t(i);
            all_x[i] = uint8_t(i);
        }
    }

    long double get(const vector<uint8_t> &sbox) {
        static uint8_t comp[256][256];
        static uint8_t lin[256][256];

        for (int i = 0; i < 256; i++)
            for (int x = 0; x < 256; x++)
                comp[i][x] = dot_mask(all_masks[i], sbox[x]);
        
        for (int b = 0; b < 256; b++)
            for (int x = 0; x < 256; x++)
                lin[b][x] = dot_mask(all_masks[b], all_x[x]);

        long double total = 0.0L;

        for (int i = 0; i < 256; i++) {
            for (int b = 0; b < 256; b++) {

                int sum = 0;
                for (int x = 0; x < 256; x++) {
                    uint8_t bit = comp[i][x] ^ lin[b][x];
                    sum += (bit ? -1 : +1);
                }

                long double val = fabsl((long double)sum);
                long double w = 1.0;
                for (int k = 0; k < R; k++) w *= val;

                total += w;
            }
        }
        return total;
    }
};

vector<int> fwht(vector<int> a) {
    int n = a.size();
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h*2) {
            for (int j = i; j < i + h; j++) {
                int x = a[j], y = a[j+h];
                a[j] = x + y;
                a[j+h] = x - y;
            }
        }
    }
    return a;
}

vector<int> get_component_truth(const vector<uint8_t> &sbox, int mask) {
    vector<int> truth(256);
    for (int x = 0; x < 256; x++) {
        uint8_t y = sbox[x];
        int bit = POPCOUNT[y & mask] & 1;
        truth[x] = (bit ? -1 : +1);
    }
    return truth;
}

int calculate_nonlinearity(const vector<uint8_t> &sbox) {
    int min_nl = 256;

    for (int mask = 1; mask < 256; mask++) {
        auto f = get_component_truth(sbox, mask);
        auto spec = fwht(f);

        int max_abs = 0;
        for (int v : spec) max_abs = max(max_abs, abs(v));

        int nl = 128 - (max_abs / 2);
        min_nl = min(min_nl, nl);
    }
    return min_nl;
}

vector<uint8_t> mutate_swap(const vector<uint8_t> &arr) {
    vector<uint8_t> S(arr);
    int i = rand() % 256, j = rand() % 256;
    while (j == i) j = rand() % 256;
    swap(S[i], S[j]);
    return S;
}

struct Individual {
    vector<uint8_t> sbox;
    int nl;
    long double whs;
};

bool cmpElite(const Individual &a, const Individual &b) {
    if (a.nl != b.nl) return a.nl > b.nl;
    return a.whs < b.whs;
}

vector<Individual> elite_selection(vector<Individual> pop, int K_pop) {
    sort(pop.begin(), pop.end(), cmpElite);
    if ((int)pop.size() > K_pop)
        pop.resize(K_pop);
    return pop;
}

tuple<vector<uint8_t>, int, vector<int>> modified_genetic_algorithm(
    WHSMetric &whs_metric,
    int K_iter = 150000,
    int K_pop = 1,
    int K_mut = 7,
    int target_nl = 104
) {
    vector<Individual> S_pop;
    vector<int> best_nl_each_iter;

    // Init population
    for (int p = 0; p < K_pop; p++) {
        vector<uint8_t> arr(256);
        iota(arr.begin(), arr.end(), 0);
        random_shuffle(arr.begin(), arr.end());
        int nl = calculate_nonlinearity(arr);
        long double whs = whs_metric.get(arr);
        S_pop.push_back({arr, nl, whs});
    }

    // GA loop
    for (int iter = 0; iter < K_iter; iter++) {

        if (iter % 1000 == 0) {
            double pct = 100.0 * iter / K_iter;
            cout << "Iter " << iter << "/" << K_iter
                 << " (" << fixed << setprecision(2) << pct << "%)\r";
            cout.flush();
        }

        S_pop = elite_selection(S_pop, K_pop);

        // track best NL this generation
        best_nl_each_iter.push_back(S_pop[0].nl);

        vector<Individual> new_items;

        for (auto &ind : S_pop) {
            for (int m = 0; m < K_mut; m++) {
                auto S_prime = mutate_swap(ind.sbox);
                int nl_prime = calculate_nonlinearity(S_prime);
                long double whs_prime = whs_metric.get(S_prime);

                if (nl_prime >= target_nl) {
                    cout << "\nReached NL ≥ " << target_nl << endl;
                    return {S_prime, iter, best_nl_each_iter};
                }

                new_items.push_back({S_prime, nl_prime, whs_prime});
            }
        }
        S_pop.insert(S_pop.end(), new_items.begin(), new_items.end());
    }

    sort(S_pop.begin(), S_pop.end(), cmpElite);
    return {S_pop[0].sbox, K_iter, best_nl_each_iter};
}

int main(int argc, char** argv) {
    srand(time(NULL));
    init_popcount();

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <output_directory>\n";
        return 1;
    }

    string outdir = argv[1];

    if (outdir.back() != '/' && outdir.back() != '\\')
        outdir += "/";

    WHSMetric whs_metric(12);

    cout << "Running GA...\n";

    auto [best, iterations_used, progress] =
        modified_genetic_algorithm(whs_metric, 1500, 1, 7, 104);

    int nl = calculate_nonlinearity(best);

    {
        ofstream fout(outdir + "result.txt");
        fout << "Best nonlinearity: " << nl << "\n";
        fout << "Iterations used: " << iterations_used << "\n";
        fout << "S-box values:\n";
        for (int i = 0; i < 256; i++) {
            fout << (int)best[i];
            if (i != 255) fout << ", ";
        }
    }

    {
        ofstream prog(outdir + "best_nonlinearity_progress.txt");
        for (int val : progress)
            prog << val << "\n";
    }

    cout << "\nDone.\n";
    cout << "Final NL = " << nl << endl;
    cout << "Iterations = " << iterations_used << endl;
    cout << "Results saved in directory: " << outdir << endl;

    return 0;
}