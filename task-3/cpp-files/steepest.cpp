#include <bits/stdc++.h>
#include <chrono>
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

struct Individual {
    vector<uint8_t> sbox;
    int nl;
    long double whs;
};

tuple<vector<uint8_t>, long long, vector<int>> steepest_algorithm(
    WHSMetric &whs_metric,
    int mode,                     // 0 = stop when stuck, 1 = time limit+resets, 2 = iteration limit+resets
    long long iter_limit,         // used only when mode=2
    double time_limit,            // used only when mode=1
    int target_nl                 // stop when reaching target
) {
    vector<int> best_nl_each_iter;

    vector<uint8_t> curr_arr(256);
    iota(curr_arr.begin(), curr_arr.end(), 0);
    random_shuffle(curr_arr.begin(), curr_arr.end());

    int curr_nl = calculate_nonlinearity(curr_arr);
    long double curr_whs = whs_metric.get(curr_arr);
    long double curr_swap_whs = curr_whs;

    vector<uint8_t> best_arr = curr_arr;
    int best_nl = curr_nl;

    long long num_iters = 0;

    bool use_time_limit = (mode == 1);
    auto start_time = chrono::high_resolution_clock::now();
    chrono::duration<double> time_limit_d(time_limit);

    while (true) {

        bool improved = false;
        int curr_swap_nl = curr_nl;
        vector<uint8_t> curr_swap_arr = curr_arr;
        curr_swap_whs = curr_whs;

        for (int i = 0; i < 256; i++) {
            for (int j = i + 1; j < 256; j++) {

                num_iters++;
                best_nl_each_iter.push_back(best_nl);

                // ----- MODE C: check iteration limit -----
                if (mode == 2 && num_iters >= iter_limit)
                    return {best_arr, num_iters, best_nl_each_iter};

                // ----- MODE A: check time limit -----
                if (mode == 1 && num_iters % 300 == 0) {
                    auto now = chrono::high_resolution_clock::now();
                    if (now - start_time >= time_limit_d)
                        return {best_arr, num_iters, best_nl_each_iter};
                }

                // try swap
                vector<uint8_t> new_arr = curr_arr;
                swap(new_arr[i], new_arr[j]);

                int new_nl = calculate_nonlinearity(new_arr);
                if (new_nl > curr_swap_nl) {
                    curr_swap_arr = new_arr;
                    curr_swap_nl = new_nl;
                    curr_swap_whs = whs_metric.get(new_arr);
                    improved = true;

                    if (curr_swap_nl > best_nl) {
                        best_nl = curr_swap_nl;
                        best_arr = curr_swap_arr;
                    }

                    if (best_nl >= target_nl)
                        return {best_arr, num_iters, best_nl_each_iter};
                }
            }
        }

        if (improved) {
            curr_arr = curr_swap_arr;
            curr_nl = curr_swap_nl;
            curr_whs = curr_swap_whs;
        }

        // ---- NO IMPROVEMENT FOUND ----
        if (!improved) {

            if (mode == 0) {
                // Mode B → stop when stuck
                return {best_arr, num_iters, best_nl_each_iter};
            }

            if (mode == 1 || mode == 2) {
                // Mode A or C → reset to random permutation
                random_shuffle(curr_arr.begin(), curr_arr.end());
                curr_nl = calculate_nonlinearity(curr_arr);
                curr_whs = whs_metric.get(curr_arr);
            }
        }
    }

    return {best_arr, num_iters, best_nl_each_iter};
}


int main(int argc, char** argv) {
    
    init_popcount();

    // Required:
    //   <outdir> <num_runs> <mode> <value>
    //
    // mode:
    //   stuck           → stop when stuck
    //   time <seconds>  → time limit per run, resets when stuck
    //   iter <iters>    → iteration limit per run, resets when stuck
    //
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <output_directory> <num_runs> <mode> <value>\n";
        cerr << "  Modes:\n";
        cerr << "    stuck                Stop immediately when stuck (no reset)\n";
        cerr << "    time <seconds>       Use time limit per run (with resets)\n";
        cerr << "    iter <iterations>    Use iteration limit per run (with resets)\n";
        return 1;
    }

    string outdir = argv[1];
    int num_runs;
    string mode_str = argv[3];
    string value_str = argv[4];

    int mode = 0;                 // 0 = stuck, 1 = time, 2 = iter
    double time_limit = 0.0;      // used only when mode=1
    long long iter_limit = 0;     // used only when mode=2

    // Parse num_runs
    try {
        num_runs = stoi(argv[2]);
        if (num_runs <= 0) {
            cerr << "Error: Number of runs must be positive.\n";
            return 1;
        }
    } catch (...) {
        cerr << "Error: Invalid number of runs.\n";
        return 1;
    }

    // Parse mode and value
    if (mode_str == "stuck") {
        mode = 0;
    }
    else if (mode_str == "time") {
        mode = 1;
        try {
            time_limit = stod(value_str);
            if (time_limit <= 0.0) {
                cerr << "Error: Time limit must be positive.\n";
                return 1;
            }
        } catch (...) {
            cerr << "Error: Invalid time limit.\n";
            return 1;
        }
    }
    else if (mode_str == "iter") {
        mode = 2;
        try {
            iter_limit = stoll(value_str);
            if (iter_limit <= 0) {
                cerr << "Error: Iteration limit must be positive.\n";
                return 1;
            }
        } catch (...) {
            cerr << "Error: Invalid iteration limit.\n";
            return 1;
        }
    }
    else {
        cerr << "Error: Unknown mode. Use 'stuck', 'time', or 'iter'.\n";
        return 1;
    }

    if (outdir.back() != '/' && outdir.back() != '\\')
        outdir += "/";

    WHSMetric whs_metric(12);

    cout << "Starting " << num_runs << " independent runs of Greedy Algorithm...\n";
    cout << "Mode: " << mode_str << "\n";
    if (mode == 1) cout << "Time limit = " << time_limit << " seconds\n";
    if (mode == 2) cout << "Iteration limit = " << iter_limit << "\n";
    cout << "------------------------------------------\n";

    for (int run = 1; run <= num_runs; run++) {

        srand(time(NULL) + run);

        cout << "\nRunning Iteration " << run << " of " << num_runs << "...\n";
        
        auto start_time = chrono::high_resolution_clock::now();

        auto [best, iterations_used, progress] =
            steepest_algorithm(whs_metric, mode, iter_limit, time_limit, 104);

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_time - start_time;

        double total_time_seconds = duration.count();
        int nl = calculate_nonlinearity(best);

        string run_suffix = "_" + to_string(run);

        // Save results
        {
            ofstream fout(outdir + "result" + run_suffix + ".txt");
            fout << "Run: " << run << "\n";
            fout << "Mode: " << mode_str << "\n";
            if (mode == 1) fout << "Time limit: " << time_limit << " seconds\n";
            if (mode == 2) fout << "Iteration limit: " << iter_limit << "\n";
            fout << "Best nonlinearity: " << nl << "\n";
            fout << "Iterations used: " << iterations_used << "\n";
            fout << "Total time: " << fixed << setprecision(4) 
                 << total_time_seconds << " seconds\n";
            fout << "S-box values:\n";
            for (int i = 0; i < 256; i++) {
                fout << (int)best[i];
                if (i != 255) fout << ", ";
            }
            fout << "\n";
        }

        {
            ofstream prog(outdir + "best_nonlinearity_progress" + run_suffix + ".txt");
            for (int v : progress)
                prog << v << "\n";
        }

        {
            ofstream tfile(outdir + "time_taken" + run_suffix + ".txt");
            tfile << "Run: " << run << "\n";
            tfile << "Total calculation time: " << fixed << setprecision(4)
                  << total_time_seconds << " seconds\n";
        }

        cout << "\nIteration " << run << " finished.\n";
        cout << "Final NL = " << nl << "\n";
        cout << "Iterations = " << iterations_used << "\n";
        cout << "Time taken = " << fixed << setprecision(4) 
             << total_time_seconds << " seconds\n";
        cout << "Results saved with suffix " << run_suffix << "\n";
        cout << "------------------------------------------\n";
    }

    cout << "\nAll " << num_runs << " runs completed.\n";
    cout << "Results saved to directory: " << outdir << endl;

    return 0;
}
