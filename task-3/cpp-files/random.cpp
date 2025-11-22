#include <bits/stdc++.h>
#include <chrono>
#include <random> // For std::shuffle and std::mt19937
#include <limits> // For LLONG_MAX

using namespace std;
#ifndef LLONG_MAX
#define LLONG_MAX 9223372036854775807LL
#endif

// --- Global Data and Utility Functions ---

// Precomputed population count (number of set bits)
uint8_t POPCOUNT[256];

/**
 * @brief Initializes the POPCOUNT array for fast calculation of the number of set bits.
 */
void init_popcount() {
    for (int i = 0; i < 256; i++)
        POPCOUNT[i] = __builtin_popcount(i);
}

/**
 * @brief Computes the dot product modulo 2 of two 8-bit masks.
 * @param a The first 8-bit mask.
 * @param b The second 8-bit mask.
 * @return 1 if the dot product is odd, 0 if even.
 */
inline uint8_t dot_mask(uint8_t a, uint8_t b) {
    return POPCOUNT[a & b] & 1;
}

// --- WHS Metric Calculation ---

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

    /**
     * @brief Calculates the WHS (Walsh-Hadamard Spectrum) metric for a given S-box.
     * @param sbox The 256-element S-box.
     * @return The WHS metric value.
     */
    long double get(const vector<uint8_t> &sbox) {
        // These can be declared static inside the function to avoid re-allocation
        // if the structure member `get` is called many times on the same WHSMetric instance.
        static uint8_t comp[256][256]; // Component value: dot_mask(i, sbox[x])
        static uint8_t lin[256][256];  // Linear value: dot_mask(b, x)

        // Precompute component values
        for (int i = 0; i < 256; i++)
            for (int x = 0; x < 256; x++)
                comp[i][x] = dot_mask(all_masks[i], sbox[x]);
        
        // Precompute linear values
        for (int b = 0; b < 256; b++)
            for (int x = 0; x < 256; x++)
                lin[b][x] = dot_mask(all_masks[b], all_x[x]);

        long double total = 0.0L;

        // Iterate over all input mask 'i' and output mask 'b'
        for (int i = 0; i < 256; i++) {
            for (int b = 0; b < 256; b++) {

                int sum = 0; // The Walsh-Hadamard coefficient W(i, b)
                for (int x = 0; x < 256; x++) {
                    // W(i, b) = sum_x (-1)^(dot(i, sbox[x]) ^ dot(b, x))
                    uint8_t bit = comp[i][x] ^ lin[b][x];
                    sum += (bit ? -1 : +1);
                }

                long double val = fabsl((long double)sum);
                long double w = 1.0;
                // Raise the absolute value of the coefficient to the power R
                for (int k = 0; k < R; k++) w *= val;

                total += w;
            }
        }
        return total;
    }
};

// --- Non-linearity Calculation (using Fast Walsh-Hadamard Transform) ---

/**
 * @brief Performs the Fast Walsh-Hadamard Transform (FWHT).
 * @param a The input vector (truth table).
 * @return The Walsh-Hadamard spectrum.
 */
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

/**
 * @brief Generates the truth table for a component function (y = dot(mask, sbox[x])).
 * @param sbox The S-box.
 * @param mask The output mask for the component.
 * @return The truth table vector (+1/-1 representation).
 */
vector<int> get_component_truth(const vector<uint8_t> &sbox, int mask) {
    vector<int> truth(256);
    for (int x = 0; x < 256; x++) {
        uint8_t y = sbox[x];
        int bit = POPCOUNT[y & mask] & 1;
        truth[x] = (bit ? -1 : +1);
    }
    return truth;
}

/**
 * @brief Calculates the non-linearity (NL) of the S-box.
 * NL = 128 - max_abs_Walsh_coefficient / 2
 * @param sbox The S-box.
 * @return The minimum non-linearity across all non-zero components.
 */
int calculate_nonlinearity(const vector<uint8_t> &sbox) {
    int min_nl = 256;

    // Iterate through all 255 non-zero component masks
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

// --- Random Search Algorithm (Updated for Time/Iteration limits) ---

/**
 * @brief Executes the random search algorithm, limited by either iterations or time.
 * @param whs_metric The WHS metric structure (not strictly needed inside but kept for signature consistency).
 * @param max_iterations The maximum number of iterations to run (set to LLONG_MAX if time limit is used).
 * @param time_limit_seconds The time limit in seconds (0.0 for iteration-only limit).
 * @param target_nl The target non-linearity to stop the search.
 * @param run_seed Seed value for the random number generator.
 * @return A tuple containing: (best S-box found, iterations used, non-linearity progress)
 */
tuple<vector<uint8_t>, long long, vector<int>> random_algorithm(
    WHSMetric &whs_metric,
    long long max_iterations,
    double time_limit_seconds,
    int target_nl,
    unsigned int run_seed
) {
    vector<int> best_nl_each_iter;
    int best_so_far = 0;
    vector<uint8_t> best_arr(256);

    auto start_time = std::chrono::high_resolution_clock::now();
    bool use_time_limit = time_limit_seconds > 0.0;
    auto time_limit = std::chrono::duration<double>(time_limit_seconds);

    // Initialize random number generator for shuffling using the run_seed
    std::mt19937 g(run_seed);

    for (long long iter = 0; iter < max_iterations; iter++) {
        vector<uint8_t> arr(256);
        // Create an array with values 0 to 255
        iota(arr.begin(), arr.end(), 0);
        // Shuffle the array to get a random permutation (S-box)
        std::shuffle(arr.begin(), arr.end(), g);
        
        int nl = calculate_nonlinearity(arr);

        if (nl >= best_so_far) {
            best_so_far = nl;
            best_arr = arr;
        }
        // Save the best NL found so far at this iteration
        best_nl_each_iter.push_back(best_so_far);

        // Check for target non-linearity
        if (nl >= target_nl) {
            cout << "\nReached NL >= " << target_nl << " at Iteration " << iter + 1 << endl;
            // Return the S-box that met the target NL
            return {arr, iter + 1, best_nl_each_iter};
        }

        // Check time limit periodically (e.g., every 1000 iterations for performance)
        // Skip checking if no time limit is set.
        if (use_time_limit && (iter % 1000 == 0 || iter == 0)) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            if (elapsed >= time_limit) {
                cout << "\nTime limit reached (" << fixed << setprecision(2) << time_limit_seconds << "s) at Iteration " << iter + 1 << endl;
                // Return the best S-box found so far when time runs out
                return {best_arr, iter + 1, best_nl_each_iter};
            }
        }
    }
    // Return the best S-box found after max_iterations
    return {best_arr, max_iterations, best_nl_each_iter};
}

// --- Main Program (Updated to run multiple trials) ---

int main(int argc, char** argv) {
    
    init_popcount();

    // Check for the new argument count: <outdir> <num_runs> <mode> <value>
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <output_directory> <num_runs> <mode> <value>\n";
        cerr << "  <num_runs>: The total number of independent trials to perform.\n";
        cerr << "  <mode>: 'iter' (for iteration limit) or 'time' (for time limit in seconds).\n";
        cerr << "  <value>: Number of iterations (if 'iter') or time in seconds (if 'time').\n";
        return 1;
    }

    string outdir = argv[1];
    int num_runs;
    string mode = argv[3];
    string value_str = argv[4];

    long long inner_max_iterations = 0;
    double inner_time_limit_seconds = 0.0;
    
    cout << "------------------------------------------\n";

    // 1. Parse and validate num_runs (Outer Loop Count)
    try {
        num_runs = stoi(argv[2]);
        if (num_runs <= 0) {
            cerr << "Error: Number of runs must be positive.\n";
            return 1;
        }
    } catch (const std::invalid_argument& e) {
        cerr << "Error: Invalid number of runs format.\n";
        return 1;
    } catch (const std::out_of_range& e) {
        cerr << "Error: Number of runs is too large.\n";
        return 1;
    }


    // 2. Parse and validate mode and value (Inner Loop Limit)
    if (mode == "iter") {
        try {
            inner_max_iterations = stoll(value_str);
            if (inner_max_iterations <= 0) {
                cerr << "Error: Inner iteration limit must be positive.\n";
                return 1;
            }
        } catch (const std::invalid_argument& e) {
            cerr << "Error: Invalid iteration value format.\n";
            return 1;
        } catch (const std::out_of_range& e) {
            cerr << "Error: Iteration value is too large or too small.\n";
            return 1;
        }
        // Set time limit to 0.0 to disable time checking
        inner_time_limit_seconds = 0.0;
        cout << "Mode: Iteration-limited (Max " << inner_max_iterations << " iterations per run)\n";
    } else if (mode == "time") {
        try {
            inner_time_limit_seconds = stod(value_str);
            if (inner_time_limit_seconds <= 0.0) {
                cerr << "Error: Time limit must be positive.\n";
                return 1;
            }
        } catch (const std::invalid_argument& e) {
            cerr << "Error: Invalid time value format.\n";
            return 1;
        } catch (const std::out_of_range& e) {
            cerr << "Error: Time value is out of range.\n";
            return 1;
        }
        // Set max_iterations to the maximum possible value to make time checking the limit
        inner_max_iterations = LLONG_MAX; 
        cout << "Mode: Time-limited (Max " << fixed << setprecision(2) << inner_time_limit_seconds << " seconds per run)\n";
    } else {
        cerr << "Error: Invalid mode '" << mode << "'. Use 'iter' or 'time'.\n";
        return 1;
    }

    if (outdir.back() != '/' && outdir.back() != '\\')
        outdir += "/";

    WHSMetric whs_metric(12);
    
    cout << "Starting " << num_runs << " independent runs of Random Search...\n";
    cout << "------------------------------------------\n";

    // --- Outer Loop for Independent Trials ---
    for (int run = 1; run <= num_runs; run++) {
        
        // Use a unique seed for each run for independence, combining current time and run number
        unsigned int run_seed = (unsigned int)time(NULL) + run;
        
        cout << "\nRunning Trial " << run << " of " << num_runs << "...\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Call the random_algorithm with the determined limits and unique seed
        auto [best, iterations_used, progress] =
            random_algorithm(whs_metric, inner_max_iterations, inner_time_limit_seconds, 104, run_seed);

        auto end_time = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> duration = end_time - start_time;
        double total_time_seconds = duration.count();
        int nl = calculate_nonlinearity(best);
        string run_suffix = "_" + to_string(run);

        // --- Output Saving ---
        {
            ofstream fout(outdir + "result" + run_suffix + ".txt");
            fout << "Run: " << run << "\n";
            fout << "Mode used: " << mode << "\n";
            if (mode == "iter") fout << "Target Iterations: " << inner_max_iterations << "\n";
            if (mode == "time") fout << "Time Limit (s): " << fixed << setprecision(2) << inner_time_limit_seconds << "\n";
            fout << "Best nonlinearity: " << nl << "\n";
            fout << "Iterations used: " << iterations_used << "\n";
            fout << "Total time taken (s): " << fixed << setprecision(4) << total_time_seconds << "\n";
            fout << "S-box values:\n";
            for (int i = 0; i < 256; i++) {
                fout << (int)best[i];
                if (i != 255) fout << ", ";
            }
            fout << "\n";
        }

        {
            ofstream prog(outdir + "best_nonlinearity_progress" + run_suffix + ".txt");
            for (int val : progress)
                prog << val << "\n";
        }

        {
            // Note: Time taken is already included in result file, but kept for consistency
            ofstream time_file(outdir + "time_taken" + run_suffix + ".txt");
            time_file << "Run: " << run << "\n";
            time_file << "Total calculation time: " << fixed << setprecision(4) << total_time_seconds << " seconds\n";
        }

        cout << "\nTrial " << run << " finished.\n";
        cout << "Final NL = " << nl << endl;
        cout << "Iterations = " << iterations_used << endl;
        cout << "Time taken: " << fixed << setprecision(4) << total_time_seconds << " seconds" << endl;
        cout << "Results for run " << run << " saved to files with suffix " << run_suffix << "\n";
        cout << "------------------------------------------\n";
    }

    cout << "\n\u2705 All " << num_runs << " trials completed.\n";
    cout << "All results saved in directory: " << outdir << endl;

    return 0;
}