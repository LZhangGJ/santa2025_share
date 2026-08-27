// miner.cpp - Kaggle Santa 2025 Sub-structure Miner
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp miner.cpp -o miner

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::map;

// ================= Geometry Kernel (Same as Squeezer) =================
static constexpr double SCALE = 1e18; 
static constexpr int NV = 15;
static constexpr double PI = 3.1415926535897932384626433832795;

alignas(64) static const double TX0[NV] = {
    0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, 
    -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
};
alignas(64) static const double TY0[NV] = {
    0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, 
    -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5
};

alignas(64) static double TX[NV], TY[NV];

struct Bounds { double minx, miny, maxx, maxy; };
struct Poly { double px[NV], py[NV]; Bounds b; };

static inline void update_poly(double cx, double cy, double deg, Poly& out) {
    double rad = deg * (PI / 180.0);
    double s = std::sin(rad), c = std::cos(rad);
    out.b = {1e99, 1e99, -1e99, -1e99};
    for (int i = 0; i < NV; i++) {
        out.px[i] = TX[i] * c - TY[i] * s + cx;
        out.py[i] = TX[i] * s + TY[i] * c + cy;
        if (out.px[i] < out.b.minx) out.b.minx = out.px[i];
        if (out.px[i] > out.b.maxx) out.b.maxx = out.px[i];
        if (out.py[i] < out.b.miny) out.b.miny = out.py[i];
        if (out.py[i] > out.b.maxy) out.b.maxy = out.py[i];
    }
}

// Helper to calculate side length of a list of polys
static inline double calc_side(const vector<Poly>& polys) {
    if (polys.empty()) return 0.0;
    double min_x = 1e99, max_x = -1e99, min_y = 1e99, max_y = -1e99;
    for (const auto& p : polys) {
        if (p.b.minx < min_x) min_x = p.b.minx;
        if (p.b.maxx > max_x) max_x = p.b.maxx;
        if (p.b.miny < min_y) min_y = p.b.miny;
        if (p.b.maxy > max_y) max_y = p.b.maxy;
    }
    return std::max(max_x - min_x, max_y - min_y);
}

// ================= IO Utils =================
struct TreeRow { string item_id; double x, y, deg; };
double parse_s_val(string s) {
    size_t idx = s.find_first_of("-0123456789");
    if (idx == string::npos) return 0.0;
    return std::stod(s.substr(idx));
}

int main(int argc, char** argv) {
    // Init Geometry
    for(int i=0; i<NV; ++i) { TX[i] = TX0[i] * SCALE; TY[i] = TY0[i] * SCALE; }

    string in_path = "submission.csv";
    string out_path = "submission_mined.csv";
    
    if (argc > 1) in_path = argv[1];
    if (argc > 2) out_path = argv[2];

    // Load CSV
    map<int, vector<TreeRow>> groups; // Key is int N now
    {
        std::ifstream file(in_path);
        if(!file.is_open()) { cerr << "Error opening " << in_path << endl; return 1; }
        string line; std::getline(file, line);
        while(std::getline(file, line)) {
            if(line.empty()) continue;
            for(char &c : line) if(c==',') c=' ';
            std::stringstream ss(line);
            string full_id, sx, sy, sdeg; ss >> full_id >> sx >> sy >> sdeg;
            size_t us = full_id.find('_');
            int gid = std::stoi(full_id.substr(0, us));
            string iid = full_id.substr(us+1);
            groups[gid].push_back({iid, parse_s_val(sx)*SCALE, parse_s_val(sy)*SCALE, parse_s_val(sdeg)});
        }
    }

    int improvements = 0;
    
    // --- THE MINING LOOP ---
    // Iterate downwards from Max N to 2
    // Because a better N could imply a better N-1, and so on.
    
    int max_n = 0;
    for(auto const& [k, v] : groups) max_n = std::max(max_n, k);

    cout << ">>> Starting Miner (Reverse Pruning) <<<" << endl;
    cout << "Scanning from N=" << max_n << " down to 2..." << endl;

    for (int n = max_n; n > 1; --n) {
        if (groups.find(n) == groups.end()) continue;
        
        // Current solution for N
        const auto& current_rows = groups[n];
        
        // Target: N-1
        int target_n = n - 1;
        
        // Calculate current score of N-1 (if exists)
        double best_prev_side = 1e99;
        if (groups.find(target_n) != groups.end()) {
            vector<Poly> prev_polys;
            for(const auto& r : groups[target_n]) {
                Poly p; update_poly(r.x, r.y, r.deg, p);
                prev_polys.push_back(p);
            }
            best_prev_side = calc_side(prev_polys);
        }

        // Try deleting each tree from N
        int best_delete_idx = -1;
        double found_side = 1e99;

        // Pre-compute polys for N to speed up
        vector<Poly> current_polys(n);
        for(int i=0; i<n; ++i) update_poly(current_rows[i].x, current_rows[i].y, current_rows[i].deg, current_polys[i]);

        for (int i = 0; i < n; ++i) {
            // Construct subset without tree i
            // We don't need to copy vectors, just calc bounds skipping i
            double min_x = 1e99, max_x = -1e99, min_y = 1e99, max_y = -1e99;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& b = current_polys[j].b;
                if (b.minx < min_x) min_x = b.minx;
                if (b.maxx > max_x) max_x = b.maxx;
                if (b.miny < min_y) min_y = b.miny;
                if (b.maxy > max_y) max_y = b.maxy;
            }
            double side = std::max(max_x - min_x, max_y - min_y);

            // Check improvement
            // Strictly less than current best for N-1
            if (side < best_prev_side - 1e-9) { // 1e-9 tolerance
                // Found a candidate! 
                // We want the BEST candidate among all deletions
                if (side < found_side) {
                    found_side = side;
                    best_delete_idx = i;
                }
            }
        }

        // If we found a better subset
        if (best_delete_idx != -1) {
            cout << "[MINE] BINGO! Found better N=" << target_n 
                 << " inside N=" << n 
                 << ". Side: " << (best_prev_side/SCALE) << " -> " << (found_side/SCALE) 
                 << " (Imp: " << ((best_prev_side - found_side)/SCALE) << ")" << endl;
            
            // Construct new rows
            vector<TreeRow> new_target_rows;
            for (int i = 0; i < n; ++i) {
                if (i == best_delete_idx) continue;
                new_target_rows.push_back(current_rows[i]);
            }
            // Re-assign item_ids (0 to N-2)
            for (int i = 0; i < new_target_rows.size(); ++i) {
                new_target_rows[i].item_id = std::to_string(i);
            }
            
            // Update the map immediately so the next iteration (N-1 -> N-2) uses this new best!
            groups[target_n] = new_target_rows;
            improvements++;
        }
    }

    // Save CSV
    std::ofstream outfile(out_path);
    outfile << "id,x,y,deg" << endl << std::fixed << std::setprecision(18);
    
    for (int n = 1; n <= max_n; ++n) {
        if (groups.find(n) == groups.end()) continue;
        const auto& rows = groups[n];
        string gid = "";
        if (n < 10) gid = "00" + std::to_string(n);
        else if (n < 100) gid = "0" + std::to_string(n);
        else gid = std::to_string(n);

        for (const auto& r : rows) {
             outfile << gid << "_" << r.item_id << ",s" << (r.x / SCALE) << ",s" << (r.y / SCALE) << ",s" << r.deg << endl;
        }
    }

    cout << "Mining Complete. Total Cascading Improvements: " << improvements << endl;
    return 0;
}
