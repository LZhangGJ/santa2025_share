// squeezer.cpp - Kaggle Santa 2025 V2: Cyclic Boundary Attack (Fixed & Optimized)
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp squeezer.cpp -o squeezer

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <set>
#include <tuple>   // 关键修复
#include <utility> // 关键修复

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map;

static constexpr double SCALE = 1e18; 
static constexpr long double PI = 3.1415926535897932384626433832795L;
static constexpr int NV = 15;

alignas(64) static const double TX0[NV] = {
    0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, 
    -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
};
alignas(64) static const double TY0[NV] = {
    0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, 
    -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5
};

alignas(64) static double TX[NV], TY[NV];

struct Bounds { 
    double minx, miny, maxx, maxy; 
    inline bool intersects(const Bounds& o) const {
        return (minx <= o.maxx && maxx >= o.minx && miny <= o.maxy && maxy >= o.miny);
    }
};

struct Poly { double px[NV], py[NV]; Bounds b; };

static inline long double cross_product(long double ax, long double ay, long double bx, long double by, long double cx, long double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

static inline bool on_segment(long double ax, long double ay, long double bx, long double by, long double px, long double py) {
    return std::min(ax, bx) <= px && px <= std::max(ax, bx) && std::min(ay, by) <= py && py <= std::max(ay, by);
}

static inline bool segments_intersect_strict(long double ax, long double ay, long double bx, long double by, long double cx, long double cy, long double dx, long double dy) {
    long double d1 = cross_product(cx, cy, dx, dy, ax, ay);
    long double d2 = cross_product(cx, cy, dx, dy, bx, by);
    long double d3 = cross_product(ax, ay, bx, by, cx, cy);
    long double d4 = cross_product(ax, ay, bx, by, dx, dy);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true; 
    return false;
}

static inline bool is_point_strictly_inside(double px, double py, const Poly& poly) {
    if (px < poly.b.minx || px > poly.b.maxx || py < poly.b.miny || py > poly.b.maxy) return false;
    for (int i = 0; i < NV; i++) {
        int j = (i + 1) % NV;
        long double cp = cross_product(poly.px[i], poly.py[i], poly.px[j], poly.py[j], px, py);
        if (std::abs(cp) < 1e-9 && on_segment(poly.px[i], poly.py[i], poly.px[j], poly.py[j], px, py)) return false;
    }
    bool inside = false;
    for (int i = 0, j = NV - 1; i < NV; j = i++) {
        if (((poly.py[i] > py) != (poly.py[j] > py)) &&
            (px < (poly.px[j] - poly.px[i]) * (py - poly.py[i]) / (poly.py[j] - poly.py[i]) + poly.px[i])) {
            inside = !inside;
        }
    }
    return inside;
}

static inline bool check_overlap_strict(const Poly& A, const Poly& B) {
    if (!A.b.intersects(B.b)) return false;
    for (int i = 0; i < NV; i++) {
        if (is_point_strictly_inside(A.px[i], A.py[i], B)) return true;
        if (is_point_strictly_inside(B.px[i], B.py[i], A)) return true;
    }
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segments_intersect_strict(A.px[i], A.py[i], A.px[ni], A.py[ni], B.px[j], B.py[j], B.px[nj], B.py[nj])) return true;
        }
    }
    return false;
}

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

struct TreeRow { string item_id; double x, y, deg; };

struct FastRNG {
    uint64_t state;
    FastRNG(uint64_t seed) : state(seed | 1) {}
    inline uint64_t next() {
        uint64_t x = state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return state = x;
    }
    inline double next_double() { return (next() & 0xFFFFFFFFFFFFF) * (1.0 / 4503599627370496.0); }
    inline int next_int(int n) { return next() % n; }
};

double parse_s_val(string s) {
    size_t idx = s.find_first_of("-0123456789");
    if (idx == string::npos) return 0.0;
    return std::stod(s.substr(idx));
}

struct State { double x, y, deg; Poly p; };

vector<TreeRow> squeeze_group(string gid, vector<TreeRow> initial_rows, 
                              long long total_iter, double t_base, 
                              int cycles, int seed) {
    int n = initial_rows.size();
    if (n == 0) return initial_rows;

    vector<State> st(n);
    FastRNG rng(seed + std::hash<string>{}(gid));

    for(int i=0; i<n; ++i) {
        st[i].x = initial_rows[i].x; st[i].y = initial_rows[i].y; st[i].deg = initial_rows[i].deg;
        update_poly(st[i].x, st[i].y, st[i].deg, st[i].p);
    }

    auto calculate_energy_detailed = [&](const vector<State>& s) {
        double min_x = 1e99, max_x = -1e99, min_y = 1e99, max_y = -1e99;
        double dist_sq_sum = 0;
        vector<int> boundary_indices;
        
        for(int i=0; i<n; ++i) {
            const auto& t = s[i];
            if(t.p.b.minx < min_x) min_x = t.p.b.minx; 
            if(t.p.b.maxx > max_x) max_x = t.p.b.maxx;
            if(t.p.b.miny < min_y) min_y = t.p.b.miny; 
            if(t.p.b.maxy > max_y) max_y = t.p.b.maxy;
            dist_sq_sum += (t.x*t.x + t.y*t.y);
        }
        
        double eps = 1e-5 * SCALE;
        for(int i=0; i<n; ++i) {
            const auto& t = s[i];
            if (std::abs(t.p.b.minx - min_x) < eps || std::abs(t.p.b.maxx - max_x) < eps ||
                std::abs(t.p.b.miny - min_y) < eps || std::abs(t.p.b.maxy - max_y) < eps) {
                boundary_indices.push_back(i);
            }
        }

        double side = std::max(max_x - min_x, max_y - min_y);
        double score = (side * side) + (dist_sq_sum * 1e-6); 
        return std::make_tuple(score, boundary_indices);
    };

    auto [current_energy, boundary_trees] = calculate_energy_detailed(st);
    double best_energy = current_energy;
    vector<State> best_st = st;

    long long iter_per_cycle = total_iter / cycles;

    for (int c = 0; c < cycles; ++c) {
        double cycle_factor = 1.0 - ((double)c / cycles); 
        double T = t_base * cycle_factor;
        if (c > 0) T = std::max(T, 1e-5); 
        
        double t_end = 1e-7;
        double decay = std::pow(t_end / T, 1.0 / iter_per_cycle);
        double step_mv = 0.05 * SCALE; 
        double step_rot = 0.5;

        for (long long k = 0; k < iter_per_cycle; ++k) {
            int idx;
            bool force_inward = false;

            if (boundary_trees.size() > 0 && rng.next() % 4 == 0) {
                idx = boundary_trees[rng.next_int(boundary_trees.size())];
                force_inward = true;
            } else {
                idx = rng.next_int(n);
            }

            State backup = st[idx];
            double ratio = T / t_base;
            double cur_mv = step_mv * ratio + (step_mv * 0.01); 
            double cur_rot = step_rot * ratio + (step_rot * 0.01);

            if (force_inward) {
                double dx = -st[idx].x;
                double dy = -st[idx].y;
                double len = std::sqrt(dx*dx + dy*dy);
                if (len > 1e-9) {
                    dx /= len; dy /= len;
                    st[idx].x += (dx + (rng.next_double()-0.5)*0.5) * cur_mv;
                    st[idx].y += (dy + (rng.next_double()-0.5)*0.5) * cur_mv;
                    st[idx].deg += (rng.next_double() - 0.5) * cur_rot;
                }
            } else {
                if (rng.next() % 2 == 0) {
                    st[idx].x += (rng.next_double() - 0.5) * 2.0 * cur_mv;
                    st[idx].y += (rng.next_double() - 0.5) * 2.0 * cur_mv;
                } else {
                    st[idx].deg += (rng.next_double() - 0.5) * 2.0 * cur_rot;
                }
            }

            update_poly(st[idx].x, st[idx].y, st[idx].deg, st[idx].p);

            bool collision = false;
            for (int j = 0; j < n; ++j) {
                if (idx == j) continue;
                if (check_overlap_strict(st[idx].p, st[j].p)) { collision = true; break; }
            }

            if (collision) {
                st[idx] = backup;
            } else {
                bool need_boundary_update = (k % 100 == 0);
                
                double min_x=1e99, max_x=-1e99, min_y=1e99, max_y=-1e99;
                double dist_sum = 0;
                for(const auto& t : st) {
                    if(t.p.b.minx<min_x) min_x=t.p.b.minx; if(t.p.b.maxx>max_x) max_x=t.p.b.maxx;
                    if(t.p.b.miny<min_y) min_y=t.p.b.miny; if(t.p.b.maxy>max_y) max_y=t.p.b.maxy;
                    dist_sum += (t.x*t.x + t.y*t.y);
                }
                double side = std::max(max_x - min_x, max_y - min_y);
                double new_energy = (side * side) + (dist_sum * 1e-6);

                double delta = new_energy - current_energy;

                if (delta < 0 || rng.next_double() < std::exp(-delta / (T * SCALE * SCALE))) {
                    current_energy = new_energy;
                    if (current_energy < best_energy) { 
                        best_energy = current_energy; 
                        best_st = st; 
                    }
                    if (need_boundary_update) {
                        auto res = calculate_energy_detailed(st);
                        boundary_trees = std::get<1>(res);
                    }
                } else {
                    st[idx] = backup;
                }
            }
            T *= decay;
        }
        st = best_st;
        auto res = calculate_energy_detailed(st);
        current_energy = std::get<0>(res);
        boundary_trees = std::get<1>(res);
    }

    vector<TreeRow> result;
    for(int i=0; i<n; ++i) result.push_back({initial_rows[i].item_id, best_st[i].x, best_st[i].y, best_st[i].deg});
    return result;
}

int main(int argc, char** argv) {
    for(int i=0; i<NV; ++i) { TX[i] = TX0[i] * SCALE; TY[i] = TY0[i] * SCALE; }
    
    string in_path = "submission.csv";
    string out_path = "submission_squeezed.csv";
    long long iter = 5000000;
    double t_start = 0.01; 
    int cycles = 5;
    int threads = 4;

    for(int i=1; i<argc; ++i) {
        string arg = argv[i];
        if(arg == "-i") in_path = argv[++i];
        if(arg == "-o") out_path = argv[++i];
        if(arg == "-iter") iter = std::stoll(argv[++i]);
        if(arg == "-tstart") t_start = std::stod(argv[++i]);
        if(arg == "-cycles") cycles = std::stoi(argv[++i]);
        if(arg == "-threads") threads = std::stoi(argv[++i]);
    }

    map<string, vector<TreeRow>> groups;
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
            string gid = full_id.substr(0, us);
            string iid = full_id.substr(us+1);
            groups[gid].push_back({iid, parse_s_val(sx)*SCALE, parse_s_val(sy)*SCALE, parse_s_val(sdeg)});
        }
    }

    vector<string> keys;
    for(auto const& [k,v] : groups) keys.push_back(k);
    std::sort(keys.begin(), keys.end(), [&](const string& a, const string& b){ return groups[a].size() > groups[b].size(); });

    std::atomic<int> improved_count = 0;
    std::atomic<int> processed_count = 0;
    
    cout << ">>> Santa 2025 Squeezer V2 <<<" << endl;
    cout << "Threads: " << threads << " | Cycles: " << cycles << " | Iter/Group: " << iter << endl;

    #ifdef _OPENMP
    omp_set_num_threads(threads);
    #endif

    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<keys.size(); ++i) {
        string gid = keys[i];
        auto rows = groups[gid];
        auto new_rows = squeeze_group(gid, rows, iter, t_start, cycles, 42 + i);
        if(new_rows[0].x != rows[0].x) {
            #pragma omp critical 
            { groups[gid] = new_rows; improved_count++; cout << "[SQZ] Improved " << gid << endl; }
        }
        int p = ++processed_count;
        if(p % 20 == 0) { 
            #pragma omp critical 
            cout << "Progress: " << p << "/" << keys.size() << endl; 
        }
    }

    std::ofstream outfile(out_path);
    outfile << "id,x,y,deg" << endl << std::fixed << std::setprecision(18);
    std::sort(keys.begin(), keys.end(), [&](const string& a, const string& b){ return std::stoi(a) < std::stoi(b); });
    for(const string& gid : keys) {
        auto& rows = groups[gid];
        std::sort(rows.begin(), rows.end(), [&](const TreeRow& a, const TreeRow& b){ return std::stoi(a.item_id) < std::stoi(b.item_id); });
        for(const auto& r : rows) outfile << gid << "_" << r.item_id << ",s" << (r.x / SCALE) << ",s" << (r.y / SCALE) << ",s" << r.deg << endl;
    }
    cout << "Finished! Total Squeezed: " << improved_count << endl;
    return 0;
}
