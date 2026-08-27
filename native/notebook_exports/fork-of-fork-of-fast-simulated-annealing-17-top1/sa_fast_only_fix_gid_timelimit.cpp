// sa_fast_only_fix_gid_timelimit.cpp
// ONLY optimize groups specified by -only, match numeric gid (58 == 058)
// Add time limit: -sec <seconds>  (e.g. 3600 = 1 hour)
//
// Build:
//   g++ -Ofast -march=native -std=c++17 -fopenmp sa_fast_only_fix_gid_timelimit.cpp -o sa_fast_only
//
// Run:
//   ./sa_fast_only -i in.csv -o out.csv -only "58,65,57,151,134" -sec 3600 -tstart 0.8 -tend 0.001 -seed 42 -threads 4
//
// Note: -iter is still supported as a hard cap; runtime stop happens first.

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
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cerr;
using std::cout;
using std::endl;

static constexpr int NV = 15;
static constexpr double PI = 3.1415926535897932384626433832795;
static constexpr double SCALE = 1e18;

alignas(64) static const double TX0[NV] = {
    0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
};
alignas(64) static const double TY0[NV] = {
    0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5
};

alignas(64) static double TX[NV], TY[NV];

struct Bounds { double minx, miny, maxx, maxy; };
struct Poly { double px[NV], py[NV]; Bounds b; };

static inline Bounds bounds_from_poly(const Poly& p) {
    Bounds b;
    b.minx = b.maxx = p.px[0];
    b.miny = b.maxy = p.py[0];
    for (int i = 1; i < NV; i++) {
        b.minx = std::min(b.minx, p.px[i]);
        b.miny = std::min(b.miny, p.py[i]);
        b.maxx = std::max(b.maxx, p.px[i]);
        b.maxy = std::max(b.maxy, p.py[i]);
    }
    return b;
}

static inline void build_poly(double cx, double cy, double ang_deg, Poly& out) {
    long double a = (long double)ang_deg * PI / 180.0L;
    long double c = std::cos(a), s = std::sin(a);
    for (int i = 0; i < NV; i++) {
        long double x = (long double)TX[i];
        long double y = (long double)TY[i];
        long double rx = c * x - s * y;
        long double ry = s * x + c * y;
        out.px[i] = (double)(rx + (long double)cx);
        out.py[i] = (double)(ry + (long double)cy);
    }
    out.b = bounds_from_poly(out);
}

static inline double parse_s_double(const std::string& s) {
    if (!s.empty() && (s[0] == 's' || s[0] == 'S')) return std::stod(s.substr(1));
    return std::stod(s);
}
static inline void split_id(const std::string& id, std::string& gid, std::string& item) {
    auto pos = id.find('_');
    if (pos == std::string::npos) { gid = id; item = ""; return; }
    gid = id.substr(0, pos);
    item = id.substr(pos + 1);
}

struct TreeRow { std::string item_id; double x, y, deg; };
using GroupMap = std::map<std::string, std::vector<TreeRow>>;

static GroupMap load_csv_rows(const std::string& fn) {
    std::ifstream f(fn);
    if (!f) throw std::runtime_error("Could not open input CSV: " + fn);

    std::string header;
    std::getline(f, header);

    GroupMap groups;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find(',');
        size_t p2 = (p1 == std::string::npos) ? std::string::npos : line.find(',', p1 + 1);
        size_t p3 = (p2 == std::string::npos) ? std::string::npos : line.find(',', p2 + 1);
        if (p1 == std::string::npos || p2 == std::string::npos || p3 == std::string::npos) continue;

        std::string id = line.substr(0, p1);
        std::string xs = line.substr(p1 + 1, p2 - (p1 + 1));
        std::string ys = line.substr(p2 + 1, p3 - (p2 + 1));
        std::string ds = line.substr(p3 + 1);

        std::string gid, item;
        split_id(id, gid, item);

        TreeRow r;
        r.item_id = item;
        r.x = parse_s_double(xs);
        r.y = parse_s_double(ys);
        r.deg = parse_s_double(ds);

        groups[gid].push_back(r);
    }
    return groups;
}

static void save_csv_rows(const std::string& fn, const GroupMap& groups) {
    std::ofstream f(fn);
    if (!f) throw std::runtime_error("Could not open output CSV: " + fn);

    f << "id,x,y,deg\n";
    f << std::setprecision(20);

    std::vector<std::string> keys;
    keys.reserve(groups.size());
    for (auto& kv : groups) keys.push_back(kv.first);

    std::sort(keys.begin(), keys.end(), [](const std::string& a, const std::string& b) {
        long long ia = 0, ib = 0;
        try { ia = std::stoll(a); } catch (...) {}
        try { ib = std::stoll(b); } catch (...) {}
        return ia < ib;
    });

    for (const auto& gid : keys) {
        const auto& v = groups.at(gid);
        for (size_t i = 0; i < v.size(); i++) {
            const auto& r = v[i];
            std::string item = (!r.item_id.empty()) ? r.item_id : std::to_string(i);
            f << gid << "_" << item
              << ",s" << r.x
              << ",s" << r.y
              << ",s" << r.deg
              << "\n";
        }
    }
}

struct SAParams {
    long long max_iter = 3500000LL;
    double t_start = 0.8;
    double t_end = 0.001;
    uint64_t base_seed = 42;
    double time_limit_sec = -1.0; // <0 means no time limit
};

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
struct FastRNG {
    uint64_t s;
    explicit FastRNG(uint64_t seed) : s(seed) {}
    inline uint64_t next_u64() { s = splitmix64(s); return s; }
    inline double rf01() { return (next_u64() >> 11) * (1.0 / 9007199254740992.0); }
    inline int ri(int n) { return (int)(next_u64() % (uint64_t)n); }
};

enum class SegHit : uint8_t { NONE = 0, TOUCH = 1, PROPER = 2 };

static inline long double orient(long double ax, long double ay, long double bx, long double by,
                                 long double cx, long double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}
static inline bool on_segment(long double ax, long double ay, long double bx, long double by,
                              long double px, long double py) {
    return std::min(ax, bx) <= px && px <= std::max(ax, bx) &&
           std::min(ay, by) <= py && py <= std::max(ay, by);
}
static inline SegHit seg_intersect_type(long double ax, long double ay, long double bx, long double by,
                                        long double cx, long double cy, long double dx, long double dy) {
    long double o1 = orient(ax, ay, bx, by, cx, cy);
    long double o2 = orient(ax, ay, bx, by, dx, dy);
    long double o3 = orient(cx, cy, dx, dy, ax, ay);
    long double o4 = orient(cx, cy, dx, dy, bx, by);

    auto sgn = [](long double v) -> int { return (v > 0) - (v < 0); };
    int a1 = sgn(o1), a2 = sgn(o2), a3 = sgn(o3), a4 = sgn(o4);

    if (a1 * a2 < 0 && a3 * a4 < 0) return SegHit::PROPER;

    if (a1 == 0 && on_segment(ax, ay, bx, by, cx, cy)) return SegHit::TOUCH;
    if (a2 == 0 && on_segment(ax, ay, bx, by, dx, dy)) return SegHit::TOUCH;
    if (a3 == 0 && on_segment(cx, cy, dx, dy, ax, ay)) return SegHit::TOUCH;
    if (a4 == 0 && on_segment(cx, cy, dx, dy, bx, by)) return SegHit::TOUCH;

    return SegHit::NONE;
}

static inline bool pip_strict(long double px, long double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        long double xi = q.px[i], yi = q.py[i];
        long double xj = q.px[j], yj = q.py[j];
        bool cond = ((yi > py) != (yj > py));
        if (cond) {
            long double xint = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if (px < xint) in = !in;
        }
        j = i;
    }
    return in;
}

static inline bool overlap_disallow_touch(const Poly& a, const Poly& b) {
    if (a.b.maxx < b.b.minx || a.b.minx > b.b.maxx ||
        a.b.maxy < b.b.miny || a.b.miny > b.b.maxy)
        return false;

    for (int i = 0; i < NV; i++) {
        if (pip_strict(a.px[i], a.py[i], b)) return true;
        if (pip_strict(b.px[i], b.py[i], a)) return true;
    }

    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        long double ax = a.px[i], ay = a.py[i];
        long double bx = a.px[ni], by = a.py[ni];
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            long double cx = b.px[j], cy = b.py[j];
            long double dx = b.px[nj], dy = b.py[nj];
            SegHit hit = seg_intersect_type(ax, ay, bx, by, cx, cy, dx, dy);
            if (hit == SegHit::PROPER) return true;
        }
    }
    return false;
}

static inline bool validate_no_overlaps(const std::vector<Poly>& polys) {
    for (size_t i = 0; i < polys.size(); i++) {
        for (size_t j = i + 1; j < polys.size(); j++) {
            if (overlap_disallow_touch(polys[i], polys[j])) return false;
        }
    }
    return true;
}

struct StateItem { double cx_scaled, cy_scaled; double angle_deg; Poly poly; Bounds b; };

static inline Bounds envelope_from_bounds(const std::vector<Bounds>& bs) {
    Bounds e{bs[0].minx, bs[0].miny, bs[0].maxx, bs[0].maxy};
    for (size_t i = 1; i < bs.size(); i++) {
        e.minx = std::min(e.minx, bs[i].minx);
        e.miny = std::min(e.miny, bs[i].miny);
        e.maxx = std::max(e.maxx, bs[i].maxx);
        e.maxy = std::max(e.maxy, bs[i].maxy);
    }
    return e;
}
static inline Bounds envelope_from_bounds_replace(const std::vector<Bounds>& bs, size_t idx, const Bounds& nb) {
    Bounds e;
    bool first = true;
    for (size_t i = 0; i < bs.size(); i++) {
        const Bounds& b = (i == idx ? nb : bs[i]);
        if (first) { e = b; first = false; }
        else {
            e.minx = std::min(e.minx, b.minx);
            e.miny = std::min(e.miny, b.miny);
            e.maxx = std::max(e.maxx, b.maxx);
            e.maxy = std::max(e.maxy, b.maxy);
        }
    }
    return e;
}
static inline double side_len_from_env(const Bounds& e) {
    double w = (e.maxx - e.minx) / SCALE;
    double h = (e.maxy - e.miny) / SCALE;
    return std::max(w, h);
}
static inline double side_len_fast(const std::vector<StateItem>& st) {
    Bounds e{st[0].b.minx, st[0].b.miny, st[0].b.maxx, st[0].b.maxy};
    for (size_t i = 1; i < st.size(); i++) {
        e.minx = std::min(e.minx, st[i].b.minx);
        e.miny = std::min(e.miny, st[i].b.miny);
        e.maxx = std::max(e.maxx, st[i].b.maxx);
        e.maxy = std::max(e.maxy, st[i].b.maxy);
    }
    return side_len_from_env(e);
}

static std::pair<std::vector<TreeRow>, double> optimize_group(
    const std::string& group_id,
    const std::vector<TreeRow>& rows,
    const SAParams& P
) {
    const int n = (int)rows.size();
    if (n == 0) return {rows, 0.0};

    uint64_t gid_int = 0;
    try { gid_int = (uint64_t)std::stoll(group_id); } catch (...) { gid_int = 0; }
    uint64_t task_seed = splitmix64((P.base_seed ^ (gid_int * 0x9E3779B97F4A7C15ULL)));
    FastRNG rng(task_seed);

    bool is_small_n = (n <= 50);
    long long effective_max_iter = is_small_n ? (P.max_iter * 3LL) : P.max_iter;
    double effective_t_start = is_small_n ? (P.t_start * 2.0) : P.t_start;
    double gravity_weight = is_small_n ? 1e-4 : 1e-6;

    std::vector<StateItem> state(n);
    std::vector<Bounds> bounds_list(n);

    for (int i = 0; i < n; i++) {
        state[i].cx_scaled = rows[i].x * SCALE;
        state[i].cy_scaled = rows[i].y * SCALE;
        state[i].angle_deg = rows[i].deg;
        build_poly(state[i].cx_scaled, state[i].cy_scaled, state[i].angle_deg, state[i].poly);
        state[i].b = state[i].poly.b;
        bounds_list[i] = state[i].b;
    }

    Bounds env = envelope_from_bounds(bounds_list);

    double dist_sum = 0.0;
    for (int i = 0; i < n; i++) {
        dist_sum += state[i].cx_scaled * state[i].cx_scaled + state[i].cy_scaled * state[i].cy_scaled;
    }

    const double inv_scale = 1.0 / SCALE;
    const double inv_scale2 = 1.0 / (SCALE * SCALE);

    auto energy_from = [&](const Bounds& e, double dist) -> std::pair<double, double> {
        double side = std::max(e.maxx - e.minx, e.maxy - e.miny) * inv_scale;
        double norm_dist = (dist * inv_scale2) / std::max(1, n);
        double energy = side + gravity_weight * norm_dist;
        return {energy, side};
    };

    auto cur_pair = energy_from(env, dist_sum);
    double cur_energy = cur_pair.first;
    double cur_side = cur_pair.second;

    struct BestParam { double cx_scaled, cy_scaled, angle_deg; };
    std::vector<BestParam> best_params(n);
    for (int i = 0; i < n; i++) best_params[i] = {state[i].cx_scaled, state[i].cy_scaled, state[i].angle_deg};
    double best_real_score = cur_side;

    auto t0 = std::chrono::steady_clock::now();
    auto time_ok = [&]() -> bool {
        if (P.time_limit_sec <= 0) return true;
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return sec < P.time_limit_sec;
    };

    double T = effective_t_start;
    double cooling_rate = std::pow(P.t_end / effective_t_start, 1.0 / (double)effective_max_iter);

    for (long long it = 0; it < effective_max_iter; it++) {
        if (!time_ok()) break;

        double progress = (double)it / (double)effective_max_iter;

        double move_scale, rotate_scale;
        if (is_small_n) {
            move_scale = std::max(0.005, 3.0 * (1.0 - progress));
            rotate_scale = std::max(0.001, 5.0 * (1.0 - progress));
        } else {
            move_scale = std::max(0.001, 1.0 * (T / effective_t_start));
            rotate_scale = std::max(0.002, 5.0 * (T / effective_t_start));
        }

        int idx = rng.ri(n);
        auto& tgt = state[idx];

        Bounds orig_bounds = bounds_list[idx];
        double orig_cx = tgt.cx_scaled, orig_cy = tgt.cy_scaled, orig_ang = tgt.angle_deg;

        double dx = (rng.rf01() - 0.5) * SCALE * 0.1 * move_scale;
        double dy = (rng.rf01() - 0.5) * SCALE * 0.1 * move_scale;
        double dA = (rng.rf01() - 0.5) * rotate_scale;

        double new_cx = orig_cx + dx;
        double new_cy = orig_cy + dy;
        double new_ang = orig_ang + dA;

        Poly cand_poly;
        build_poly(new_cx, new_cy, new_ang, cand_poly);
        Bounds new_bounds = cand_poly.b;

        bool collision = false;
        for (int k = 0; k < n; k++) {
            if (k == idx) continue;
            const Bounds& ob = bounds_list[k];
            if (new_bounds.maxx < ob.minx || new_bounds.minx > ob.maxx ||
                new_bounds.maxy < ob.miny || new_bounds.miny > ob.maxy)
                continue;
            if (overlap_disallow_touch(cand_poly, state[k].poly)) {
                collision = true;
                break;
            }
        }
        if (collision) { T *= cooling_rate; continue; }

        double old_d = orig_cx * orig_cx + orig_cy * orig_cy;
        double new_d = new_cx * new_cx + new_cy * new_cy;
        double cand_dist_sum = dist_sum - old_d + new_d;

        bool need_recompute =
            (orig_bounds.minx == env.minx && new_bounds.minx > env.minx) ||
            (orig_bounds.miny == env.miny && new_bounds.miny > env.miny) ||
            (orig_bounds.maxx == env.maxx && new_bounds.maxx < env.maxx) ||
            (orig_bounds.maxy == env.maxy && new_bounds.maxy < env.maxy);

        Bounds cand_env;
        if (need_recompute) cand_env = envelope_from_bounds_replace(bounds_list, (size_t)idx, new_bounds);
        else {
            cand_env = {
                std::min(env.minx, new_bounds.minx),
                std::min(env.miny, new_bounds.miny),
                std::max(env.maxx, new_bounds.maxx),
                std::max(env.maxy, new_bounds.maxy)
            };
        }

        auto cand_pair = energy_from(cand_env, cand_dist_sum);
        double new_energy = cand_pair.first;
        double new_real = cand_pair.second;

        double delta = new_energy - cur_energy;

        bool accept = false;
        if (delta < 0) accept = true;
        else if (T > 1e-10) {
            double prob = std::exp(-delta * 1000.0 / T);
            accept = (rng.rf01() < prob);
        }

        if (accept) {
            tgt.poly = cand_poly;
            tgt.b = new_bounds;
            bounds_list[idx] = new_bounds;
            tgt.cx_scaled = new_cx;
            tgt.cy_scaled = new_cy;
            tgt.angle_deg = new_ang;

            env = cand_env;
            dist_sum = cand_dist_sum;
            cur_energy = new_energy;

            if (new_real < best_real_score) {
                best_real_score = new_real;
                for (int k = 0; k < n; k++) {
                    best_params[k] = {state[k].cx_scaled, state[k].cy_scaled, state[k].angle_deg};
                }
            }
        }
        T *= cooling_rate;
    }

    std::vector<TreeRow> out = rows;
    std::vector<Poly> final_polys(n);

    for (int i = 0; i < n; i++) {
        out[i].x = best_params[i].cx_scaled / SCALE;
        out[i].y = best_params[i].cy_scaled / SCALE;
        out[i].deg = best_params[i].angle_deg;
        build_poly(best_params[i].cx_scaled, best_params[i].cy_scaled, best_params[i].angle_deg, final_polys[i]);
    }

    if (!validate_no_overlaps(final_polys)) {
        std::vector<StateItem> orig_state(n);
        for (int i = 0; i < n; i++) {
            orig_state[i].cx_scaled = rows[i].x * SCALE;
            orig_state[i].cy_scaled = rows[i].y * SCALE;
            orig_state[i].angle_deg = rows[i].deg;
            build_poly(orig_state[i].cx_scaled, orig_state[i].cy_scaled, orig_state[i].angle_deg, orig_state[i].poly);
            orig_state[i].b = orig_state[i].poly.b;
        }
        double orig_side = side_len_fast(orig_state);
        return {rows, orig_side};
    }

    return {out, best_real_score};
}

static inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

static std::vector<long long> parse_only_list_numeric(const std::string& spec) {
    std::vector<long long> out;
    if (spec.empty()) return out;

    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (tok.empty()) continue;

        auto dash = tok.find('-');
        if (dash != std::string::npos) {
            std::string a = trim(tok.substr(0, dash));
            std::string b = trim(tok.substr(dash + 1));
            long long lo = 0, hi = 0;
            try { lo = std::stoll(a); hi = std::stoll(b); } catch (...) { continue; }
            if (hi < lo) std::swap(lo, hi);
            for (long long v = lo; v <= hi; v++) out.push_back(v);
        } else {
            try { out.push_back(std::stoll(tok)); } catch (...) {}
        }
    }

    std::unordered_set<long long> seen;
    std::vector<long long> uniq;
    uniq.reserve(out.size());
    for (auto v : out) if (seen.insert(v).second) uniq.push_back(v);
    return uniq;
}

int main(int argc, char** argv) {
    for (int i = 0; i < NV; i++) { TX[i] = TX0[i] * SCALE; TY[i] = TY0[i] * SCALE; }

    std::string in = "submission1.csv";
    std::string out = "submission2.csv";
    std::string only_spec = "";
    SAParams P;
    int threads = 16;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-only" && i + 1 < argc) only_spec = argv[++i];
        else if (a == "-iter" && i + 1 < argc) P.max_iter = std::stoll(argv[++i]);
        else if (a == "-tstart" && i + 1 < argc) P.t_start = std::stod(argv[++i]);
        else if (a == "-tend" && i + 1 < argc) P.t_end = std::stod(argv[++i]);
        else if (a == "-seed" && i + 1 < argc) P.base_seed = (uint64_t)std::stoull(argv[++i]);
        else if (a == "-threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (a == "-sec" && i + 1 < argc) P.time_limit_sec = std::stod(argv[++i]);
    }

#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    auto only_nums = parse_only_list_numeric(only_spec);
    if (only_nums.empty()) {
        cerr << "ERROR: -only list is empty. Example: -only \"58,65,57,151,134\" or -only \"21-40,41-60\"\n";
        return 2;
    }

    GroupMap groups;
    try {
        cout << "Loading csv: " << in << "\n";
        groups = load_csv_rows(in);
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::unordered_map<long long, std::string> num2gid;
    num2gid.reserve(groups.size() * 2);
    for (const auto& kv : groups) {
        long long gid_num = -1;
        try { gid_num = std::stoll(kv.first); } catch (...) { continue; }
        if (num2gid.find(gid_num) == num2gid.end()) num2gid[gid_num] = kv.first;
    }

    std::vector<std::string> to_opt;
    to_opt.reserve(only_nums.size());
    for (auto g : only_nums) {
        auto it = num2gid.find(g);
        if (it != num2gid.end()) to_opt.push_back(it->second);
        else cerr << "WARN: group " << g << " not found in CSV (skip)\n";
    }
    if (to_opt.empty()) {
        cerr << "ERROR: none of requested groups exist in CSV.\n";
        return 3;
    }

    cout << "Optimize ONLY these groups (count=" << to_opt.size() << "): ";
    for (size_t i = 0; i < to_opt.size(); i++) cout << to_opt[i] << (i + 1 < to_opt.size() ? "," : "");
    cout << "\n";

    cout << "Threads=" << threads
         << "  Seed=" << P.base_seed
         << "  MAX_ITER=" << P.max_iter
         << "  T_START=" << P.t_start
         << "  T_END=" << P.t_end;
    if (P.time_limit_sec > 0) cout << "  TIME_LIMIT=" << P.time_limit_sec << "s";
    cout << "\n";

    std::mutex io_mu;
    std::atomic<int> improved{0};
    std::atomic<int> finished{0};
    const int total = (int)to_opt.size();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; i++) {
        const std::string& gid = to_opt[i];
        const auto& rows = groups[gid];

        std::vector<StateItem> st(rows.size());
        for (size_t k = 0; k < rows.size(); k++) {
            st[k].cx_scaled = rows[k].x * SCALE;
            st[k].cy_scaled = rows[k].y * SCALE;
            st[k].angle_deg = rows[k].deg;
            build_poly(st[k].cx_scaled, st[k].cy_scaled, st[k].angle_deg, st[k].poly);
            st[k].b = st[k].poly.b;
        }
        double orig_side = side_len_fast(st);

        auto [new_rows, best_side] = optimize_group(gid, rows, P);
        bool imp = (best_side < orig_side - 1e-12);

        if (imp) {
#pragma omp critical
            { groups[gid] = std::move(new_rows); }
            improved.fetch_add(1);
        }

        int done = finished.fetch_add(1) + 1;
        {
            std::lock_guard<std::mutex> lk(io_mu);
            double diff = orig_side - best_side;
            double pct = (orig_side != 0.0) ? (diff / orig_side * 100.0) : 0.0;

            cout << "[" << done << "/" << total << "] G:" << gid
                 << " " << std::fixed << std::setprecision(6)
                 << orig_side << " -> " << best_side;

            if (imp) cout << "  improved: -" << std::setprecision(6) << diff << " (" << std::setprecision(3) << pct << "%)";
            else cout << "  (no improve)";
            cout << "\n";
        }
    }

    cout << "\nFinal Save. Improved groups: " << improved.load() << "/" << total << "\n";
    try {
        save_csv_rows(out, groups);
        cout << "Saved to " << out << "\n";
    } catch (const std::exception& e) {
        cerr << "Final save failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
