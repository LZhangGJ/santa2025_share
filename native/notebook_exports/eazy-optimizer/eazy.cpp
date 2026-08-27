#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <iomanip>
#include <chrono>
#include <random>
#include <complex>
#include <omp.h>

using namespace std;

typedef complex<double> cd;

// --- CONSTANTS ---
constexpr int MAX_N = 205;
constexpr int NV = 25;
constexpr double PI = 3.14159265358979323846;

// fixed tree shape (15 vertices)
const double TX[NV] = {
    0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125
};
const double TY[NV] = {
    0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5
};

thread_local mt19937_64 rng(random_device{}());
inline double rf() { return uniform_real_distribution<double>(0, 1)(rng); }
inline int ri(int n) { return uniform_int_distribution<int>(0, n - 1)(rng); }

struct Pt { double x, y; };

struct Poly {
    Pt p[NV];
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = p[0].x; y0 = y1 = p[0].y;
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, p[i].x); x1 = max(x1, p[i].x);
            y0 = min(y0, p[i].y); y1 = max(y1, p[i].y);
        }
    }
};

// --- GEOMETRY ENGINE ---
static inline bool contains(const Poly& poly, Pt pt) {
    bool inside = false;
    for (int i = 0, j = NV - 1; i < NV; j = i++) {
        if (((poly.p[i].y > pt.y) != (poly.p[j].y > pt.y)) &&
            (pt.x < (poly.p[j].x - poly.p[i].x) * (pt.y - poly.p[i].y) / (poly.p[j].y - poly.p[i].y) + poly.p[i].x))
            inside = !inside;
    }
    return inside;
}

static inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 - 1e-13 || b.x1 < a.x0 - 1e-13 || a.y1 < b.y0 - 1e-13 || b.y1 < a.y0 - 1e-13) return false;

    auto ccw = [](Pt p, Pt q, Pt r) {
        long double v = (long double)(q.y - p.y) * (r.x - q.x) - (long double)(q.x - p.x) * (r.y - q.y);
        return (v > 1e-20L) ? 1 : (v < -1e-20L ? -1 : 0);
    };

    for (int i = 0; i < NV; i++) {
        for (int j = 0; j < NV; j++) {
            Pt p1 = a.p[i], q1 = a.p[(i+1)%NV], p2 = b.p[j], q2 = b.p[(j+1)%NV];
            if (ccw(p1, q1, p2) != ccw(p1, q1, q2) && ccw(p2, q2, p1) != ccw(p2, q2, q1))
                return true;
        }
    }
    return contains(a, b.p[0]) || contains(b, a.p[0]);
}

static inline Poly getPoly(double cx, double cy, double deg) {
    Poly q;
    double r = deg * PI / 180.0, c = cos(r), s = sin(r);
    for (int i = 0; i < NV; i++) {
        q.p[i].x = TX[i] * c - TY[i] * s + cx;
        q.p[i].y = TX[i] * s + TY[i] * c + cy;
    }
    q.bbox();
    return q;
}

struct Cfg {
    int n = 0;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];

    // cached bbox
    double xmin, xmax, ymin, ymax;

    // ✅ IMPORTANT: init arrays to avoid UB
    Cfg() {
        for (int i = 0; i < MAX_N; i++) {
            x[i] = 0.0;
            y[i] = 0.0;
            a[i] = 0.0;
        }
        xmin = ymin = 0.0;
        xmax = ymax = 0.0;
    }

    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }

    void update_bbox() {
        xmin = 1e18; xmax = -1e18; ymin = 1e18; ymax = -1e18;
        for (int i = 0; i < n; i++) {
            xmin = min(xmin, pl[i].x0); xmax = max(xmax, pl[i].x1);
            ymin = min(ymin, pl[i].y0); ymax = max(ymax, pl[i].y1);
        }
    }

    void updAll() {
        for (int i = 0; i < n; i++) upd(i);
        update_bbox();
    }

    inline double side() const { return max(xmax - xmin, ymax - ymin); }
    inline double cx() const { return 0.5 * (xmin + xmax); }
    inline double cy() const { return 0.5 * (ymin + ymax); }

    void recenter_to_origin() {
        double dx = cx();
        double dy = cy();
        if (fabs(dx) < 1e-15 && fabs(dy) < 1e-15) return;
        for (int i = 0; i < n; i++) {
            x[i] -= dx;
            y[i] -= dy;
            for (int k = 0; k < NV; k++) { pl[i].p[k].x -= dx; pl[i].p[k].y -= dy; }
            pl[i].x0 -= dx; pl[i].x1 -= dx;
            pl[i].y0 -= dy; pl[i].y1 -= dy;
        }
        xmin -= dx; xmax -= dx; ymin -= dy; ymax -= dy;
    }

    bool check_valid() {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return false;
        return true;
    }
};

// boundary-biased pick list
static inline vector<int> boundary_indices(const Cfg& cur, double eps = 1e-10) {
    vector<int> idx;
    idx.reserve(cur.n);
    for (int i = 0; i < cur.n; i++) {
        const Poly& p = cur.pl[i];
        if (fabs(p.x0 - cur.xmin) <= eps || fabs(p.x1 - cur.xmax) <= eps ||
            fabs(p.y0 - cur.ymin) <= eps || fabs(p.y1 - cur.ymax) <= eps) {
            idx.push_back(i);
        }
    }
    if (idx.empty()) {
        for (int i = 0; i < cur.n; i++) idx.push_back(i);
    }
    return idx;
}

// --- Square pressure (bbox-centered) ---
static inline void apply_square_pressure(Cfg& cur, int i, double target_side, double step_scale) {
    double cx = cur.cx();
    double cy = cur.cy();

    double x_mid = 0.5 * (cur.pl[i].x0 + cur.pl[i].x1) - cx;
    double y_mid = 0.5 * (cur.pl[i].y0 + cur.pl[i].y1) - cy;

    double L = target_side * 0.5;

    auto get_grad = [&](double pos) {
        double d1 = L - pos;
        double d2 = L + pos;
        if (d1 < 1e-9) d1 = 1e-9;
        if (d2 < 1e-9) d2 = 1e-9;
        return (1.0 / d1) - (1.0 / d2);
    };

    double gx = get_grad(x_mid);
    double gy = get_grad(y_mid);

    cur.x[i] -= gx * step_scale * 0.01;
    cur.y[i] -= gy * step_scale * 0.01;
}

static inline void rotate_around(double& px, double& py, double cx, double cy, double ang_rad) {
    double dx = px - cx, dy = py - cy;
    double c = cos(ang_rad), s = sin(ang_rad);
    double nx = dx * c - dy * s;
    double ny = dx * s + dy * c;
    px = cx + nx;
    py = cy + ny;
}

static inline void sample_pivot_inside_poly(const Cfg& cur, int i, double& px, double& py) {
    const Poly& p = cur.pl[i];
    for (int t = 0; t < 6; t++) {
        double x = p.x0 + rf() * (p.x1 - p.x0);
        double y = p.y0 + rf() * (p.y1 - p.y0);
        if (contains(p, Pt{x, y})) { px = x; py = y; return; }
    }
    px = 0.5 * (p.x0 + p.x1);
    py = 0.5 * (p.y0 + p.y1);
}

static inline void sample_pivot_in_bbox_fast(const Cfg& cur, int i, double& px, double& py) {
    const Poly& p = cur.pl[i];
    px = p.x0 + rf() * (p.x1 - p.x0);
    py = p.y0 + rf() * (p.y1 - p.y0);
}

// --- Random scaling mixer ---
static inline double pick_step_scale(double base_scale, int stagnation) {
    static const double pool[] = {
        1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 1e-9
    };
    constexpr int K = sizeof(pool)/sizeof(pool[0]);

    double p_big = 0.03;
    p_big = min(0.35, p_big + 0.01 * stagnation);

    double u = rf();
    if (u < p_big) {
        int j = ri(3);
        return pool[j];
    }

    int best = 0;
    double bestd = 1e100;
    for (int i = 0; i < K; i++) {
        double d = fabs(log(pool[i]) - log(base_scale));
        if (d < bestd) bestd = d, best = i;
    }
    int lo = max(0, best - 2), hi = min(K-1, best + 2);
    return pool[ uniform_int_distribution<int>(lo, hi)(rng) ];
}

// --- MASTER: SA + Greedy Hybrid ---
static inline double run_powerhouse_cycle_sa(Cfg& cur, int iter, double base_scale) {
    cur.update_bbox();
    cur.recenter_to_origin();

    double curr_s = cur.side();
    double best_s = curr_s;

    vector<int> bnd = boundary_indices(cur);
    int stagnation = 0;

    for (int it = 0; it < iter; it++) {
        if ((it % 200) == 0) {
            cur.update_bbox();
            bnd = boundary_indices(cur, 1e-10);
        }
        if ((it % 4000) == 0) {
            cur.update_bbox();
            cur.recenter_to_origin();
        }

        double step_scale = pick_step_scale(base_scale, stagnation);
        double T = step_scale * 5e-3;

        int i;
        if (!bnd.empty() && rf() < 0.90) i = bnd[ri((int)bnd.size())];
        else i = ri(cur.n);

        double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
        Poly op = cur.pl[i];

        double r_move = rf();

        if (r_move < 0.34) {
            cur.x[i] += (rf()-0.5) * step_scale;
            cur.y[i] += (rf()-0.5) * step_scale;

        } else if (r_move < 0.60) {
            double px, py;
            double mode = rf();

            if (mode < 0.70) {
                sample_pivot_inside_poly(cur, i, px, py);
            } else if (mode < 0.90) {
                sample_pivot_in_bbox_fast(cur, i, px, py);
            } else {
                cur.update_bbox();
                double cx = cur.cx(), cy = cur.cy();
                double R = 0.15;
                px = cx + (rf()-0.5) * 2.0 * R;
                py = cy + (rf()-0.5) * 2.0 * R;
            }

            double ang = (rf()-0.5) * step_scale * 0.35;
            rotate_around(cur.x[i], cur.y[i], px, py, ang);

        } else if (r_move < 0.90) {
            cur.a[i] += (rf()-0.5) * step_scale * 45.0;

        } else {
            cur.update_bbox();
            apply_square_pressure(cur, i, best_s, step_scale);
        }

        cur.upd(i);

        bool hit = false;
        for (int j = 0; j < cur.n; j++) {
            if (i == j) continue;
            if (overlap(cur.pl[i], cur.pl[j])) { hit = true; break; }
        }

        if (!hit) {
            cur.update_bbox();
            double new_s = cur.side();

            bool accept = false;
            if (new_s <= curr_s + 1e-15) {
                accept = true;
            } else {
                double delta = new_s - curr_s;
                double prob = exp(-delta / max(1e-18, T));
                if (rf() < prob) accept = true;
            }

            if (accept) {
                curr_s = new_s;
                if (curr_s < best_s - 1e-15) {
                    best_s = curr_s;
                    stagnation = 0;
                } else {
                    stagnation++;
                }
                continue;
            }
        }

        // rollback
        cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.pl[i]=op;
        cur.update_bbox();
        stagnation++;
    }

    cur.recenter_to_origin();
    return best_s;
}

// --- IO ---
static inline map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> res;
    ifstream f(fn);
    string ln, h;
    if(!f) return res;
    getline(f, h);

    auto pnum = [](const string& s) {
        size_t st = s.find_first_of("0123456789.-");
        return (st == string::npos) ? 0.0 : stod(s.substr(st));
    };

    while (getline(f, ln)) {
        if (ln.empty()) continue;
        stringstream ss(ln);
        string id, sx, sy, sa;
        if(!getline(ss, id, ',')) continue;
        if(!getline(ss, sx, ',')) continue;
        if(!getline(ss, sy, ',')) continue;
        if(!getline(ss, sa, ',')) continue;

        if (id.size() < 5) continue;           // e.g. "001_0"
        int n = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));

        if (n <= 0 || n >= MAX_N) continue;
        if (idx < 0 || idx >= n) continue;

        if (res[n].n == 0) res[n].n = n;

        res[n].x[idx] = pnum(sx);
        res[n].y[idx] = pnum(sy);
        res[n].a[idx] = pnum(sa);
    }

    for (auto& kv : res) {
        kv.second.updAll();
        kv.second.recenter_to_origin();
    }
    return res;
}

static inline void saveCSV(const string& fn, map<int, Cfg>& res) {
    ofstream f(fn);
    f << "id,x,y,deg\n";

    for (int n = 1; n <= 200; n++) {
        if (!res.count(n)) continue;
        res[n].update_bbox();
        res[n].recenter_to_origin();
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i
              << ",s" << fixed << setprecision(18) << res[n].x[i]
              << ",s" << res[n].y[i]
              << ",s" << res[n].a[i] << "\n";
        }
    }
}

// ---- time utils ----
static inline long long elapsed_sec(const chrono::steady_clock::time_point& t0) {
    return chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - t0).count();
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const string INPUT = "/kaggle/input/intergration-of-existing-result-current-best/submission.csv";
    auto res = loadCSV(INPUT);
    if(res.empty()) {
        cerr << "Failed to load: " << INPUT << "\n";
        return 1;
    }

    // ✅ only optimize these N
    unordered_set<int> focusN = {
        122, 46, 123, 94, 69, 135, 78, 121, 54, 136,
        188, 194, 193, 187, 79, 95, 190, 93, 113, 186
    };

    // keys only contain focusN
    vector<int> keys;
    keys.reserve(focusN.size());
    for (auto const& [n, g] : res) {
        if (focusN.count(n)) keys.push_back(n);
    }
    sort(keys.rbegin(), keys.rend());

    cout << "Focus keys = " << keys.size() << "\n";
    if (keys.empty()) {
        cerr << "No focus keys found in csv.\n";
        saveCSV("submission.csv", res);
        return 0;
    }

    // ✅ GLOBAL TIME LIMIT: 3 hours (hard stop)
    const long long GLOBAL_LIMIT_SEC = 3LL * 3600LL;

    // ✅ checkpoint save interval (seconds)
    const long long CHECKPOINT_EVERY_SEC = 300; // 5 min

    // Base scales cycle (coarse -> fine)
    vector<double> base_scales = {1e-3, 1e-5, 1e-7, 1e-9};

    // Iter chunk: we run many chunks until 3h ends
    // Increase this if you want more "SA depth per attempt"
    const int ITER_CHUNK = 350000; // chunk iterations per call (bigger = slower, deeper)

    // Fail tolerance before "switch base_scale / move on"
    const int FAIL_TOL = 15;

    auto global_start = chrono::steady_clock::now();
    auto last_save = global_start;

    // Init bbox for focused N
    for (int n : keys) {
        res[n].update_bbox();
        res[n].recenter_to_origin();
    }

    // Keep best side cache
    map<int, double> best_side;
    for (int n : keys) best_side[n] = res[n].side();

    cout << "Start global optimize, time_limit=" << GLOBAL_LIMIT_SEC << " sec\n";
    cout << "ITER_CHUNK=" << ITER_CHUNK << ", FAIL_TOL=" << FAIL_TOL << "\n";

    int round_id = 0;

    // ✅ infinite optimize loop until 3h
    while (true) {
        long long gsec = elapsed_sec(global_start);
        if (gsec >= GLOBAL_LIMIT_SEC) break;

        round_id++;
        cout << "\n========== ROUND " << round_id
             << " | elapsed=" << gsec << " sec ==========\n";

        for (double base : base_scales) {
            gsec = elapsed_sec(global_start);
            if (gsec >= GLOBAL_LIMIT_SEC) break;

            cout << "\n>>> BASE_SCALE=" << scientific << setprecision(2) << base
                 << " | elapsed=" << fixed << setprecision(0) << gsec << " sec <<<\n";

            // parallel optimize each focus N, but check global limit inside
            #pragma omp parallel for schedule(dynamic, 1)
            for (int t = 0; t < (int)keys.size(); t++) {
                int n = keys[t];

                // each N local fail counter
                int fails = 0;

                while (true) {
                    long long cursec = elapsed_sec(global_start);
                    if (cursec >= GLOBAL_LIMIT_SEC) break;

                    double before = best_side[n];
                    double after  = run_powerhouse_cycle_sa(res[n], ITER_CHUNK, base);

                    if (after < before - 1e-15) {
                        best_side[n] = after;
                        fails = 0;
                    } else {
                        fails++;
                        if (fails >= FAIL_TOL) break;
                    }
                }

                #pragma omp critical
                {
                    cout << "[N=" << n << "] best_side=" << fixed << setprecision(12) << best_side[n] << "\n";
                }
            }

            // checkpoint save if needed
            long long since_save = elapsed_sec(last_save);
            if (since_save >= CHECKPOINT_EVERY_SEC) {
                #pragma omp critical
                {
                    cout << "\n[Checkpoint] saving submission.csv ... elapsed="
                         << elapsed_sec(global_start) << " sec\n";
                    saveCSV("submission.csv", res);
                    last_save = chrono::steady_clock::now();
                }
            }
        }

        // end of round save (always)
        cout << "\n[Round End] saving submission.csv ... elapsed="
             << elapsed_sec(global_start) << " sec\n";
        saveCSV("submission.csv", res);
        last_save = chrono::steady_clock::now();
    }

    cout << "\n[Done] reached global time limit. Final save.\n";
    saveCSV("submission.csv", res);

    return 0;
}
