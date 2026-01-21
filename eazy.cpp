#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>
#include <random>
#include <complex>
#include <omp.h>
#include <new>     // ✅ 加这一行，修 placement new


using namespace std;

typedef complex<double> cd;

// --- CONSTANTS ---
constexpr int MAX_N = 205;
constexpr int NV = 15;
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
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];

    // cached bbox
    double xmin, xmax, ymin, ymax;

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

// --- MASTER: SA + Greedy Hybrid + 2-tree cooperative move ---
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

        // pick i (boundary-biased)
        int i;
        if (!bnd.empty() && rf() < 0.90) i = bnd[ri((int)bnd.size())];
        else i = ri(cur.n);

        // with some probability, do 2-tree cooperative move
        bool do_two = (cur.n >= 2) && (rf() < 0.18);

        if (!do_two) {
            // -------- single-tree move (original) --------
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
        } else {
            // -------- 2-tree cooperative move (NEW) --------
            // pick j (prefer boundary too)
            int j = -1;
            if (!bnd.empty() && (int)bnd.size() >= 2 && rf() < 0.70) {
                // pick boundary j != i
                for (int tries = 0; tries < 6; tries++) {
                    int cand = bnd[ri((int)bnd.size())];
                    if (cand != i) { j = cand; break; }
                }
            }
            if (j < 0) {
                // fallback: random j != i
                j = ri(cur.n - 1);
                if (j >= i) j++;
            }

            double oxi = cur.x[i], oyi = cur.y[i], oai = cur.a[i];
            double oxj = cur.x[j], oyj = cur.y[j], oaj = cur.a[j];
            Poly opi = cur.pl[i], opj = cur.pl[j];

            double mode = rf();

            if (mode < 0.50) {
                // (1) correlated opposite translation: keep center-of-mass
                double dx = (rf()-0.5) * step_scale;
                double dy = (rf()-0.5) * step_scale;
                cur.x[i] += dx; cur.y[i] += dy;
                cur.x[j] -= dx; cur.y[j] -= dy;

            } else if (mode < 0.80) {
                // (2) rotate both points around their midpoint (coordinated orbit)
                double mx = 0.5 * (cur.x[i] + cur.x[j]);
                double my = 0.5 * (cur.y[i] + cur.y[j]);
                double ang = (rf()-0.5) * step_scale * 0.25; // radians
                rotate_around(cur.x[i], cur.y[i], mx, my, ang);
                rotate_around(cur.x[j], cur.y[j], mx, my, ang);

                // small self-rotation to diversify
                if (rf() < 0.5) cur.a[i] += (rf()-0.5) * step_scale * 20.0;
                if (rf() < 0.5) cur.a[j] += (rf()-0.5) * step_scale * 20.0;

            } else {
                // (3) double square pressure: squeeze both toward target square
                cur.update_bbox();
                apply_square_pressure(cur, i, best_s, step_scale);
                apply_square_pressure(cur, j, best_s, step_scale);
            }

            cur.upd(i);
            cur.upd(j);

            bool hit = false;

            // first check i-j
            if (overlap(cur.pl[i], cur.pl[j])) hit = true;

            // then check against others
            if (!hit) {
                for (int k = 0; k < cur.n; k++) {
                    if (k == i || k == j) continue;
                    if (overlap(cur.pl[i], cur.pl[k]) || overlap(cur.pl[j], cur.pl[k])) { hit = true; break; }
                }
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

            // rollback both
            cur.x[i]=oxi; cur.y[i]=oyi; cur.a[i]=oai; cur.pl[i]=opi;
            cur.x[j]=oxj; cur.y[j]=oyj; cur.a[j]=oaj; cur.pl[j]=opj;
            cur.update_bbox();
            stagnation++;
        }
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

    while (getline(f, ln)) {
        stringstream ss(ln);
        string id, sx, sy, sa;
        if(!getline(ss, id, ',')) continue;
        getline(ss, sx, ','); getline(ss, sy, ','); getline(ss, sa, ',');

        int n = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));

        auto p = [](const string& s) {
            size_t st = s.find_first_of("0123456789.-");
            return (st == string::npos) ? 0.0 : stod(s.substr(st));
        };

        res[n].n = n;
        res[n].x[idx] = p(sx);
        res[n].y[idx] = p(sy);
        res[n].a[idx] = p(sa);
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

int main() {
    auto res = loadCSV("submission.csv");
    if(res.empty()) return 1;

    vector<int> keys;
    for (auto const& [n, g] : res) keys.push_back(n);
    sort(keys.rbegin(), keys.rend());

    vector<double> base_scales = {1e-3, 1e-5, 1e-7, 1e-9};

    for (double base : base_scales) {
        cout << "\n>>> SA-HYBRID PHASE | base_scale=" << base << " <<<\n";

        #pragma omp parallel for schedule(dynamic, 1)
        for (int t = 0; t < (int)keys.size(); t++) {
            int n = keys[t];
            auto start = chrono::steady_clock::now();

            res[n].update_bbox();
            res[n].recenter_to_origin();

            double best = res[n].side();
            int fails = 0;

            while (true) {
                double sec = chrono::duration_cast<chrono::seconds>(
                    chrono::steady_clock::now() - start
                ).count();
                if (sec >= 20) break;

                double after = run_powerhouse_cycle_sa(res[n], 250000, base);

                if (after < best - 1e-15) {
                    best = after;
                    fails = 0;
                } else {
                    if (++fails >= 3) break;
                }
            }

            #pragma omp critical
            cout << "[N=" << n << "] best_side=" << fixed << setprecision(12) << best << "\n";
        }

        saveCSV("submission.csv", res);
    }

    return 0;
}
