// single_group_optimizer_v2.cpp
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o single_group_optimizer_v2 single_group_optimizer_v2.cpp
//
// Run:
//   ./single_group_optimizer_v2 -i submission.csv -o submission.csv -g 58 -n 250000 -r 160
//
// Notes:
// - Designed for large N groups (e.g., 58/65/57/134/151)
// - Stronger mutation + repair + adaptive SA

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr long double PI = 3.141592653589793238462643383279502884L;

alignas(64) const long double TX[NV] = {
    0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
};
alignas(64) const long double TY[NV] = {
    0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5
};

// ------------------------------
// Fast RNG (xorshift128+ like)
// ------------------------------
struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1];
        uint64_t r = s0 + s1;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s[1] = rotl(s1, 37);
        return r;
    }
    inline long double rf() { return (next() >> 11) * 0x1.0p-53L; }   // [0,1)
    inline long double rf2() { return rf() * 2.0L - 1.0L; }           // (-1,1)
    inline int ri(int n) { return (int)(next() % (uint64_t)n); }
    inline long double gaussian() {
        long double u1 = rf() + 1e-12L;
        long double u2 = rf();
        return sqrtl(-2.0L * logl(u1)) * cosl(2.0L * PI * u2);
    }
};

struct Poly {
    long double px[NV], py[NV];
    long double x0, y0, x1, y1; // AABB
};

static inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);
    long double s = sinl(rad), c = cosl(rad);
    long double minx = 1e30L, miny = 1e30L, maxx = -1e30L, maxy = -1e30L;
    for (int i = 0; i < NV; i++) {
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        minx = min(minx, x); maxx = max(maxx, x);
        miny = min(miny, y); maxy = max(maxy, y);
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

// point-in-polygon
static inline bool pip(long double px, long double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py)) {
            long double xint = (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i];
            if (px < xint) in = !in;
        }
        j = i;
    }
    return in;
}

// segment intersection (strict-ish)
static inline bool segInt(long double ax, long double ay, long double bx, long double by,
                          long double cx, long double cy, long double dx, long double dy) {
    long double d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    long double d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
    long double d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    long double d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

static inline bool overlap(const Poly& a, const Poly& b) {
    // AABB reject
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;

    // vertex-in-poly
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    // edge-edge
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                       b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

struct Cfg {
    int n = 0;
    long double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    long double gx0, gy0, gx1, gy1;

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }

    inline void updAll() {
        for (int i = 0; i < n; i++) upd(i);
        updGlobal();
    }

    inline void updGlobal() {
        gx0 = gy0 = 1e30L; gx1 = gy1 = -1e30L;
        for (int i = 0; i < n; i++) {
            gx0 = min(gx0, pl[i].x0);
            gx1 = max(gx1, pl[i].x1);
            gy0 = min(gy0, pl[i].y0);
            gy1 = max(gy1, pl[i].y1);
        }
    }

    inline bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            if (overlap(pl[i], pl[j])) return true;
        }
        return false;
    }

    inline bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline long double side() const { return max(gx1 - gx0, gy1 - gy0); }
    inline long double score() const { long double s = side(); return s * s / (long double)n; }

    inline void center(long double &cx, long double &cy) const {
        cx = (gx0 + gx1) / 2.0L;
        cy = (gy0 + gy1) / 2.0L;
    }

    void getBoundary(vector<int>& b) const {
        b.clear();
        long double eps = 0.008L; // tighter for large N
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 - gx0 < eps || gx1 - pl[i].x1 < eps ||
                pl[i].y0 - gy0 < eps || gy1 - pl[i].y1 < eps)
                b.push_back(i);
        }
    }
};

// ------------------------------
// Helpers
// ------------------------------
static inline long double clampAngle(long double deg) {
    while (deg < 0) deg += 360.0L;
    while (deg >= 360.0L) deg -= 360.0L;
    return deg;
}

// Squeeze: safe uniform shrink around center
static Cfg squeeze(Cfg c) {
    long double cx, cy; c.center(cx, cy);
    for (long double scale = 0.9996L; scale >= 0.97L; scale -= 0.0006L) {
        Cfg t = c;
        for (int i = 0; i < c.n; i++) {
            t.x[i] = cx + (c.x[i] - cx) * scale;
            t.y[i] = cy + (c.y[i] - cy) * scale;
        }
        t.updAll();
        if (!t.anyOvl()) c = t;
        else break;
    }
    return c;
}

// Strong local "pull to center"
static Cfg compaction(Cfg c, int iters) {
    long double bestSide = c.side();
    for (int it = 0; it < iters; it++) {
        bool improved = false;
        long double cx, cy; c.center(cx, cy);
        for (int i = 0; i < c.n; i++) {
            long double ox = c.x[i], oy = c.y[i];
            long double dx = cx - c.x[i], dy = cy - c.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d < 1e-9L) continue;

            // step ladder
            const long double steps[] = {0.020L, 0.008L, 0.003L, 0.0012L, 0.0005L};
            for (long double st : steps) {
                c.x[i] = ox + dx / d * st;
                c.y[i] = oy + dy / d * st;
                c.upd(i);
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    long double s = c.side();
                    if (s + 1e-14L < bestSide) {
                        bestSide = s;
                        improved = true;
                        ox = c.x[i]; oy = c.y[i]; // keep
                    } else {
                        // revert
                        c.x[i] = ox; c.y[i] = oy; c.upd(i);
                    }
                } else {
                    c.x[i] = ox; c.y[i] = oy; c.upd(i);
                }
            }
        }
        c.updGlobal();
        if (!improved) break;
    }
    return c;
}

// Local search: tiny moves + tiny rotations
static Cfg localSearch(Cfg c, int maxIter) {
    long double bestSide = c.side();
    const long double steps[] = {0.006L, 0.0022L, 0.0009L, 0.00035L, 0.00015L};
    const long double rots[]  = {2.0L, 0.8L, 0.3L, 0.12L};

    const int dx8[] = {1,-1,0,0,1,1,-1,-1};
    const int dy8[] = {0,0,1,-1,1,-1,1,-1};

    for (int it = 0; it < maxIter; it++) {
        bool improved = false;
        long double cx, cy; c.center(cx, cy);

        for (int i = 0; i < c.n; i++) {
            // pull toward center
            long double vx = cx - c.x[i], vy = cy - c.y[i];
            long double dist = sqrtl(vx*vx + vy*vy);

            if (dist > 1e-9L) {
                for (long double st : steps) {
                    long double ox = c.x[i], oy = c.y[i];
                    c.x[i] += vx / dist * st;
                    c.y[i] += vy / dist * st;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        long double s = c.side();
                        if (s + 1e-14L < bestSide) { bestSide = s; improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); }
                    } else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }

            // tiny translation
            for (long double st : steps) {
                for (int d = 0; d < 8; d++) {
                    long double ox = c.x[i], oy = c.y[i];
                    c.x[i] += (long double)dx8[d] * st;
                    c.y[i] += (long double)dy8[d] * st;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        long double s = c.side();
                        if (s + 1e-14L < bestSide) { bestSide = s; improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); }
                    } else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }

            // tiny rotation
            for (long double rt : rots) {
                for (long double da : {rt, -rt}) {
                    long double oa = c.a[i];
                    c.a[i] = clampAngle(c.a[i] + da);
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        long double s = c.side();
                        if (s + 1e-14L < bestSide) { bestSide = s; improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); }
                    } else { c.a[i]=oa; c.upd(i); }
                }
            }
        }

        if (!improved) break;
    }
    return c;
}

// Try fix overlaps (repair) after a mutation
static bool repairOverlaps(Cfg &c, FastRNG &rng, int maxPass=200) {
    // push overlapped items outward + slight rotate
    for (int pass = 0; pass < maxPass; pass++) {
        bool any = false;
        long double cx, cy; c.center(cx, cy);

        for (int i = 0; i < c.n; i++) {
            if (!c.hasOvl(i)) continue;
            any = true;

            long double ox = c.x[i], oy = c.y[i], oa = c.a[i];
            // push away from center OR random
            long double vx = c.x[i] - cx;
            long double vy = c.y[i] - cy;
            long double d = sqrtl(vx*vx + vy*vy);

            long double step = 0.015L + 0.01L * rng.rf();
            if (d > 1e-9L) {
                c.x[i] += vx / d * step;
                c.y[i] += vy / d * step;
            } else {
                c.x[i] += rng.rf2() * step;
                c.y[i] += rng.rf2() * step;
            }
            c.a[i] = clampAngle(c.a[i] + rng.rf2() * (8.0L + 15.0L*rng.rf()));
            c.upd(i);

            if (c.hasOvl(i)) {
                // revert partially and try another
                c.x[i] = ox + rng.rf2() * 0.006L;
                c.y[i] = oy + rng.rf2() * 0.006L;
                c.a[i] = clampAngle(oa + rng.rf2() * 12.0L);
                c.upd(i);
                if (c.hasOvl(i)) {
                    // full revert
                    c.x[i] = ox; c.y[i] = oy; c.a[i] = oa;
                    c.upd(i);
                }
            }
        }

        c.updGlobal();
        if (!any) return true;
    }
    return !c.anyOvl();
}

// Mutation operators for big-N
static Cfg mutate(Cfg c, FastRNG &rng, long double strength) {
    // strength ~ [0.02, 0.12]
    int n = c.n;

    int mode = rng.ri(5);

    long double cx, cy; c.center(cx, cy);

    // number of items to mutate
    int k = (int)max(2.0L, (long double)n * (0.06L + 0.25L*strength));
    k = min(k, n);

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    for (int i = 0; i < k; i++) swap(idx[i], idx[rng.ri(n)]);

    if (mode == 0) {
        // Random re-scatter selected items near center
        for (int t = 0; t < k; t++) {
            int i = idx[t];
            c.x[i] = cx + rng.gaussian() * (0.18L + 1.4L*strength);
            c.y[i] = cy + rng.gaussian() * (0.18L + 1.4L*strength);
            c.a[i] = clampAngle(c.a[i] + rng.gaussian() * (25.0L + 120.0L*strength));
            c.upd(i);
        }
    } else if (mode == 1) {
        // Ring relocation: put selected items on a circle, then let repair fix
        long double R = (0.35L + 1.8L*strength) * (0.55L + 0.015L*n);
        for (int t = 0; t < k; t++) {
            int i = idx[t];
            long double ang = 2.0L * PI * rng.rf();
            c.x[i] = cx + cosl(ang) * R + rng.gaussian() * 0.03L;
            c.y[i] = cy + sinl(ang) * R + rng.gaussian() * 0.03L;
            c.a[i] = clampAngle(360.0L * rng.rf());
            c.upd(i);
        }
    } else if (mode == 2) {
        // Block move: choose a random direction and move selected items together
        long double dx = rng.rf2() * (0.08L + 0.9L*strength);
        long double dy = rng.rf2() * (0.08L + 0.9L*strength);
        long double da = rng.rf2() * (5.0L + 50.0L*strength);
        for (int t = 0; t < k; t++) {
            int i = idx[t];
            c.x[i] += dx;
            c.y[i] += dy;
            c.a[i] = clampAngle(c.a[i] + da);
            c.upd(i);
        }
    } else if (mode == 3) {
        // Boundary squeeze mutation: pull boundary inward aggressively
        vector<int> boundary;
        c.getBoundary(boundary);
        if (!boundary.empty()) {
            int take = min((int)boundary.size(), max(2, (int)(boundary.size() * (0.35L + 0.5L*strength))));
            for (int t = 0; t < take; t++) {
                int i = boundary[rng.ri((int)boundary.size())];
                long double vx = cx - c.x[i], vy = cy - c.y[i];
                long double d = sqrtl(vx*vx + vy*vy);
                if (d > 1e-9L) {
                    c.x[i] += vx / d * (0.03L + 0.25L*strength);
                    c.y[i] += vy / d * (0.03L + 0.25L*strength);
                }
                c.a[i] = clampAngle(c.a[i] + rng.rf2() * (10.0L + 70.0L*strength));
                c.upd(i);
            }
        } else {
            // fallback random
            for (int t = 0; t < k; t++) {
                int i = idx[t];
                c.x[i] += rng.gaussian() * (0.05L + 0.8L*strength);
                c.y[i] += rng.gaussian() * (0.05L + 0.8L*strength);
                c.a[i] = clampAngle(c.a[i] + rng.gaussian() * (10.0L + 90.0L*strength));
                c.upd(i);
            }
        }
    } else {
        // Multi-tree swap/rotate shuffle
        for (int t = 0; t < k; t++) {
            int i = idx[t];
            int j = rng.ri(n);
            if (i == j) continue;
            swap(c.x[i], c.x[j]);
            swap(c.y[i], c.y[j]);
            if (rng.rf() < 0.6) swap(c.a[i], c.a[j]);
            c.a[i] = clampAngle(c.a[i] + rng.rf2() * 15.0L);
            c.a[j] = clampAngle(c.a[j] + rng.rf2() * 15.0L);
            c.upd(i); c.upd(j);
        }
    }

    c.updGlobal();
    return c;
}

// SA step (with multiple move types)
static Cfg sa_opt(Cfg start, int iter, long double T0, long double Tmin, uint64_t seed) {
    FastRNG rng(seed);

    Cfg cur = start;
    Cfg best = start;
    long double curSide = cur.side();
    long double bestSide = curSide;

    long double T = T0;
    long double alpha = powl(Tmin / T0, 1.0L / (long double)iter);

    int stuck = 0;
    int accepted = 0;
    int tried = 0;

    for (int it = 0; it < iter; it++) {
        tried++;

        // make a trial
        Cfg trial = cur;

        int mt = rng.ri(12);
        long double sc = T / T0;

        bool ok = true;

        if (mt <= 3) {
            // single tree move (gaussian)
            int i = rng.ri(trial.n);
            long double ox = trial.x[i], oy = trial.y[i], oa = trial.a[i];
            if (mt == 0) {
                trial.x[i] += rng.gaussian() * (0.25L * sc);
                trial.y[i] += rng.gaussian() * (0.25L * sc);
            } else if (mt == 1) {
                long double cx, cy; trial.center(cx, cy);
                long double dx = cx - trial.x[i], dy = cy - trial.y[i];
                long double d = sqrtl(dx*dx + dy*dy);
                if (d > 1e-9L) {
                    long double st = rng.rf() * (0.35L*sc);
                    trial.x[i] += dx / d * st;
                    trial.y[i] += dy / d * st;
                }
            } else if (mt == 2) {
                trial.a[i] = clampAngle(trial.a[i] + rng.gaussian() * (70.0L * sc));
            } else {
                trial.x[i] += rng.rf2() * (0.35L*sc);
                trial.y[i] += rng.rf2() * (0.35L*sc);
                trial.a[i] = clampAngle(trial.a[i] + rng.rf2() * (60.0L*sc));
            }
            trial.upd(i);
            if (trial.hasOvl(i)) {
                trial.x[i]=ox; trial.y[i]=oy; trial.a[i]=oa;
                trial.upd(i);
                ok = false;
            }
        }
        else if (mt == 4) {
            // swap two trees
            int i = rng.ri(trial.n);
            int j = rng.ri(trial.n);
            if (i == j) ok = false;
            else {
                swap(trial.x[i], trial.x[j]);
                swap(trial.y[i], trial.y[j]);
                if (rng.rf() < 0.8) swap(trial.a[i], trial.a[j]);
                trial.upd(i); trial.upd(j);
                if (trial.hasOvl(i) || trial.hasOvl(j)) ok = false;
            }
        }
        else if (mt == 5) {
            // small group move
            int k = max(2, (int)(trial.n * 0.03));
            long double dx = rng.rf2() * (0.18L * sc);
            long double dy = rng.rf2() * (0.18L * sc);
            for (int t = 0; t < k; t++) {
                int i = rng.ri(trial.n);
                trial.x[i] += dx;
                trial.y[i] += dy;
                trial.upd(i);
            }
            // overlap may appear, attempt quick repair
            if (!repairOverlaps(trial, rng, 35)) ok = false;
        }
        else if (mt == 6) {
            // global squeeze attempt (rare)
            long double factor = 1.0L - rng.rf() * (0.0035L * sc);
            long double cx, cy; trial.center(cx, cy);
            for (int i = 0; i < trial.n; i++) {
                trial.x[i] = cx + (trial.x[i] - cx) * factor;
                trial.y[i] = cy + (trial.y[i] - cy) * factor;
            }
            trial.updAll();
            if (trial.anyOvl()) ok = false;
        }
        else if (mt == 7) {
            // strong mutation then repair (this is the "突变" key)
            long double strength = 0.02L + 0.10L * rng.rf() + 0.10L * (1.0L - sc);
            trial = mutate(trial, rng, strength);
            if (!repairOverlaps(trial, rng, 120)) ok = false;
        }
        else if (mt == 8) {
            // boundary inward + rotate a little
            vector<int> boundary;
            trial.getBoundary(boundary);
            if (boundary.empty()) ok = false;
            else {
                int take = min((int)boundary.size(), max(2, (int)(boundary.size() * 0.5)));
                long double cx, cy; trial.center(cx, cy);
                for (int t = 0; t < take; t++) {
                    int i = boundary[rng.ri((int)boundary.size())];
                    long double ox = trial.x[i], oy = trial.y[i], oa = trial.a[i];
                    long double dx = cx - trial.x[i], dy = cy - trial.y[i];
                    long double d = sqrtl(dx*dx + dy*dy);
                    if (d > 1e-9L) {
                        trial.x[i] += dx/d * (0.01L + 0.12L*sc);
                        trial.y[i] += dy/d * (0.01L + 0.12L*sc);
                    }
                    trial.a[i] = clampAngle(trial.a[i] + rng.rf2() * (10.0L + 40.0L*sc));
                    trial.upd(i);
                    if (trial.hasOvl(i)) { trial.x[i]=ox; trial.y[i]=oy; trial.a[i]=oa; trial.upd(i); }
                }
                trial.updGlobal();
                if (trial.anyOvl()) ok = false;
            }
        }
        else {
            // tiny random jitter (stabilizer)
            int i = rng.ri(trial.n);
            long double ox = trial.x[i], oy = trial.y[i];
            trial.x[i] += rng.rf2() * (0.0015L + 0.012L*sc);
            trial.y[i] += rng.rf2() * (0.0015L + 0.012L*sc);
            trial.upd(i);
            if (trial.hasOvl(i)) { trial.x[i]=ox; trial.y[i]=oy; trial.upd(i); ok=false; }
        }

        if (!ok) {
            stuck++;
            // if too stuck, reheat slightly
            if (stuck > 3000) {
                T = min(T0, T * 1.8L);
                stuck = 0;
                accepted = 0; tried = 0;
            }
            T *= alpha;
            if (T < Tmin) T = Tmin;
            continue;
        }

        trial.updGlobal();
        long double ns = trial.side();
        long double delta = ns - curSide;

        bool accept = false;
        if (delta <= 0) accept = true;
        else {
            long double prob = expl(-delta / T);
            if (rng.rf() < prob) accept = true;
        }

        if (accept) {
            cur = trial;
            curSide = ns;
            accepted++;
            stuck = 0;

            if (ns + 1e-14L < bestSide) {
                bestSide = ns;
                best = cur;
            }
        } else {
            stuck++;
        }

        // acceptance-rate based temperature tweak
        if (it % 1500 == 1499) {
            long double accRate = (tried > 0 ? (long double)accepted / (long double)tried : 0.0L);
            // if too cold (almost no accept), warm it a bit
            if (accRate < 0.01L) T = min(T0, T * 1.6L);
            // if too hot (accept too many), cool faster
            if (accRate > 0.25L) T *= 0.6L;

            accepted = 0;
            tried = 0;
        }

        T *= alpha;
        if (T < Tmin) T = Tmin;
    }

    return best;
}

// Full pipeline for one start config
static Cfg polish(Cfg c) {
    c = squeeze(c);
    c = compaction(c, 60);
    c = localSearch(c, 80);
    c = squeeze(c);
    c = compaction(c, 50);
    c = localSearch(c, 120);
    return c;
}

// Parallel multi-restart optimization
static Cfg optimizeParallel(const Cfg& base, int iters, int restarts) {
    Cfg globalBest = base;
    long double globalBestSide = base.side();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        FastRNG rng(7777 + (uint64_t)tid * 99991ULL + (uint64_t)base.n * 131ULL);

        Cfg localBest = base;
        long double localBestSide = base.side();

        #pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; r++) {
            Cfg s = base;

            // restart strength schedule
            long double strength = 0.015L + 0.12L * ((long double)r / (long double)max(1, restarts-1));
            strength += 0.04L * rng.rf();

            // create a new start by mutating base
            if (r > 0) {
                s = mutate(s, rng, strength);
                if (!repairOverlaps(s, rng, 180)) continue;
            }

            // SA
            uint64_t seed = 20240101ULL + (uint64_t)r * 10007ULL + (uint64_t)tid * 999983ULL + (uint64_t)base.n;
            long double T0 = 2.8L;
            long double Tmin = 5e-7L;
            Cfg o = sa_opt(s, iters, T0, Tmin, seed);

            // final polishing
            o = polish(o);

            if (!o.anyOvl()) {
                long double os = o.side();
                if (os + 1e-14L < localBestSide) {
                    localBestSide = os;
                    localBest = o;
                }
            }
        }

        #pragma omp critical
        {
            if (!localBest.anyOvl() && localBestSide + 1e-14L < globalBestSide) {
                globalBestSide = localBestSide;
                globalBest = localBest;
            }
        }
    }

    // final polish global best
    globalBest = polish(globalBest);
    if (globalBest.anyOvl()) return base;
    return globalBest;
}

// ------------------------------
// CSV IO
// ------------------------------
static map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln;
    getline(f, ln); // header

    map<int, vector<tuple<int,long double,long double,long double>>> data;

    while (getline(f, ln)) {
        if (ln.empty()) continue;
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        if (p1==string::npos||p2==string::npos||p3==string::npos) continue;

        string id=ln.substr(0,p1);
        string xs=ln.substr(p1+1,p2-p1-1);
        string ys=ln.substr(p2+1,p3-p2-1);
        string ds=ln.substr(p3+1);

        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);

        int n=stoi(id.substr(0,3));
        int idx=stoi(id.substr(4));
        data[n].push_back({idx, stold(xs), stold(ys), stold(ds)});
    }

    for (auto &kv : data) {
        int n = kv.first;
        auto &v = kv.second;
        Cfg c; c.n = n;
        // init defaults
        for (int i = 0; i < n; i++) { c.x[i]=0; c.y[i]=0; c.a[i]=0; }
        for (auto &t : v) {
            int i; long double x,y,d;
            tie(i,x,y,d)=t;
            if (i>=0 && i<n) {
                c.x[i]=x; c.y[i]=y; c.a[i]=d;
            }
        }
        c.updAll();
        cfg[n]=c;
    }
    return cfg;
}

static void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(17) << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        auto it = cfg.find(n);
        if (it == cfg.end()) continue;
        const Cfg& c = it->second;
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i
              << ",s" << c.x[i]
              << ",s" << c.y[i]
              << ",s" << c.a[i] << "\n";
        }
    }
}

static void printHelp() {
    printf("single_group_optimizer_v2\n");
    printf("Usage:\n");
    printf("  ./single_group_optimizer_v2 -i submission.csv -o submission.csv -g 58 -n 250000 -r 160\n\n");
    printf("Args:\n");
    printf("  -i <input_csv>   (default: submission.csv)\n");
    printf("  -o <output_csv>  (default: submission_optimized.csv)\n");
    printf("  -g <group_n>     (required) group number 1..200\n");
    printf("  -n <iters>       SA iterations per restart (default: 220000)\n");
    printf("  -r <restarts>    number of restarts (default: 140)\n");
}

int main(int argc, char** argv) {
    string in="submission.csv";
    string out="submission_optimized.csv";
    int iters=220000;
    int restarts=140;
    int targetN=-1;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) in=argv[++i];
        else if (a=="-o" && i+1<argc) out=argv[++i];
        else if (a=="-n" && i+1<argc) iters=stoi(argv[++i]);
        else if (a=="-r" && i+1<argc) restarts=stoi(argv[++i]);
        else if (a=="-g" && i+1<argc) targetN=stoi(argv[++i]);
        else if (a=="-h" || a=="--help") { printHelp(); return 0; }
    }

    if (targetN < 1 || targetN > 200) {
        printf("Error: you must set -g <group_n> between 1..200\n\n");
        printHelp();
        return 1;
    }

    int numThreads = omp_get_max_threads();
    printf("Single Group Optimizer v2 (%d threads)\n", numThreads);
    printf("Target group: n=%d\n", targetN);
    printf("Iters: %d, Restarts: %d\n", iters, restarts);
    printf("Loading %s...\n", in.c_str());

    auto cfg = loadCSV(in);
    if (cfg.empty()) {
        printf("No data loaded from %s\n", in.c_str());
        return 1;
    }
    if (!cfg.count(targetN)) {
        printf("Error: group n=%d not found in input\n", targetN);
        return 1;
    }

    Cfg c = cfg[targetN];
    long double os = c.score();
    long double oSide = c.side();
    printf("Initial: score=%.12Lf, side=%.12Lf\n", os, oSide);
    if (c.anyOvl()) printf("WARNING: initial config has overlap!\n");

    // Big-group adaptive scaling
    int n = targetN;
    int adjIters = iters;
    int adjRestarts = restarts;

    if (n >= 120) { adjIters = (int)(iters * 1.10); adjRestarts = (int)(restarts * 1.10); }
    if (n >= 150) { adjIters = (int)(iters * 1.25); adjRestarts = (int)(restarts * 1.25); }
    if (n <= 30)  { adjIters = (int)(iters * 1.50); adjRestarts = (int)(restarts * 1.50); }

    adjRestarts = max(20, adjRestarts);

    auto t0 = chrono::high_resolution_clock::now();
    printf("Optimizing: iters=%d restarts=%d ...\n", adjIters, adjRestarts);

    Cfg best = optimizeParallel(c, adjIters, adjRestarts);

    auto t1 = chrono::high_resolution_clock::now();
    long double el = chrono::duration_cast<chrono::milliseconds>(t1-t0).count()/1000.0L;

    bool ok = !best.anyOvl();
    long double ns = best.score();
    long double nSide = best.side();

    printf("\n========================================\n");
    printf("Result n=%d\n", targetN);
    printf("Side: %.12Lf -> %.12Lf\n", oSide, nSide);
    printf("Score: %.12Lf -> %.12Lf\n", os, ns);
    if (!ok) {
        printf("WARNING: overlap detected, keep original\n");
        best = c;
        ns = os;
    } else if (ns + 1e-14L < os) {
        printf("Improvement: %.6Lf (%.3Lf%%)\n", (os-ns), (os-ns)/os*100.0L);
    } else {
        printf("No improvement\n");
        best = c;
        ns = os;
    }
    printf("Time: %.1Lfs\n", el);
    printf("========================================\n");

    cfg[targetN] = best;
    saveCSV(out, cfg);
    printf("Saved -> %s\n", out.c_str());
    return 0;
}
