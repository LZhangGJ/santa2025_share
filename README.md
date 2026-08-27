# Santa 2025 Kaggle archive

Code and experiment history from Kaggle account [`ngyzly`](https://www.kaggle.com/ngyzly) for the [Santa 2025 — Christmas Tree Packing Challenge](https://www.kaggle.com/competitions/santa-2025).

## Repository map

- [`notebooks/`](notebooks/README.md): all 85 downloaded Kaggle notebooks with original metadata and checksums.
- [`native/`](native/README.md): 15 C/C++ exports (8 unique file contents) extracted verbatim from notebook cells.
- Root-level `.py`, `.cpp`, executables, logs, and `best_submission.csv`: the pre-existing working snapshot retained for reproducibility.

## Native build

The performance-critical code in this project is C++17 rather than ISO C because it uses STL and OpenMP. A typical Linux/Kaggle build is:

```bash
g++ -O3 -march=native -std=c++17 -fopenmp source.cpp -o solver
```

Exact filenames, inputs, and flags are preserved in the originating notebooks. Generated binaries and submitted results may be platform-specific.

## Provenance

This is an archival repository, not a claim that every dependency or fork was authored from scratch. Use each notebook's Kaggle link and `kernel-metadata.json` to inspect upstream kernels and datasets. No new license is asserted over third-party material.
