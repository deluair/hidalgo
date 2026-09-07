# hidalgo

Fast economic-complexity kernels (ECI, PCI, product proximity, density) written in Rust, used from Python. It reproduces the Hidalgo-Hausmann eigenvector method that TradeWeave's data builds use, checked against TradeWeave's published rankings, and on the benchmark in this repository it ran 82x to 115x faster than the NumPy path it replaces (three runs on one machine, 2026-09-07; the method, the machine, and the caveats are in [Performance](#performance-the-measured-numbers)).

Counts and timings marked (2026-09-07) were re-measured on that date against this working tree and the TradeWeave parquet on the author's laptop. Where an earlier recorded figure did not reproduce, this document reports the new measurement and says so.

If you are an economist and not a programmer: think of this as a very fast research assistant for one specific job. You hand it a country-by-product export table, and it hands back the standard complexity toolkit: revealed comparative advantage (RCA), the binary specialization matrix, the Economic Complexity Index (ECI), the Product Complexity Index (PCI), product proximity, and density. You keep working in Python, exactly as before. The commands you type look the same; only the machinery behind them changed, the way a Stata command would feel identical if its internals were rewritten in a faster language.

## What this is and the problem it solves

Computing complexity metrics on real trade data is numerically expensive in one specific place. The country side is cheap (there are only a couple hundred countries). The product side is not: PCI is defined through a matrix that has one row and one column per product, roughly 6,000 x 6,000 for HS6 trade data. The standard scientific-Python route, NumPy's `np.linalg.eig`, solves that problem by computing *every* eigenvalue and eigenvector of the matrix, all ~6,000 of them, and then throwing away everything except the single eigenvector the method actually needs. That is like estimating every possible regression specification when you only wanted one coefficient. On real data this one step takes about half a minute per year of data, and a full historical rebuild loops over many years.

`hidalgo` replaces that step with a purpose-built calculation engine. It is a Rust "crate" (Rust's word for a code library: a bundle of functions other programs call, with no user interface of its own) plus a thin bridge that makes it callable from Python. It computes only the eigenvector that matters, in parallel, and returns plain NumPy arrays.

The name is for César Hidalgo, co-author of the method it implements (Hidalgo and Hausmann's economic complexity framework and the 2007 product-space proximity measure).

## Why it exists and where it fits

This repository is one piece of a larger personal research platform:

- **TradeWeave** (tradeweave.org) is a trade-analytics platform built on BACI bilateral trade data. Its data-preparation scripts compute RCA, ECI, PCI, and proximity for every year and store the results as parquet files that the website then serves. Its `rca_matrix` parquet tree spans 1996 to 2024, 29 years, and the 2024 slice is 226 countries by 6,266 products (2026-09-07, counted with DuckDB over `data/parquet/rca_matrix/`). Those scripts were the original home of the slow `np.linalg.eig` step; `hidalgo` was built to replace it while reproducing the published rankings.
- **BDPolicyLab** (bdpolicylab.com) is named in the design spec as a later consumer if it computes complexity metrics locally.

An important architectural point from the design spec: all of this runs *locally*, on the researcher's own machine, during data preparation. The public web server only ever serves precomputed files. So `hidalgo` never runs on the server and the server needs no Rust installed. It is a workshop tool, not a deployed service.

## How it works end to end

### What goes in

A single numeric matrix for one year: rows are countries, columns are products, entries are export values (TradeWeave feeds it BACI values in thousands of USD, but the math is unit-agnostic; RCA is a ratio of shares, so the currency unit cancels). In Python this is a two-dimensional NumPy array, the in-memory equivalent of an Excel sheet of numbers with no labels. You keep the row and column labels (ISO country codes, HS6 product codes) on your side and map them back afterwards.

There are three entry points, depending on how far along your data already is:

- `bundle_from_exports(exports, ...)`: raw export values in, everything computed for you.
- `bundle_from_rca(rca, threshold=1.0, ...)`: you already have an RCA matrix (this repo's parity tests feed TradeWeave's exact published `rca_matrix` values in here, to isolate the complexity math from RCA reconstruction).
- `bundle_from_m(m, ...)`: you already have the binary 0/1 specialization matrix.

### What it computes

The full Hidalgo-Hausmann bundle, in economist terms:

1. **RCA (Balassa index).** A country's share of a product in its export basket, divided by that product's share of world trade. Countries with zero exports and products nobody exports are handled explicitly (no division-by-zero surprises); the behavior matches TradeWeave's Python reference function.
2. **Binary specialization matrix M.** Entry is 1 where RCA >= threshold (default 1.0), else 0: "does this country meaningfully export this product?"
3. **Diversity and ubiquity.** Row and column sums of M: how many products a country exports competitively, and how many countries export a given product.
4. **ECI and PCI.** The second eigenvector of the country-side and product-side "reflections" matrices, standardized to mean 0 and standard deviation 1 (a z-score). Sign conventions follow the Atlas of Economic Complexity: ECI is oriented to correlate positively with diversity, PCI to correlate negatively with ubiquity (complex products are made in few places). Negative PCI values are preserved, never filtered.
5. **Product proximity.** The Hidalgo et al. (2007) measure behind the product space: for each pair of products, the conditional probability of co-export, computed as the co-export count divided by the larger of the two products' ubiquities. Symmetric, with 1s on the diagonal (for any product at least one country exports; a product nobody exports gets a zero row and a zero diagonal).
6. **Density.** For each country-product pair, how close a country's current export basket sits to a product it does not yet export: the proximity-weighted share of that product's neighborhood the country already occupies.

House rules encoded in the math: countries with zero diversity are dropped before the eigenvector step; if fewer than 3 diversified countries remain (and symmetrically for products), complexity is undefined for that input and the library raises an error saying so rather than returning nonsense. Products with zero ubiquity are excluded from PCI. Ubiquity is computed over all countries before any filtering, matching the Python reference.

### What comes out

A Python dictionary (a labeled container, like a folder of named result tables) of NumPy arrays:

`eci`, `pci`, `diversity`, `ubiquity`, `proximity` (product x product), `density` (country x product), plus `kept_countries` and `kept_products` (the row and column positions that survived filtering, so you can map results back to your own country and product labels), and `eci_converged` / `pci_converged` flags confirming the solver finished properly.

One matrix per call, one year per matrix. Looping over years is the caller's job, same as the Python scripts it replaces.

## The algorithms, honestly stated

Two formulations of complexity exist in the literature, and this repo implements the second:

- **Method of reflections** (Hidalgo and Hausmann's original iterative scheme): repeatedly average diversity and ubiquity back and forth between countries and products. `hidalgo` does *not* run this iteration.
- **Eigenvector formulation** (the Atlas / spectral method, which the reflections iteration converges toward): ECI is defined directly as the second eigenvector of the country reflections matrix, and PCI as the second eigenvector of the product-side counterpart. This is what TradeWeave's Python reference computes with `np.linalg.eig`, and it is exactly what `hidalgo` computes.

An eigenvector, for this purpose, is a special weighting of countries (or products) that the reflections matrix maps onto a scaled copy of itself; the scaling factor is the eigenvalue. The *first* eigenvector of a reflections matrix is mathematically trivial (the matrix is row-stochastic, meaning each row sums to 1, so its top eigenvalue is exactly 1 with a constant eigenvector carrying no information). The *second* eigenvector is the complexity ranking.

Where `hidalgo` differs from the NumPy reference is purely in *how* it finds that second eigenvector:

- **Power iteration** instead of full decomposition. Multiply a trial vector by the matrix, rescale, repeat; the vector converges to the dominant eigenvector. This is the cheap classical workhorse, needing only matrix-times-vector products.
- **Deflation** to reach the *second* eigenvector. The first eigenvector is known in closed form, so it is projected out of the trial vector at every step, leaving the second as the dominant survivor. This subtraction is only mathematically legitimate on a symmetric matrix (where eigenvectors are orthogonal), so the solver first applies a similarity transform that turns the reflections matrix into an equivalent symmetric one, solves there, and transforms back. The full derivation is written out in `docs/superpowers/plans/2026-06-02-hidalgo.md`.
- **A shift** (iterating on S + I rather than S) so the iteration targets the second-*largest* eigenvalue algebraically, not the largest in magnitude.
- **Matrix-free.** The roughly 6,000 x 6,000 product reflections matrix is never built in memory at all. Applying it to a vector is decomposed into two multiplications with the thin country x product matrix M plus elementwise rescalings.

The payoff: cost proportional to (iterations x matrix size squared) instead of (matrix size cubed), and in practice the iteration converges in tens of rounds. Because downstream everything is z-scored and ranked, only the eigenvector's direction matters, so the result is the same ranking `np.linalg.eig` produces. That equivalence is not assumed; it is tested (next two sections). A full dense eigensolver (a hand-written Jacobi routine) also exists in the repo, but only inside test code, as an independent oracle to check the fast solver against.

**A provenance caveat, stated plainly.** TradeWeave's published `pci_rankings.parquet` is a hybrid. The HS92 core catalog, the 5,115 HS6 codes in `products.parquet` (2026-09-07), is scored by the eigenvector method, which is what `hidalgo` implements. The remaining extended products from later HS revisions (HS96 to HS22) are scored by a different formula in TradeWeave's `data/compute_ext_rankings.py` ("PCI = mean ECI of exporters with RCA >= 1") and inserted into the same table. For 2024 the published table carries 6,266 products, of which 4,762 fall in the core catalog and 1,504 are the extended add-on (2026-09-07, counted with DuckDB). `hidalgo` intentionally does not reproduce that add-on, so parity against the published file is checked on the core universe only.

## Technology choices, and what was rejected

**Why Rust instead of pure Python/NumPy.** Rust is a compiled systems language: the code is translated to machine instructions ahead of time, so tight numerical loops run at full hardware speed instead of through Python's interpreter. Honestly attributed, most of the measured gain here comes from the *algorithm* (computing one eigenvector instead of a full spectrum); a careful NumPy implementation of power iteration would also be much faster than `np.linalg.eig`. Rust then buys the rest: the proximity and density kernels are exactly the cache-friendly, embarrassingly parallel loops (memory is read in the order the CPU handles best, and each output piece can be computed independently, so extra CPU cores help almost linearly) where compiled code with explicit threading wins, and the whole bundle comes back in a quarter of a second with deterministic results across thread counts (the parallel kernels sum each output entry in a fixed order, so the same input gives bit-identical output no matter how many CPU cores run it, which matters for reproducible research).

**Why not C++ or Cython.** Both could hit similar speeds. C++ offers no memory safety net (the class of crash-and-corruption bugs Rust's compiler rules out) and a messier build story for shipping into Python. Cython lives halfway between Python and C, which means maintaining a hybrid dialect and a build chain of its own. Rust plus its packaging tools gives one clean, self-contained artifact. (This paragraph is design rationale; the repo's own documents record the Rust decision and the dependency decisions, not a formal bake-off against C++/Cython.)

**Why no BLAS/LAPACK or heavy linear-algebra dependency.** A documented decision in the implementation plan: the original design sketch considered the `faer` linear-algebra crate, and the plan explicitly dropped it. The core crate's only runtime dependency is `rayon` (a Rust library for spreading loops across CPU cores). The only dense matrix product needed is a single M-transpose-times-M, easily hand-written, and avoiding external math libraries means the crate builds anywhere a Rust compiler exists, with nothing to link against.

**Why PyO3 and maturin.** PyO3 is the bridge that lets Python call Rust functions as if they were ordinary Python functions; maturin is the build tool that compiles the Rust and installs the result into a Python environment in one command. Together they mean the economist-facing surface is entirely Python: `import hidalgo`, pass NumPy arrays, get NumPy arrays back. Nobody using the library ever reads or writes Rust. The alternative of shipping a Rust command-line program with CSV files in and out was rejected as a workflow (the design spec lists "no CLI binary" as an explicit non-goal): the data lives in Python, so the library goes to the data.

## Performance: the measured numbers

**Method.** `tests/bench_compare.py` loads TradeWeave's latest `rca_matrix` year into a dense NumPy array, then times two paths on that same matrix: a NumPy reference doing two `np.linalg.eig` calls (ECI and PCI) plus proximity, and `hidalgo.bundle_from_rca`. Each path is timed once per run with `time.perf_counter`. There is no warm-up round, no repetition inside a run, and no statistical treatment: the script reports one wall-clock time per path.

**Machine.** MacBook Air, Apple M5, 10 cores, macOS 26.6.2, hidalgo built with `maturin develop --release`, NumPy 2.4.6, Python 3.11.

**Input.** Year 2024, 226 countries by 6,266 products.

**Result, three consecutive runs on 2026-09-07:**

| Run | NumPy reference | hidalgo | Ratio |
|---|---|---|---|
| 1 | 30,714.7 ms | 266.7 ms | 115.2x |
| 2 | 44,337.1 ms | 538.4 ms | 82.4x |
| 3 | 33,508.0 ms | 299.0 ms | 112.1x |

The honest summary is a range, 82x to 115x, not a single figure. Run 2 is slow on both paths because other work was running on the same laptop; the spread between runs is machine load, not algorithmic variance, and it is the reason this document does not quote one decimal-place number. The comparison also mildly favors NumPy: hidalgo's time includes density, which the NumPy path never computes.

**On the previously published figure.** An earlier version of this README recorded 126.9x from a run on 2026-08-11 (30,732 ms against 242 ms, 226 countries by 6,429 products). That exact ratio did not reproduce on 2026-09-07 and the product count has since changed to 6,266, so it has been replaced by the measurements above rather than carried forward. Treat any single ratio from this benchmark as an order-of-magnitude statement.

The gain comes from skipping `np.linalg.eig`, which computes the full spectrum of the roughly 6,000 x 6,000 product reflections matrix just to extract one eigenvector, where hidalgo's matrix-free deflated power iteration needs only matrix-vector products and converges in tens of iterations. The two-orders-of-magnitude scale of the gap holds across all three runs; the precise multiple does not.

To reproduce: `python tests/bench_compare.py` from the project environment, with the TradeWeave parquet tree available (set the environment variable `HIDALGO_TRADEWEAVE_DATA` if it is not at `~/tradeweave/data/parquet`). Your hardware, your NumPy build, and your machine load will all move the number.

## Correctness: how we know the math is right

Speed without verified correctness would be worthless for research. The test suite (`pytest tests/`) gates every change with three layers:

1. **Algorithm proof, provenance-independent.** On the same input matrix, hidalgo must agree with NumPy's full `np.linalg.eig`. The assertion threshold is Spearman rank correlation > 0.9999999 for both ECI and PCI. This is the layer that proves the fast solver finds the same eigenvector as the textbook method, regardless of how anyone's published files were built.
2. **Deployed parity.** hidalgo must reproduce TradeWeave's published `eci_rankings` / `pci_rankings` ordering on the HS92 core catalog (Spearman > 0.99999), and its proximity must match the published `product_proximity` values to better than 1e-3. This is the layer that shows it can drop into the live pipeline without changing published results. (Core universe only, for the hybrid-PCI reason explained above.)
3. **Rust-side unit tests.** Hand-checked small examples for RCA edge cases, binarization, proximity, and density, plus an independent dense eigensolver (cyclic Jacobi) used as an oracle: on small matrices, the fast solver and the brute-force solver must agree on the second eigenvector's direction.

**Suite status on 2026-09-07: 4 passed, 1 failed.** Measured values from that run, on year 2024 of the TradeWeave parquet:

| Check | Measured | Threshold | Result |
|---|---|---|---|
| Algorithm: ECI vs `np.linalg.eig` | Spearman 0.9999989604 | > 0.9999999 | fails |
| Algorithm: PCI vs `np.linalg.eig` | not reached | > 0.9999999 | not run |
| Deployed: ECI vs published, n=226 | Spearman 0.99999896 | > 0.99999 | passes |
| Deployed: PCI vs published, n=4,762 | Spearman 0.99999999 | > 0.99999 | passes |
| Deployed: proximity, 3,005,162 pairs | max abs diff 5.00e-05 | < 1e-3 | passes |

**The open issue, stated plainly.** `test_algorithm_matches_numpy_eig` fails. hidalgo's ECI and the NumPy full-eigendecomposition ECI correlate at 0.9999989604, just under the test's own 0.9999999 bar. This is a disagreement in the ordering of a small number of adjacent countries, not a sign flip or a structurally different ranking, and the same comparison against TradeWeave's published ECI clears its looser 0.99999 bar. It is nonetheless a real failure of the repository's hardest correctness gate, and it is open: the cause has not been diagnosed, and it is not known whether the residual comes from the power iteration's convergence tolerance, from near-degenerate eigenvalues in the 2024 country matrix, or from the input data having changed since the threshold was set. An earlier version of this README reported this correlation as 1.0; that figure is superseded. The PCI half of the same test never executes, because the ECI assertion aborts the test first, so hidalgo's PCI has no current provenance-independent check. Until this is resolved, treat the deployed-parity layer, not the algorithm layer, as the evidence that the library is safe to use, and do not read the algorithm proof as passing.

The parity tests require the TradeWeave parquet data locally and skip themselves cleanly if it is absent. The pure-Python smoke tests need no external data and pass (2 passed, 2026-09-07).

## Installing and using it

Prerequisites, one-time (commands from the repo's implementation plan):

```bash
# the Rust compiler toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# maturin, the Rust-to-Python build tool, installed via uv
uv tool install maturin
```

Build and install into a project environment (run inside the `hidalgo` folder):

```bash
uv venv && source .venv/bin/activate
uv pip install numpy
maturin develop --release
```

`uv venv` creates a virtual environment (a private, disposable Python installation for this project, so nothing touches your system Python). `maturin develop --release` compiles the Rust with full optimizations and installs the result into that environment. Requires Python 3.9 or newer.

Then, from Python:

```python
import numpy as np, hidalgo

out = hidalgo.bundle_from_rca(rca_matrix, threshold=1.0)
out["eci"], out["pci"], out["proximity"], out["density"]
out["kept_countries"], out["kept_products"]  # indices mapping back to your rows/cols
```

`bundle_from_exports(exports, ...)` computes RCA for you first; `bundle_from_m(m, ...)` takes a pre-binarized matrix; `rca(exports)` returns just the RCA matrix. The package exports exactly these four names (`python/hidalgo/__init__.py`). Every bundle function accepts `max_iters` (default 100000) and `tol` (default 1e-12) for the eigenvector iteration; `bundle_from_exports` and `bundle_from_rca` also accept `threshold` (default 1.0), while `bundle_from_m` does not, because its input is already binary. Defaults verified against `crates/hidalgo-py/src/lib.rs` and `python/hidalgo/__init__.pyi` (2026-09-07). All three raise a `ValueError` ("complexity undefined for input") when fewer than 3 diversified countries or 3 exported products remain, or when the eigenvector is constant (zero variance).

To run the test suite and benchmark:

```bash
uv pip install scipy pandas duckdb pytest
pytest tests/
python tests/bench_compare.py   # needs the TradeWeave parquet, see Performance
```

## Repository map

```
hidalgo/
  Cargo.toml                    # Rust workspace manifest (lists the two crates)
  rust-toolchain.toml           # pins the stable Rust compiler channel
  pyproject.toml                # Python packaging: maturin build backend, deps
  crates/
    hidalgo-core/               # the math, pure Rust, no Python anywhere
      src/matrix.rs             #   dense matrix type + parallel matrix-vector kernels
      src/rca.rs                #   Balassa RCA and binarization to M
      src/eig.rs                #   matrix-free shifted, deflated power iteration
      src/complexity.rs         #   ECI/PCI: eigenvector -> z-score -> sign rule
      src/proximity.rs          #   product proximity and density
      src/testutil.rs           #   test-only Jacobi eigensolver (the oracle)
      src/lib.rs                #   bundle entry points tying the pipeline together
    hidalgo-py/                 # the Python bridge (PyO3), builds to hidalgo._hidalgo
      src/lib.rs                #   NumPy arrays in, dict of NumPy arrays out
  python/hidalgo/               # the Python package a user imports
    __init__.py                 #   friendly names for the bridge functions
    __init__.pyi                #   type stubs (editor autocomplete/signatures)
  tests/
    test_smoke.py               # quick no-data sanity checks
    test_parity.py              # the correctness gate vs NumPy and TradeWeave parquet
    bench_compare.py            # the wall-clock benchmark script
    conftest.py                 # locates the TradeWeave parquet tree, skips if absent
  docs/superpowers/
    specs/2026-06-02-hidalgo-design.md   # design spec (math reference, decisions)
    plans/2026-06-02-hidalgo.md          # step-by-step implementation plan
```

## Status and roadmap

Status, from repo evidence as of 2026-09-07:

- Version 0.1.0 in all three manifests (`crates/hidalgo-core/Cargo.toml`, `crates/hidalgo-py/Cargo.toml`, `pyproject.toml`). MIT is declared in the workspace manifest, but there is still no standalone LICENSE file in the repository, which for a public repo is a gap worth closing.
- All eight Rust source files named in the design spec exist, the compiled extension is built and importable, and the smoke tests pass.
- One test is failing: `test_algorithm_matches_numpy_eig`, described under Correctness. It is the repository's only open correctness issue and it is undiagnosed.
- Distribution is local-only by design: v1 explicitly does not publish to PyPI or crates.io. Consumers install with `maturin develop` or a locally built wheel.
- The core crate has exactly one runtime dependency, `rayon` 1.10 (`crates/hidalgo-core/Cargo.toml`); `approx` and `rand` are dev-dependencies only. Python requires 3.9 or newer and NumPy 1.24 or newer (`pyproject.toml`).

Declared v1 non-goals (from the design spec): no command-line binary, no internal multi-year loop, no streaming or incremental updates, no GPU, no hand-rolled SIMD, no services/green/EBOPS-specific variants (callers pass whatever country x product matrix they like).

Roadmap, per the implementation plan's final task: wire `hidalgo` into TradeWeave's ingestion scripts (swap the `np.linalg.eig` block for `hidalgo.bundle_from_m`, keep the parquet outputs identical), gated on the parity tests staying green, and later reuse from BDPolicyLab. That wiring happens in the TradeWeave repository, not here; whether it has landed is not tracked in this repo.

## Glossary

- **BACI**: CEPII's cleaned, reconciled version of world bilateral trade data at the HS6 product level; TradeWeave's underlying trade dataset.
- **Binding (Python binding)**: a bridge layer that lets Python call functions written in another language as if they were native Python functions.
- **Cargo**: Rust's build tool and package manager, the counterpart of Python's pip/uv.
- **Crate**: Rust's word for a code library or package. This repo contains two: the math core and the Python bridge.
- **Deflation**: removing a known eigenvector from a trial vector during iteration so the method converges to the next eigenvector instead. How this solver reaches the second eigenvector.
- **Density**: for a country and a product, the proximity-weighted share of that product's neighborhood already present in the country's export basket; a measure of how "nearby" a currently unexported product is.
- **Diversity**: the number of products a country exports with RCA at or above the threshold (a row sum of M).
- **DuckDB**: an in-process analytical database used by the tests to read parquet files with SQL; no server needed.
- **ECI (Economic Complexity Index)**: a country ranking derived from the second eigenvector of the country reflections matrix; standardized to mean 0, standard deviation 1.
- **Eigenvector (and eigenvalue)**: a vector that a matrix maps onto a scaled copy of itself; the scale factor is the eigenvalue. Complexity indices are defined as particular eigenvectors of the reflections matrices.
- **HS6 / HS92**: the Harmonized System, the international product classification for trade; HS6 is its six-digit (finest widely comparable) level, and HS92, HS96, ..., HS22 are its successive revisions.
- **Matrix-free**: an algorithm that never builds the big matrix it analyzes, only applies it to vectors through cheaper intermediate operations. Saves both memory and time here.
- **Maturin**: the build tool that compiles the Rust code and installs it into a Python environment as an importable module, in one command.
- **Method of reflections**: Hidalgo and Hausmann's original iterative averaging of diversity and ubiquity; converges to the same answer as the eigenvector formulation, which is what this repo implements directly.
- **NumPy**: Python's standard array and linear-algebra library; the format in which all data enters and leaves `hidalgo`.
- **Parity test**: a test asserting that a new implementation reproduces a reference implementation's results on the same input; this repo's correctness gate.
- **Parquet**: a compressed columnar file format for tabular data; how TradeWeave stores its computed metrics.
- **PCI (Product Complexity Index)**: a product ranking derived from the second eigenvector of the product reflections matrix; routinely negative for simple products, and negatives are never filtered.
- **Power iteration**: repeatedly multiplying a trial vector by a matrix and rescaling, which converges to the dominant eigenvector; the classical cheap eigenvector method.
- **Proximity**: the Hidalgo et al. (2007) product-space measure: for two products, the minimum of the two conditional probabilities of co-export, computed as co-export count over the larger ubiquity.
- **PyO3**: the Rust library implementing the Python binding layer; what makes `import hidalgo` work.
- **pytest**: Python's standard test runner; `pytest tests/` executes the correctness suite.
- **rayon**: a Rust library that parallelizes loops across CPU cores; the math core's only runtime dependency.
- **RCA (revealed comparative advantage, Balassa index)**: a country's export share in a product divided by the world export share of that product; values at or above 1 mark meaningful specialization.
- **Reflections matrix**: the country-by-country (or product-by-product) matrix built from M and the diversity/ubiquity counts whose second eigenvector defines ECI (or PCI).
- **Rust**: a compiled systems programming language combining C/C++-class speed with compile-time memory safety; the language of the math core.
- **Spearman rank correlation**: correlation between the *ranks* of two variables; equals 1.0 when two methods order every observation identically, the metric used by the parity tests.
- **Ubiquity**: the number of countries exporting a product with RCA at or above the threshold (a column sum of M).
- **uv**: a fast Python package and environment manager; used here instead of pip.
- **Virtual environment**: a private, project-local Python installation, so a project's packages never interfere with the system or with other projects.
- **Wheel**: Python's binary package format; `maturin build --release` produces one so the compiled library can be installed elsewhere without recompiling.
- **z-score**: standardizing a variable by subtracting its mean and dividing by its standard deviation; both ECI and PCI are reported as z-scores.

## Author and license

Written and maintained by Md Deluair Hossen, PhD, an international trade economist, as the computational kernel behind the TradeWeave trade-analytics platform.

MIT, declared in the workspace `Cargo.toml`. There is no standalone LICENSE file in the repository yet.
