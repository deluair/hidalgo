# hidalgo: design spec

Date: 2026-06-02
Status: approved (design), pending implementation plan

## What this is

`hidalgo` is a standalone Rust crate with Python bindings that computes the
Hidalgo-Hausmann economic complexity bundle on a country-product export matrix:
RCA, the binary specialization matrix M, ECI, PCI, product proximity, and
product density. It exists to replace the slow NumPy/`np.linalg.eig` path used in
TradeWeave's local data-ingestion scripts with a fast, parallel, bare-metal Rust
implementation, while reproducing the existing math bit-for-bit (in the sense
defined by the correctness gate below).

It is the 7th top-level repo (`~/hidalgo`), separate from the canonical 6. It is
consumed as a built wheel by TradeWeave and (later) BDPolicyLab, both of which
run the same complexity math in **local** ingestion scripts. The VPS only serves
precomputed parquet, so `hidalgo` never runs on the VPS and never needs a Rust
toolchain there.

## Why

The country side of the reflections method is tiny (238x238). The cost is on the
product side: PCI is the second eigenvector of a 5,022x5,022 reflections matrix,
and `np.linalg.eig` computes the full 5,022-value spectrum just to extract one
eigenvector. Proximity is a 5,022x5,022 dense co-occurrence operation. These are
the bottlenecks in the local data builds, and they are exactly the kind of
cache-bound, SIMD-friendly, embarrassingly-parallel work where Rust + `faer` +
`rayon` wins.

## Reference math (must match)

The canonical Python reference is TradeWeave's
`data/services_complexity_ingest.py::compute_eci` (eigenvalue / Hidalgo-Hausmann
method) and `compute_rca`. The goods pipeline uses the same construction on the
country x HS6 matrix. The crate reproduces:

- **RCA (Balassa)**, per year, on export values `x_cp`:
  `RCA_cp = (x_cp / sum_p x_cp) / (sum_c x_cp / sum_cp x_cp)`.
  Rows with zero country total are skipped; `inf`/`0/0` become NaN and are
  treated as "no RCA" (effectively 0 for the binary step), matching `compute_rca`.
- **Binary M**: `M_cp = 1.0 if RCA_cp >= 1.0 else 0.0`.
- **Diversity / ubiquity**: `k_c = sum_p M_cp`, `k_p = sum_c M_cp`.
  Drop countries with `k_c = 0`. If fewer than 3 countries remain, skip the year
  (mirrors `keep.sum() < 3`).
- **ECI**: second eigenvector (by descending real eigenvalue) of the country
  reflections matrix `Mtilde_c = (M / k_c) @ (M / k_p).T`, where division by a
  zero `k_p` yields 0 in that term (`np.nan_to_num`). Then z-standardize:
  `eci = (v - mean(v)) / std(v)`; if `corr(eci, k_c) < 0`, negate. Skip if
  `std(v) == 0`.
- **PCI**: same construction on the product reflections matrix
  `Mtilde_p = (M / k_p).T @ (M / k_c)` (5,022x5,022), second eigenvector,
  z-standardized: `pci = (v - mean(v)) / std(v)`. **Sign rule (explicit):** orient
  PCI so it correlates *negatively* with ubiquity, i.e. if `corr(pci, k_p) > 0`,
  negate (more complex products are less ubiquitous). This is the Atlas
  convention and is independent of ECI's sign. Negatives are kept; PCI is
  routinely negative, never filtered. The authoritative parity target is the
  existing `pci_rankings.parquet` (see gate below); if its producing script is
  found during implementation, match it exactly and reconcile any sign difference
  there.
- **Proximity** (Hidalgo et al. 2007):
  `phi_pp' = M_co[p,p'] / max(k_p, k_p')` where `M_co = M.T @ M` (co-export
  counts). Symmetric, diagonal = 1.
- **Density**:
  `density_cp = (M @ phi)_cp / sum_p' phi_pp'` (row-normalized by proximity sums).

All inputs/outputs follow TradeWeave conventions: ISO3 uppercase upstream, values
in thousands USD upstream (the crate is unit-agnostic; it takes a numeric
matrix), RCA dimensionless, PCI negatives preserved.

## Approach (chosen: A)

Fast path, not a literal port of `np.linalg.eig`:

- **Second eigenvector via deflated power iteration.** The reflections matrix has
  a trivial Perron eigenvalue of 1; the meaningful complexity vector is the
  *second* eigenvector. Deflate the known leading eigenvector, then power-iterate
  for the second. O(iters * n^2) instead of O(n^3) for a full spectrum we discard.
  Because the downstream steps z-standardize and rank, only the eigenvector's
  *direction* matters, not its scale, so power iteration that converges to the
  same direction yields identical z-scores and ranks (sign fixed via the
  diversity-correlation rule).
- **Proximity** = a single `M.T @ M` matmul (`faer`, SIMD microkernels) followed
  by elementwise `min`/normalize, parallelized with `rayon`.
- **Density** = `M @ phi` then row-normalize.
- **Full dense eigendecomposition (approach B) is retained only as a test
  oracle**, not the production path.

## Architecture

Two layers, two crates in one workspace:

```
hidalgo/
  Cargo.toml                 # workspace
  crates/
    hidalgo-core/            # pure Rust, no Python. The math.
      src/lib.rs
      src/rca.rs             # rca(), binarize()
      src/complexity.rs      # eci_pci() via deflated power iteration
      src/proximity.rs       # proximity(), density()
      src/eig.rs             # power iteration + deflation; full-eig oracle (cfg(test))
    hidalgo-py/              # thin PyO3 wrapper, maturin-built
      src/lib.rs             # numpy zero-copy in/out -> calls hidalgo-core
  python/
    hidalgo/__init__.py      # re-export, typing stubs (.pyi)
  tests/
    test_parity.py           # parity vs TradeWeave Python on real BACI
  pyproject.toml             # maturin build backend
  docs/superpowers/specs/
```

- **`hidalgo-core`**: operates on `faer` matrices. Public functions:
  `rca(exports: &Mat<f64>) -> Mat<f64>`,
  `binarize(rca: &Mat<f64>, threshold: f64) -> Mat<f64>`,
  `eci_pci(m: &Mat<f64>) -> ComplexityResult` (returns eci, pci, diversity,
  ubiquity, kept-row mask),
  `proximity(m: &Mat<f64>) -> Mat<f64>`,
  `density(m: &Mat<f64>, phi: &Mat<f64>) -> Mat<f64>`.
  Each is independently unit-testable with no Python in the loop.
- **`hidalgo-py`**: PyO3 + the `numpy` crate. Accepts `PyReadonlyArray2<f64>`,
  returns `numpy` arrays, zero-copy where shapes allow. One module function per
  core function, plus a convenience `complexity_bundle(exports, threshold=1.0)`
  that runs the whole pipeline and returns a dict of arrays.

## Data flow

```
exports (C x P, f64)  --rca-->  RCA (C x P)
                                  |
                       --binarize(>=1.0)-->  M (C x P)
                                  |
        +-------------------------+----------------------+
        |                         |                      |
     eci_pci(M)              proximity(M)            (M, phi)
        |                         |                      |
   eci (Ck), pci (P)          phi (P x P) ---------> density (Ck x P)
```

Years are handled by the caller (loop per year, as the Python does today); the
crate operates on one year's matrix at a time. No internal year loop in v1.

## Build / tooling

- Workspace built with `maturin` under `uv`. `maturin develop` for local editable
  install into the consumer's `uv` venv; `maturin build --release` for a wheel.
- `faer` for linear algebra, `rayon` for parallelism, `numpy` + `pyo3` for
  bindings, `ndarray` only if needed at the boundary.
- Rust 2021 edition, `cargo test` for core, `cargo bench` (criterion) for the
  matmul/eigenvector kernels.
- Consumers add `hidalgo` as a dev/build dependency in their `pyproject` and call
  it from the local `data/*.py` ingestion scripts. No production-runtime or VPS
  dependency.

## Correctness gate (non-negotiable)

Nothing ships until all pass:

1. **Python parity (`tests/test_parity.py`)**, two references:
   - **ECI** vs the live `compute_eci` in
     `data/services_complexity_ingest.py` run on the same input matrix: Spearman
     rho = 1.0 on ECI ranks and max |delta z| < 1e-6.
   - **PCI** vs the existing `pci_rankings.parquet` for a recent goods year
     (reconstruct the same C x P goods M_cp from BACI / `rca_matrix`): Spearman
     rho = 1.0 on PCI ranks and max |delta z| < 1e-6 after applying the explicit
     sign rule above. If the script that produced `pci_rankings.parquet` is
     located, match it directly instead.
   - **Proximity / density** vs a NumPy reference computed inline in the test
     from the same M (and, if `product_proximity.parquet` covers the same
     product set, cross-checked against it): max abs elementwise difference
     < 1e-9.
2. **Rust oracle test**: on small random binary matrices, deflated power
   iteration (A) agrees with full dense eigendecomposition (B) on the second
   eigenvector direction to < 1e-9 (after sign/normalization).
3. **Unit tests** for RCA edge cases (zero rows, all-zero products, NaN/inf
   handling) matching `compute_rca` behavior exactly.
4. **Determinism**: identical input -> identical output across runs and thread
   counts (rayon reductions ordered or numerically stable).

Benchmark target (informational, not a gate): proximity + PCI on the full
238x5,022 goods matrix materially faster than the current NumPy path; record
actual speedup in the README from `criterion` + a wall-clock comparison.

## Non-goals (v1)

- No CLI binary.
- No services / green / EBOPS-specific variants (just the generic C x P matrix;
  the services script can pass its own matrix in).
- No internal multi-year loop, no streaming/incremental updates.
- No GPU, no hand-rolled SIMD (faer's microkernels suffice).
- No publishing to crates.io / PyPI in v1 (local wheel only).

## Risks

- **Power iteration not converging to the exact `np.linalg.eig` eigenvector** when
  the 2nd and 3rd eigenvalues are near-degenerate. Mitigation: the oracle test
  catches it on small matrices; on the real matrix, the parity test on ranks/z is
  the backstop. If a real year is degenerate, fall back to the full-eig path for
  that year (B is already implemented as the oracle).
- **Sign convention drift** between ECI and PCI. Mitigation: replicate the
  diversity-correlation sign rule exactly and assert it in parity.
- **Float reproducibility across thread counts.** Mitigation: deterministic
  reduction order or single-threaded final reduction in the eigenvector step.

## Consumers (wiring, post-v1)

- TradeWeave: `data/services_complexity_ingest.py` and the goods complexity build
  swap their `np.linalg.eig` block for `hidalgo.complexity_bundle(...)`, gated
  behind the parity test. Parquet outputs unchanged.
- BDPolicyLab: same crate reused if/when it computes complexity locally.
