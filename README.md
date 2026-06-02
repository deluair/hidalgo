# hidalgo

Fast Rust economic-complexity kernels (ECI, PCI, product proximity, density) with
Python bindings. Reproduces the Hidalgo-Hausmann eigenvalue method used by
TradeWeave, verified against its published rankings.

## Use

```python
import numpy as np, hidalgo
out = hidalgo.bundle_from_rca(rca_matrix, threshold=1.0)
out["eci"], out["pci"], out["proximity"], out["density"]
out["kept_countries"], out["kept_products"]  # indices mapping back to your rows/cols
```

`bundle_from_exports(exports, ...)` computes RCA for you first; `bundle_from_m(m, ...)`
takes a pre-binarized matrix.

## Build

```bash
uv venv && source .venv/bin/activate
uv pip install numpy
maturin develop --release
```

## Performance

Benchmarked on the live TradeWeave RCA data (year 2024, 226 countries x 6429
products, Apple Silicon M5). The NumPy path (two `np.linalg.eig` calls on the
reflections matrices plus proximity) takes 30732 ms; hidalgo completes ECI, PCI,
proximity, and density in 242 ms, a 126.9x speedup. The gain comes from skipping
`np.linalg.eig`, which computes the full spectrum of the roughly 6000x6000 product
reflections matrix just to extract one eigenvector. hidalgo uses matrix-free
deflated power iteration instead, which needs only matrix-vector products and
converges in tens of iterations. Reproduce with:
`python tests/bench_compare.py` (set `HIDALGO_TRADEWEAVE_DATA` if your parquet
tree is not at `~/tradeweave/data/parquet`).

## Correctness

Verified by `pytest tests/`:
- Algorithm proof: hidalgo agrees with NumPy `np.linalg.eig` on the same matrix
  (Spearman = 1.0 for ECI, 0.99999999 for PCI).
- Deployed parity: reproduces TradeWeave's published `eci_rankings` /
  `pci_rankings` order on the HS92 core catalog (Spearman > 0.99999), and
  proximity matches `product_proximity` to ~5e-5.

Provenance note: `pci_rankings.parquet` is a hybrid. The HS92 core (the HS6 codes
in `products.parquet`) is scored by the eigenvector method (what hidalgo
implements). The extended HS96-HS22 products are scored by a different formula in
TradeWeave's `data/compute_ext_rankings.py` ("PCI = mean ECI of exporters with
RCA >= 1"), so parity is checked on the core universe only.

## Layout

- `crates/hidalgo-core` pure-Rust math (only `rayon`), matrix-free second
  eigenvector via shifted deflated power iteration.
- `crates/hidalgo-py` PyO3 bindings, built with maturin.
