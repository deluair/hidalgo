"""Parity gate: hidalgo vs TradeWeave's published economic-complexity data.

Two layers of verification:

1. ALGORITHM PROOF (provenance-independent, the hard gate): build the binary M
   from the same RCA matrix, compute ECI/PCI two ways -- hidalgo's matrix-free
   power iteration and NumPy's full `np.linalg.eig` -- and require identical
   ranking. This proves the Rust math is correct regardless of how the published
   parquet was built. Expect Spearman ~ 1.0.

2. DEPLOYED PARITY: hidalgo must reproduce the published `eci_rankings` /
   `pci_rankings` ordering. IMPORTANT provenance note: `pci_rankings.parquet` is
   a HYBRID. The HS92 core catalog (the ~5,022 HS6 codes in `products.parquet`)
   is scored by the eigenvector method -- which is what hidalgo implements. The
   ~1,400 extended products (HS96-HS22 revisions) are scored by a DIFFERENT
   formula in `data/compute_ext_rankings.py`: "PCI = mean ECI of countries with
   RCA >= 1", then inserted into the same table. hidalgo neither reproduces nor
   should reproduce that add-on, and including those products also perturbs the
   core eigenvector. So deployed parity is checked on the HS92 core universe
   only. Expect Spearman > 0.99999.

3. Proximity: hidalgo's phi must match `product_proximity.parquet` on overlapping
   pairs to rounding.
"""

import numpy as np
import duckdb
from scipy.stats import spearmanr
import hidalgo


def _core_products(data_dir):
    """The HS92 core catalog = HS6 codes in products.parquet (the eigenvector
    universe). product code column is named `code` in that table."""
    con = duckdb.connect()
    cols = [c[0] for c in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{data_dir}/products.parquet')").fetchall()]
    col = "code" if "code" in cols else "product_code"
    codes = con.execute(
        f"SELECT DISTINCT {col} AS c FROM read_parquet('{data_dir}/products.parquet')"
    ).df()["c"].astype(str)
    con.close()
    return set(c for c in codes if len(c) == 6)


def _latest_common_year(data_dir):
    con = duckdb.connect()
    rca_years = {r[0] for r in con.execute(
        f"SELECT DISTINCT year FROM read_parquet('{data_dir}/rca_matrix/**/*.parquet')").fetchall()}
    eci_years = {r[0] for r in con.execute(
        f"SELECT DISTINCT year FROM read_parquet('{data_dir}/eci_rankings.parquet')").fetchall()}
    pci_years = {r[0] for r in con.execute(
        f"SELECT DISTINCT year FROM read_parquet('{data_dir}/pci_rankings.parquet')").fetchall()}
    con.close()
    common = rca_years & eci_years & pci_years
    assert common, "no overlapping year across rca_matrix/eci_rankings/pci_rankings"
    return max(common)


def _build(data_dir, year, core):
    con = duckdb.connect()
    rca = con.execute(
        f"""SELECT country_code, product_code, rca
            FROM read_parquet('{data_dir}/rca_matrix/**/*.parquet')
            WHERE year = {year} AND rca IS NOT NULL""").df()
    eci_ref = con.execute(
        f"""SELECT country_code, eci FROM read_parquet('{data_dir}/eci_rankings.parquet')
            WHERE year = {year}""").df()
    pci_ref = con.execute(
        f"""SELECT product_code, pci FROM read_parquet('{data_dir}/pci_rankings.parquet')
            WHERE year = {year}""").df()
    con.close()
    rca = rca[rca["product_code"].isin(core)]
    countries = sorted(rca["country_code"].unique())
    products = sorted(rca["product_code"].unique())
    ci = {c: i for i, c in enumerate(countries)}
    pj = {p: j for j, p in enumerate(products)}
    mat = np.zeros((len(countries), len(products)), dtype=np.float64)
    for cc, pc, v in rca.itertuples(index=False):
        mat[ci[cc], pj[pc]] = v
    return mat, countries, products, eci_ref, pci_ref


def test_algorithm_matches_numpy_eig(data_dir):
    """Provenance-independent: hidalgo must agree with NumPy's full eig on the
    SAME matrix. This is the definitive correctness proof for the Rust solver."""
    year = _latest_common_year(data_dir)
    core = _core_products(data_dir)
    mat, countries, products, _, _ = _build(data_dir, year, core)
    out = hidalgo.bundle_from_rca(mat, threshold=1.0)
    assert out["eci_converged"] and out["pci_converged"]

    B = (mat >= 1.0).astype(float)
    kc, kp = B.sum(1), B.sum(0)
    keepc, keepp = kc > 0, kp > 0
    Bk = B[keepc][:, keepp]
    kck, kpp = kc[keepc], kp[keepp]

    mt_c = (Bk / kck[:, None]) @ (Bk / kpp[None, :]).T
    w, v = np.linalg.eig(mt_c)
    np_eci = v[:, np.argsort(w.real)[::-1][1]].real
    if spearmanr(np_eci, kck)[0] < 0:
        np_eci = -np_eci
    eci = dict(zip([countries[i] for i in out["kept_countries"]], out["eci"]))
    hid_eci = np.array([eci[countries[i]] for i in np.where(keepc)[0]])
    rho_e, _ = spearmanr(hid_eci, np_eci)
    print(f"[algorithm] year={year} ECI hidalgo-vs-numpy spearman={rho_e:.10f}")
    assert rho_e > 0.9999999, f"ECI vs numpy eig: {rho_e}"

    mt_p = (Bk / kpp[None, :]).T @ (Bk / kck[:, None])
    w2, v2 = np.linalg.eig(mt_p)
    np_pci = v2[:, np.argsort(w2.real)[::-1][1]].real
    if spearmanr(np_pci, kpp)[0] > 0:
        np_pci = -np_pci
    pci = dict(zip([products[j] for j in out["kept_products"]], out["pci"]))
    hid_pci = np.array([pci[products[j]] for j in np.where(keepp)[0]])
    rho_p, _ = spearmanr(hid_pci, np_pci)
    print(f"[algorithm] year={year} PCI hidalgo-vs-numpy spearman={rho_p:.10f}")
    assert rho_p > 0.9999999, f"PCI vs numpy eig: {rho_p}"


def test_deployed_parity_core_universe(data_dir):
    """hidalgo reproduces the published eigenvector-method rankings on the HS92
    core catalog. See module docstring for why extended products are excluded."""
    year = _latest_common_year(data_dir)
    core = _core_products(data_dir)
    mat, countries, products, eci_ref, pci_ref = _build(data_dir, year, core)
    out = hidalgo.bundle_from_rca(mat, threshold=1.0)

    eci = dict(zip([countries[i] for i in out["kept_countries"]], out["eci"]))
    pci = dict(zip([products[j] for j in out["kept_products"]], out["pci"]))

    e = eci_ref[eci_ref["country_code"].isin(eci)]
    ours = np.array([eci[c] for c in e["country_code"]])
    rho_e, _ = spearmanr(ours, e["eci"].to_numpy())
    print(f"[deployed] year={year} ECI n={len(e)} spearman={rho_e:.8f}")
    assert rho_e > 0.99999, f"ECI deployed parity: {rho_e}"

    p = pci_ref[pci_ref["product_code"].isin(pci)]
    ours_p = np.array([pci[x] for x in p["product_code"]])
    rho_p, _ = spearmanr(ours_p, p["pci"].to_numpy())
    print(f"[deployed] year={year} PCI n={len(p)} spearman={rho_p:.8f}")
    assert rho_p > 0.99999, f"PCI deployed parity: {rho_p}"


def test_proximity_parity(data_dir):
    year = _latest_common_year(data_dir)
    core = _core_products(data_dir)
    mat, countries, products, _, _ = _build(data_dir, year, core)
    pj = {p: j for j, p in enumerate(products)}
    out = hidalgo.bundle_from_rca(mat, threshold=1.0)
    phi = out["proximity"]
    con = duckdb.connect()
    ref = con.execute(
        f"""SELECT product_a, product_b, proximity
            FROM read_parquet('{data_dir}/product_proximity/**/*.parquet')
            WHERE year = {year}""").df()
    con.close()
    diffs = [abs(phi[pj[pa], pj[pb]] - pv)
             for pa, pb, pv in ref.itertuples(index=False)
             if pa in pj and pb in pj]
    md = max(diffs) if diffs else 0.0
    print(f"[deployed] year={year} proximity overlap={len(diffs)} max|diff|={md:.2e}")
    assert diffs, "no overlapping product pairs to compare"
    assert md < 1e-3, f"proximity disagreement max|diff|={md:.2e}"
