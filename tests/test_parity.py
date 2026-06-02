import numpy as np
import duckdb
from scipy.stats import spearmanr
import hidalgo


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


def _build_inputs(data_dir, year):
    con = duckdb.connect()
    rca = con.execute(
        f"""SELECT country_code, product_code, rca
            FROM read_parquet('{data_dir}/rca_matrix/**/*.parquet')
            WHERE year = {year} AND rca IS NOT NULL""").df()
    eci_ref = con.execute(
        f"""SELECT country_code, eci, eci_rank
            FROM read_parquet('{data_dir}/eci_rankings.parquet') WHERE year = {year}""").df()
    pci_ref = con.execute(
        f"""SELECT product_code, pci, pci_rank
            FROM read_parquet('{data_dir}/pci_rankings.parquet') WHERE year = {year}""").df()
    con.close()
    return rca, eci_ref, pci_ref


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std()


def _matrix(rca_df):
    countries = sorted(rca_df["country_code"].unique())
    products = sorted(rca_df["product_code"].unique())
    ci = {c: i for i, c in enumerate(countries)}
    pj = {p: j for j, p in enumerate(products)}
    mat = np.zeros((len(countries), len(products)), dtype=np.float64)
    for cc, pc, v in rca_df.itertuples(index=False):
        mat[ci[cc], pj[pc]] = v
    return mat, countries, products, pj


def test_eci_pci_parity(data_dir):
    year = _latest_common_year(data_dir)
    rca_df, eci_ref, pci_ref = _build_inputs(data_dir, year)
    rca_mat, countries, products, _ = _matrix(rca_df)
    out = hidalgo.bundle_from_rca(rca_mat, threshold=1.0)
    assert out["eci_converged"] and out["pci_converged"], "solver did not converge"

    kept_c = [countries[i] for i in out["kept_countries"]]
    kept_p = [products[j] for j in out["kept_products"]]
    eci = dict(zip(kept_c, out["eci"]))
    pci = dict(zip(kept_p, out["pci"]))

    eci_ref = eci_ref[eci_ref["country_code"].isin(eci)].copy()
    ours = np.array([eci[c] for c in eci_ref["country_code"]])
    ref = eci_ref["eci"].to_numpy()
    rho, _ = spearmanr(ours, ref)
    dz = np.max(np.abs(_zscore(ours) - _zscore(ref)))
    print(f"[parity] year={year} ECI n={len(ref)} spearman={rho:.8f} max|dz|={dz:.2e}")
    assert rho > 0.99999, f"ECI rank parity failed: spearman={rho:.6f} (year {year})"

    pci_ref = pci_ref[pci_ref["product_code"].isin(pci)].copy()
    ours_p = np.array([pci[p] for p in pci_ref["product_code"]])
    ref_p = pci_ref["pci"].to_numpy()
    rho_p, _ = spearmanr(ours_p, ref_p)
    dz_p = np.max(np.abs(_zscore(ours_p) - _zscore(ref_p)))
    print(f"[parity] year={year} PCI n={len(ref_p)} spearman={rho_p:.8f} max|dz|={dz_p:.2e}")
    assert rho_p > 0.99999, f"PCI rank parity failed: spearman={rho_p:.6f} (year {year})"


def test_proximity_parity(data_dir):
    year = _latest_common_year(data_dir)
    rca_df, _, _ = _build_inputs(data_dir, year)
    rca_mat, countries, products, pj = _matrix(rca_df)
    out = hidalgo.bundle_from_rca(rca_mat, threshold=1.0)
    phi = out["proximity"]
    con = duckdb.connect()
    ref = con.execute(
        f"""SELECT product_a, product_b, proximity
            FROM read_parquet('{data_dir}/product_proximity/**/*.parquet')
            WHERE year = {year}""").df()
    con.close()
    diffs = []
    for pa, pb, pv in ref.itertuples(index=False):
        if pa in pj and pb in pj:
            diffs.append(abs(phi[pj[pa], pj[pb]] - pv))
    md = max(diffs) if diffs else 0.0
    print(f"[parity] year={year} proximity overlap={len(diffs)} max|diff|={md:.2e}")
    assert diffs, "no overlapping product pairs to compare"
    assert md < 1e-3, f"proximity disagreement max|diff|={md:.2e}"
