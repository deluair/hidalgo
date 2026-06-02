"""Reproducible speed comparison: hidalgo vs the NumPy np.linalg.eig path that
TradeWeave's complexity ingestion uses today. Run from the project venv with the
TradeWeave parquet available (set HIDALGO_TRADEWEAVE_DATA or default
~/tradeweave/data/parquet)."""
import time, os, pathlib, numpy as np, duckdb, hidalgo

dd = pathlib.Path(os.environ.get("HIDALGO_TRADEWEAVE_DATA",
                                 str(pathlib.Path.home() / "tradeweave" / "data" / "parquet")))
con = duckdb.connect()
year = con.execute(f"SELECT max(year) FROM read_parquet('{dd}/rca_matrix/**/*.parquet')").fetchone()[0]
rca_df = con.execute(
    f"SELECT country_code, product_code, rca FROM read_parquet('{dd}/rca_matrix/**/*.parquet') "
    f"WHERE year={year} AND rca IS NOT NULL").df()
con.close()
countries = sorted(rca_df.country_code.unique()); products = sorted(rca_df.product_code.unique())
ci = {c: i for i, c in enumerate(countries)}; pj = {p: j for j, p in enumerate(products)}
rca_mat = np.zeros((len(countries), len(products)))
for cc, pc, v in rca_df.itertuples(index=False):
    rca_mat[ci[cc], pj[pc]] = v
print(f"year={year}  matrix = {rca_mat.shape[0]} countries x {rca_mat.shape[1]} products")


def numpy_path(rca_mat):
    M = (rca_mat >= 1.0).astype(float)
    kc = M.sum(1); kp = M.sum(0)
    keepc = kc > 0; keepp = kp > 0
    Mk = M[keepc][:, keepp]; kck = kc[keepc]; kpp = kp[keepp]
    wc, _ = np.linalg.eig((Mk / kck[:, None]) @ (Mk / kpp[None, :]).T)          # ECI
    wp, _ = np.linalg.eig((Mk / kpp[None, :]).T @ (Mk / kck[:, None]))          # PCI (big)
    co = M.T @ M; den = np.maximum.outer(kp, kp)
    phi = np.divide(co, den, out=np.zeros_like(co), where=den > 0)              # proximity
    return wc, wp, phi


t = time.perf_counter(); numpy_path(rca_mat); t_np = time.perf_counter() - t
t = time.perf_counter(); out = hidalgo.bundle_from_rca(rca_mat, 1.0); t_rs = time.perf_counter() - t
print(f"numpy  (ECI eig + PCI eig + proximity)         : {t_np*1000:9.1f} ms")
print(f"hidalgo (ECI + PCI + proximity + density)      : {t_rs*1000:9.1f} ms")
print(f"speedup: {t_np/t_rs:.1f}x  (hidalgo also computes density; numpy path does not)")
