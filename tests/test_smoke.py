import numpy as np
import hidalgo


def test_nested_matrix_bundle():
    c = p = 6
    m = np.zeros((c, p), dtype=np.float64)
    for i in range(c):
        for j in range(i + 1):
            m[i, j] = 1.0
    out = hidalgo.bundle_from_m(m)
    assert out["eci_converged"] and out["pci_converged"]
    eci = out["eci"]
    div = out["diversity"]
    from scipy.stats import spearmanr
    rho, _ = spearmanr(eci, div)
    assert rho > 0.99
    assert out["proximity"].shape == (p, p)
    assert out["density"].shape == (c, p)
    assert np.allclose(out["proximity"], out["proximity"].T, atol=1e-12)
    assert np.allclose(np.diag(out["proximity"]), 1.0, atol=1e-12)


def test_rca_matches_hand_calc():
    ex = np.array([[10.0, 0.0], [0.0, 10.0]])
    r = hidalgo.rca(ex)
    assert abs(r[0, 0] - 2.0) < 1e-12
    assert abs(r[0, 1]) < 1e-12
