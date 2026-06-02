//! ECI and PCI from the binary specialization matrix M.

use crate::eig::second_eigenvector;
use crate::matrix::Matrix;

pub struct ComplexityResult {
    pub eci: Vec<f64>,            // len = kept countries
    pub pci: Vec<f64>,            // len = kept products (k_p > 0)
    pub diversity: Vec<f64>,      // k_c for kept countries
    pub ubiquity: Vec<f64>,       // k_p for ALL products (len = m.cols)
    pub kept_countries: Vec<usize>,
    pub kept_products: Vec<usize>,
    pub eci_converged: bool,
    pub pci_converged: bool,
}

/// Population standardization (ddof=0), matching NumPy `.std()`.
fn zscore(v: &[f64]) -> Option<Vec<f64>> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd == 0.0 {
        return None;
    }
    Some(v.iter().map(|x| (x - mean) / sd).collect())
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

/// Compute ECI and PCI from the binary specialization matrix M.
/// Returns None if fewer than 3 countries (or products) have positive degree,
/// or if a standardization has zero variance.
pub fn eci_pci(m: &Matrix, max_iters: usize, tol: f64) -> Option<ComplexityResult> {
    let (c, p) = (m.rows, m.cols);
    let kc: Vec<f64> = (0..c).map(|i| m.row(i).iter().sum()).collect();
    let mut kp = vec![0.0; p];
    for i in 0..c {
        let r = m.row(i);
        for j in 0..p {
            kp[j] += r[j];
        }
    }

    let kept_countries: Vec<usize> = (0..c).filter(|&i| kc[i] > 0.0).collect();
    if kept_countries.len() < 3 {
        return None;
    }
    let kept_products: Vec<usize> = (0..p).filter(|&j| kp[j] > 0.0).collect();
    if kept_products.len() < 3 {
        return None;
    }

    let ck = kept_countries.len();
    let mut mk = Matrix::zeros(ck, p);
    for (ni, &oi) in kept_countries.iter().enumerate() {
        mk.data[ni * p..(ni + 1) * p].copy_from_slice(m.row(oi));
    }
    let kc_k: Vec<f64> = kept_countries.iter().map(|&i| kc[i]).collect();

    // ---- ECI (country side): Mtilde_c = Dc^-1 Mk Dp^-1 Mk^T ----
    let eci_eig = second_eigenvector(
        ck,
        &kc_k,
        &kp,
        |x| mk.matvec(x),
        |x| mk.matvec_transpose(x),
        max_iters,
        tol,
    );
    let mut eci = zscore(&eci_eig.raw)?;
    if corr(&eci, &kc_k) < 0.0 {
        for x in eci.iter_mut() {
            *x = -*x;
        }
    }

    // ---- PCI (product side): restrict to products with k_p > 0 ----
    let pk = kept_products.len();
    let mut mkp = Matrix::zeros(ck, pk);
    for ni in 0..ck {
        for (nj, &oj) in kept_products.iter().enumerate() {
            mkp.set(ni, nj, mk.get(ni, oj));
        }
    }
    let kp_pos: Vec<f64> = kept_products.iter().map(|&j| kp[j]).collect();
    // Mtilde_p = Dp^-1 Mkp^T Dc^-1 Mkp ; A = Mkp^T : (ck)->(pk), A^T = Mkp : (pk)->(ck)
    let pci_eig = second_eigenvector(
        pk,
        &kp_pos,
        &kc_k,
        |x| mkp.matvec_transpose(x),
        |x| mkp.matvec(x),
        max_iters,
        tol,
    );
    let mut pci = zscore(&pci_eig.raw)?;
    if corr(&pci, &kp_pos) > 0.0 {
        for x in pci.iter_mut() {
            *x = -*x;
        }
    }

    Some(ComplexityResult {
        eci,
        pci,
        diversity: kc_k,
        ubiquity: kp,
        kept_countries,
        kept_products,
        eci_converged: eci_eig.converged,
        pci_converged: pci_eig.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spearman_sign(a: &[f64], b: &[f64]) -> f64 {
        let pa = rank(a);
        let pb = rank(b);
        let n = a.len() as f64;
        let mut d2 = 0.0;
        for i in 0..a.len() {
            d2 += (pa[i] - pb[i]).powi(2);
        }
        1.0 - 6.0 * d2 / (n * (n * n - 1.0))
    }
    fn rank(x: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        let mut r = vec![0.0; x.len()];
        for (rank, &i) in idx.iter().enumerate() {
            r[i] = rank as f64;
        }
        r
    }

    #[test]
    fn eci_orders_by_diversity_on_nested_matrix() {
        // Perfectly nested: country i exports the first (i+1) products.
        let c = 6;
        let p = 6;
        let mut data = vec![0.0; c * p];
        for i in 0..c {
            for j in 0..=i {
                data[i * p + j] = 1.0;
            }
        }
        let m = Matrix::from_row_major(c, p, data);
        let res = eci_pci(&m, 20000, 1e-13).expect("defined");
        let kc: Vec<f64> = res.diversity.clone();
        assert!(spearman_sign(&res.eci, &kc) > 0.99, "eci should track diversity");
        let mean: f64 = res.eci.iter().sum::<f64>() / res.eci.len() as f64;
        assert!(mean.abs() < 1e-9);
    }

    #[test]
    fn undefined_when_fewer_than_three_diversified() {
        let m = Matrix::from_row_major(3, 2, vec![1., 1., 0., 0., 0., 0.]);
        assert!(eci_pci(&m, 1000, 1e-12).is_none());
    }
}
