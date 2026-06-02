//! Test-only dense symmetric eigensolver (cyclic Jacobi) used as an oracle to
//! validate the matrix-free power-iteration solver. Returns (eigenvalues,
//! eigenvectors-as-columns) for a symmetric n x n matrix given row-major.

#![cfg(test)]

/// Cyclic Jacobi eigendecomposition of a symmetric matrix `a` (row-major, n x n).
/// Returns (eigenvalues, v) where v is row-major n x n with eigenvector k in
/// column k. Eigenvalues are NOT sorted.
pub fn jacobi_symmetric(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let at = |a: &[f64], i: usize, j: usize| a[i * n + j];
    for _sweep in 0..100 {
        // off-diagonal norm
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += at(&a, i, j).powi(2);
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let cos = 1.0 / (t * t + 1.0).sqrt();
                let sin = t * cos;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = cos * akp - sin * akq;
                    a[k * n + q] = sin * akp + cos * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = cos * apk - sin * aqk;
                    a[q * n + k] = sin * apk + cos * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = cos * vkp - sin * vkq;
                    v[k * n + q] = sin * vkp + cos * vkq;
                }
            }
        }
    }
    let eig = (0..n).map(|i| a[i * n + i]).collect();
    (eig, v)
}
