//! Second eigenvector of a reflections matrix, found matrix-free by shifted,
//! deflated power iteration on the symmetric similar operator.

#[allow(unused_imports)]
use crate::matrix::Matrix;

/// Raw second eigenvector of the reflections matrix (before standardization).
pub struct SecondEig {
    pub raw: Vec<f64>,
    pub iters: usize,
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f64]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Second eigenvector of Mtilde = D^{-1} A D_other^{-1} A^T, solved matrix-free
/// on the symmetric similar S via shifted (S+I), deflated power iteration.
///
/// `d` is the primary-axis degree vector (len `n`), `d_other` the other axis.
/// `apply_a(x_other) -> A x` (len `n`); `apply_at(x_n) -> A^T x` (len other).
pub fn second_eigenvector<FA, FT>(
    n: usize,
    d: &[f64],
    d_other: &[f64],
    apply_a: FA,
    apply_at: FT,
    max_iters: usize,
    tol: f64,
) -> SecondEig
where
    FA: Fn(&[f64]) -> Vec<f64>,
    FT: Fn(&[f64]) -> Vec<f64>,
{
    let inv_sqrt_d: Vec<f64> =
        d.iter().map(|&v| if v > 0.0 { 1.0 / v.sqrt() } else { 0.0 }).collect();
    let inv_d_other: Vec<f64> =
        d_other.iter().map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 }).collect();

    // top eigenvector of S: w1 = normalize(sqrt(d))
    let mut w1: Vec<f64> = d.iter().map(|&v| v.sqrt()).collect();
    normalize(&mut w1);

    let apply_s = |v: &[f64]| -> Vec<f64> {
        let t1: Vec<f64> = v.iter().zip(&inv_sqrt_d).map(|(a, b)| a * b).collect();
        let t2 = apply_at(&t1);
        let t3: Vec<f64> = t2.iter().zip(&inv_d_other).map(|(a, b)| a * b).collect();
        let t4 = apply_a(&t3);
        t4.iter().zip(&inv_sqrt_d).map(|(a, b)| a * b).collect()
    };

    // deterministic init, deflated against w1
    let mut v: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let c0 = dot(&w1, &v);
    for i in 0..n {
        v[i] -= c0 * w1[i];
    }
    normalize(&mut v);

    let mut prev = v.clone();
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iters {
        iters += 1;
        let sv = apply_s(&v);
        for i in 0..n {
            v[i] = sv[i] + v[i]; // (S + I) shift
        }
        let c1 = dot(&w1, &v);
        for i in 0..n {
            v[i] -= c1 * w1[i]; // deflate
        }
        normalize(&mut v);
        let mut s = dot(&v, &prev);
        if s < 0.0 {
            for x in v.iter_mut() {
                *x = -*x;
            }
            s = -s;
        }
        if 1.0 - s < tol {
            converged = true;
            break;
        }
        prev.copy_from_slice(&v);
    }

    let raw: Vec<f64> = v.iter().zip(&inv_sqrt_d).map(|(a, b)| a * b).collect();
    SecondEig { raw, iters, converged }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::jacobi_symmetric;
    use approx::assert_abs_diff_eq;

    fn aligned_cos(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        (dot / (na * nb)).abs()
    }

    #[test]
    fn second_eigenvector_matches_jacobi() {
        // M: 4 countries x 5 products, varied diversity.
        let m = Matrix::from_row_major(
            4,
            5,
            vec![
                1., 1., 1., 0., 0., // kc=3
                1., 1., 0., 0., 0., // kc=2
                1., 0., 0., 1., 1., // kc=3
                0., 0., 0., 1., 0., // kc=1
            ],
        );
        let c = m.rows;
        let p = m.cols;
        let kc: Vec<f64> = (0..c).map(|i| m.row(i).iter().sum()).collect();
        let mut kp = vec![0.0; p];
        for i in 0..c {
            for j in 0..p {
                kp[j] += m.get(i, j);
            }
        }
        let got = second_eigenvector(
            c,
            &kc,
            &kp,
            |x| m.matvec(x),
            |x| m.matvec_transpose(x),
            5000,
            1e-13,
        );
        assert!(got.converged, "power iteration did not converge");

        // oracle: build symmetric S = Dc^-1/2 M Dp^-1 M^T Dc^-1/2 explicitly
        let inv_sqrt_kc: Vec<f64> = kc.iter().map(|&v| 1.0 / v.sqrt()).collect();
        let inv_kp: Vec<f64> = kp.iter().map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 }).collect();
        let mut s = vec![0.0; c * c];
        for a in 0..c {
            for b in 0..c {
                let mut acc = 0.0;
                for q in 0..p {
                    acc += m.get(a, q) * m.get(b, q) * inv_kp[q];
                }
                s[a * c + b] = inv_sqrt_kc[a] * acc * inv_sqrt_kc[b];
            }
        }
        let (eigvals, eigvecs) = jacobi_symmetric(&s, c);
        let mut idx: Vec<usize> = (0..c).collect();
        idx.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());
        let second = idx[1];
        let w2: Vec<f64> = (0..c).map(|k| eigvecs[k * c + second]).collect();
        let oracle_raw: Vec<f64> = w2.iter().zip(&inv_sqrt_kc).map(|(a, b)| a * b).collect();

        assert_abs_diff_eq!(aligned_cos(&got.raw, &oracle_raw), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn second_eigenvector_random_both_sides() {
        // Deterministic 8x12 binary matrix with varied row/col sums.
        let rows = 8usize;
        let cols = 12usize;
        let mut data = vec![0.0f64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let bit = ((i * 7 + j * 3 + 1) % 5) < 3;
                if bit {
                    data[i * cols + j] = 1.0;
                }
            }
        }
        // Fix up empty rows: set M[i][i % cols] = 1
        for i in 0..rows {
            let row_sum: f64 = (0..cols).map(|j| data[i * cols + j]).sum();
            if row_sum == 0.0 {
                data[i * cols + (i % cols)] = 1.0;
            }
        }
        // Fix up empty cols: set M[i % rows][j] = 1
        for j in 0..cols {
            let col_sum: f64 = (0..rows).map(|i| data[i * cols + j]).sum();
            if col_sum == 0.0 {
                data[(j % rows) * cols + j] = 1.0;
            }
        }

        let m = Matrix::from_row_major(rows, cols, data);

        // Compute kc (row sums) and kp (col sums)
        let kc: Vec<f64> = (0..rows).map(|i| m.row(i).iter().sum()).collect();
        let mut kp = vec![0.0f64; cols];
        for i in 0..rows {
            for j in 0..cols {
                kp[j] += m.get(i, j);
            }
        }

        // --- COUNTRY side (n=8) ---
        let got_c = second_eigenvector(
            rows,
            &kc,
            &kp,
            |x| m.matvec(x),
            |x| m.matvec_transpose(x),
            100_000,
            1e-13,
        );
        assert!(got_c.converged, "country side did not converge");

        // Build explicit S_c = Dc^{-1/2} M Dp^{-1} M^T Dc^{-1/2} (rows x rows)
        let inv_sqrt_kc: Vec<f64> = kc.iter().map(|&v| 1.0 / v.sqrt()).collect();
        let inv_kp: Vec<f64> = kp.iter().map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 }).collect();
        let mut sc = vec![0.0f64; rows * rows];
        for a in 0..rows {
            for b in 0..rows {
                let mut acc = 0.0;
                for q in 0..cols {
                    acc += m.get(a, q) * m.get(b, q) * inv_kp[q];
                }
                sc[a * rows + b] = inv_sqrt_kc[a] * acc * inv_sqrt_kc[b];
            }
        }
        let (eigvals_c, eigvecs_c) = jacobi_symmetric(&sc, rows);
        let mut idx_c: Vec<usize> = (0..rows).collect();
        idx_c.sort_by(|&i, &j| eigvals_c[j].partial_cmp(&eigvals_c[i]).unwrap());
        let second_c = idx_c[1];
        let w2_c: Vec<f64> = (0..rows).map(|k| eigvecs_c[k * rows + second_c]).collect();
        let oracle_raw_c: Vec<f64> =
            w2_c.iter().zip(&inv_sqrt_kc).map(|(a, b)| a * b).collect();

        let cos_c = aligned_cos(&got_c.raw, &oracle_raw_c);
        assert_abs_diff_eq!(cos_c, 1.0, epsilon = 1e-7);

        // --- PRODUCT side (n=12) ---
        // apply_a for product side = M^T x (cols->rows direction reversed)
        // apply_at for product side = M x
        let got_p = second_eigenvector(
            cols,
            &kp,
            &kc,
            |x| m.matvec_transpose(x),
            |x| m.matvec(x),
            100_000,
            1e-13,
        );
        assert!(got_p.converged, "product side did not converge");

        // Build explicit S_p = Dp^{-1/2} M^T Dc^{-1} M Dp^{-1/2} (cols x cols)
        // S_p[a,b] = inv_sqrt_kp[a] * (sum_c M[c,a]*M[c,b]*inv_kc[c]) * inv_sqrt_kp[b]
        let inv_sqrt_kp: Vec<f64> = kp.iter().map(|&v| 1.0 / v.sqrt()).collect();
        let inv_kc: Vec<f64> = kc.iter().map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 }).collect();
        let mut sp = vec![0.0f64; cols * cols];
        for a in 0..cols {
            for b in 0..cols {
                let mut acc = 0.0;
                for c in 0..rows {
                    acc += m.get(c, a) * m.get(c, b) * inv_kc[c];
                }
                sp[a * cols + b] = inv_sqrt_kp[a] * acc * inv_sqrt_kp[b];
            }
        }
        let (eigvals_p, eigvecs_p) = jacobi_symmetric(&sp, cols);
        let mut idx_p: Vec<usize> = (0..cols).collect();
        idx_p.sort_by(|&i, &j| eigvals_p[j].partial_cmp(&eigvals_p[i]).unwrap());
        let second_p = idx_p[1];
        let w2_p: Vec<f64> = (0..cols).map(|k| eigvecs_p[k * cols + second_p]).collect();
        let oracle_raw_p: Vec<f64> =
            w2_p.iter().zip(&inv_sqrt_kp).map(|(a, b)| a * b).collect();

        let cos_p = aligned_cos(&got_p.raw, &oracle_raw_p);
        assert_abs_diff_eq!(cos_p, 1.0, epsilon = 1e-7);
    }
}
