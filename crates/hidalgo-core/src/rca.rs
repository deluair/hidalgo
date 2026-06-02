//! Balassa RCA and binarization to the specialization matrix M.

use crate::matrix::Matrix;

/// Balassa RCA on an exports matrix (rows = countries, cols = products).
pub fn rca(exports: &Matrix) -> Matrix {
    let (c, p) = (exports.rows, exports.cols);
    let total: f64 = exports.data.iter().sum();
    let mut out = Matrix::zeros(c, p);
    if total <= 0.0 {
        return out;
    }
    let country_tot: Vec<f64> = (0..c).map(|i| exports.row(i).iter().sum()).collect();
    let mut product_tot = vec![0.0; p];
    for i in 0..c {
        let row = exports.row(i);
        for (acc, &val) in product_tot.iter_mut().zip(row) {
            *acc += val;
        }
    }
    #[allow(clippy::needless_range_loop)] // i used as positional arg to out.set(i, ..) and exports.get(i, ..)
    for i in 0..c {
        let ct = country_tot[i];
        if ct <= 0.0 {
            continue;
        }
        #[allow(clippy::needless_range_loop)] // j used as positional arg to out.set(i, j, ...)
        for j in 0..p {
            let den = product_tot[j] / total;
            if den <= 0.0 {
                continue;
            }
            out.set(i, j, (exports.get(i, j) / ct) / den);
        }
    }
    out
}

/// M_cp = 1.0 where rca >= threshold, else 0.0.
pub fn binarize(rca: &Matrix, threshold: f64) -> Matrix {
    let data = rca
        .data
        .iter()
        .map(|&v| if v >= threshold { 1.0 } else { 0.0 })
        .collect();
    Matrix::from_row_major(rca.rows, rca.cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn rca_two_by_two() {
        // exports: A=[10,0], B=[0,10]. total=20.
        // country tot: A=10, B=10. product tot: p0=10, p1=10.
        // RCA[A,p0] = (10/10)/(10/20) = 1 / 0.5 = 2.0
        // RCA[A,p1] = (0/10)/(10/20)  = 0
        let ex = Matrix::from_row_major(2, 2, vec![10., 0., 0., 10.]);
        let r = rca(&ex);
        assert_abs_diff_eq!(r.get(0, 0), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.get(0, 1), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.get(1, 1), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn rca_zero_country_row_is_zero() {
        let ex = Matrix::from_row_major(2, 2, vec![0., 0., 5., 5.]);
        let r = rca(&ex);
        assert_eq!(r.row(0), &[0.0, 0.0]);
    }

    #[test]
    fn rca_zero_product_col_is_zero() {
        // product 1 never exported -> column stays 0, no div-by-zero
        let ex = Matrix::from_row_major(2, 2, vec![5., 0., 5., 0.]);
        let r = rca(&ex);
        assert_eq!(r.get(0, 1), 0.0);
        assert_eq!(r.get(1, 1), 0.0);
    }

    #[test]
    fn binarize_threshold() {
        let r = Matrix::from_row_major(1, 3, vec![0.99, 1.0, 2.5]);
        let m = binarize(&r, 1.0);
        assert_eq!(m.row(0), &[0.0, 1.0, 1.0]);
    }
}
