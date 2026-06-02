pub mod matrix;
pub mod rca;
pub mod eig;
pub mod complexity;
pub mod proximity;
#[cfg(test)]
mod testutil;

/// Full result of the complexity pipeline from a binary M.
pub struct Bundle {
    pub complexity: complexity::ComplexityResult,
    pub proximity: matrix::Matrix,
    pub density: matrix::Matrix,
}

/// Run the whole pipeline from a binary specialization matrix M.
pub fn bundle_from_m(m: &matrix::Matrix, max_iters: usize, tol: f64) -> Option<Bundle> {
    let complexity = complexity::eci_pci(m, max_iters, tol)?;
    let phi = proximity::proximity(m, &complexity.ubiquity);
    let dens = proximity::density(m, &phi);
    Some(Bundle { complexity, proximity: phi, density: dens })
}

/// Run from an RCA matrix: binarize at `threshold`, then `bundle_from_m`.
pub fn bundle_from_rca(rca_mat: &matrix::Matrix, threshold: f64, max_iters: usize, tol: f64) -> Option<Bundle> {
    let m = rca::binarize(rca_mat, threshold);
    bundle_from_m(&m, max_iters, tol)
}

/// Run from raw exports: RCA -> binarize -> bundle.
pub fn bundle_from_exports(exports: &matrix::Matrix, threshold: f64, max_iters: usize, tol: f64) -> Option<Bundle> {
    let r = rca::rca(exports);
    bundle_from_rca(&r, threshold, max_iters, tol)
}

#[cfg(test)]
mod bundle_tests {
    use super::*;
    #[test]
    fn bundle_runs_end_to_end() {
        let c = 6;
        let p = 6;
        let mut data = vec![0.0; c * p];
        for i in 0..c {
            for j in 0..=i {
                data[i * p + j] = 1.0;
            }
        }
        let m = matrix::Matrix::from_row_major(c, p, data);
        let b = bundle_from_m(&m, 20000, 1e-13).expect("defined");
        assert_eq!(b.proximity.rows, p);
        assert_eq!(b.density.rows, c);
        assert_eq!(b.complexity.eci.len(), b.complexity.kept_countries.len());
    }
}
