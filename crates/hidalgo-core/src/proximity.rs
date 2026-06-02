//! Product proximity (Hidalgo et al. 2007) and density.

use crate::matrix::Matrix;
use rayon::prelude::*;

/// Product proximity: phi[a,b] = (sum_c M[c,a] M[c,b]) / max(kp[a], kp[b]).
/// Symmetric P x P; diagonal is 1 where kp > 0. Parallel over output rows;
/// rayon preserves order for indexed parallel iterators, so output is
/// deterministic.
pub fn proximity(m: &Matrix, kp: &[f64]) -> Matrix {
    let (c, p) = (m.rows, m.cols);
    let data: Vec<f64> = (0..p)
        .into_par_iter()
        .flat_map_iter(|a| {
            let mut rowout = vec![0.0f64; p];
            for ci in 0..c {
                let mca = m.get(ci, a);
                if mca == 0.0 {
                    continue;
                }
                let crow = m.row(ci);
                for b in 0..p {
                    rowout[b] += mca * crow[b];
                }
            }
            for b in 0..p {
                let denom = kp[a].max(kp[b]);
                rowout[b] = if denom > 0.0 { rowout[b] / denom } else { 0.0 };
            }
            rowout.into_iter()
        })
        .collect();
    Matrix::from_row_major(p, p, data)
}

/// Density: density[c,p] = (sum_p' M[c,p'] phi[p',p]) / (sum_p' phi[p,p']).
pub fn density(m: &Matrix, phi: &Matrix) -> Matrix {
    let (c, p) = (m.rows, m.cols);
    let phi_rowsum: Vec<f64> = (0..p).map(|a| phi.row(a).iter().sum()).collect();
    let data: Vec<f64> = (0..c)
        .into_par_iter()
        .flat_map_iter(|ci| {
            let crow = m.row(ci);
            let mut out = vec![0.0f64; p];
            for pp in 0..p {
                let mcpp = crow[pp];
                if mcpp == 0.0 {
                    continue;
                }
                let phirow = phi.row(pp); // phi symmetric: phi[pp,pj] = phi[pj,pp]
                for pj in 0..p {
                    out[pj] += mcpp * phirow[pj];
                }
            }
            for pj in 0..p {
                out[pj] = if phi_rowsum[pj] > 0.0 { out[pj] / phi_rowsum[pj] } else { 0.0 };
            }
            out.into_iter()
        })
        .collect();
    Matrix::from_row_major(c, p, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn proximity_small() {
        // 3 countries x 2 products.
        // M = [[1,1],[1,0],[0,1]] -> kp = [2,2]; co[0,1] = 1 (only country 0 has both)
        // phi[0,1] = 1 / max(2,2) = 0.5 ; diagonal = 1
        let m = Matrix::from_row_major(3, 2, vec![1., 1., 1., 0., 0., 1.]);
        let kp = vec![2.0, 2.0];
        let phi = proximity(&m, &kp);
        assert_abs_diff_eq!(phi.get(0, 0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(phi.get(0, 1), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(phi.get(1, 0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn density_small() {
        let m = Matrix::from_row_major(3, 2, vec![1., 1., 1., 0., 0., 1.]);
        let kp = vec![2.0, 2.0];
        let phi = proximity(&m, &kp);
        let d = density(&m, &phi);
        // country 1 has M=[1,0]; density for product1 (target j=1):
        //   num = M[1,0]*phi[0,1] + M[1,1]*phi[1,1] = 1*0.5 + 0*1 = 0.5
        //   denom = phi_rowsum[1] = phi[1,0]+phi[1,1] = 0.5+1 = 1.5
        //   density[1,1] = 0.5/1.5 = 1/3
        assert_abs_diff_eq!(d.get(1, 1), 1.0 / 3.0, epsilon = 1e-12);
    }
}
