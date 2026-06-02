//! Row-major dense f64 matrix and the matrix-vector kernels the complexity
//! computations need. Plain Rust + rayon; deterministic across thread counts
//! (every output entry is reduced by a single thread in a fixed index order).

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, // row-major: data[i * cols + j]
}

impl Matrix {
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
        Matrix { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// y = M * x  (len(x) == cols, len(y) == rows). Parallel over output rows.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols, "matvec: len(x) must equal cols");
        self.data
            .par_chunks(self.cols)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// y = M^T * x  (len(x) == rows, len(y) == cols). Parallel over output
    /// columns so each entry is summed in fixed row order (deterministic).
    pub fn matvec_transpose(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.rows, "matvec_transpose: len(x) must equal rows");
        let (rows, cols) = (self.rows, self.cols);
        (0..cols)
            .into_par_iter()
            .map(|j| {
                let mut s = 0.0;
                for i in 0..rows {
                    s += self.data[i * cols + j] * x[i];
                }
                s
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn matvec_basic() {
        // [[1,2,3],[4,5,6]] * [1,1,1] = [6,15]
        let m = Matrix::from_row_major(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let y = m.matvec(&[1., 1., 1.]);
        assert_abs_diff_eq!(y[0], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(y[1], 15.0, epsilon = 1e-12);
    }

    #[test]
    fn matvec_transpose_basic() {
        // M^T * [1,1] where M=[[1,2,3],[4,5,6]] -> [5,7,9]
        let m = Matrix::from_row_major(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let y = m.matvec_transpose(&[1., 1.]);
        assert_eq!(y.len(), 3);
        assert_abs_diff_eq!(y[0], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(y[1], 7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(y[2], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn row_and_get() {
        let m = Matrix::from_row_major(2, 2, vec![1., 2., 3., 4.]);
        assert_eq!(m.row(1), &[3.0, 4.0]);
        assert_eq!(m.get(1, 0), 3.0);
    }
}
