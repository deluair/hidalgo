// pyo3's PyResult<T> triggers useless_conversion on the implicit PyErr coercion; suppress here.
#![allow(clippy::useless_conversion)]

use hidalgo_core::matrix::Matrix;
use hidalgo_core::rca::rca as core_rca; // `rca` is a module; import the fn explicitly
use hidalgo_core::{bundle_from_exports, bundle_from_m, bundle_from_rca, Bundle};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_matrix(arr: &PyReadonlyArray2<f64>) -> Matrix {
    let a = arr.as_array();
    let (rows, cols) = (a.nrows(), a.ncols());
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(a[[i, j]]);
        }
    }
    Matrix::from_row_major(rows, cols, data)
}

fn mat_to_py<'py>(py: Python<'py>, m: &Matrix) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((m.rows, m.cols), m.data.clone())
        .unwrap()
        .into_pyarray_bound(py)
}

fn bundle_to_dict<'py>(py: Python<'py>, b: &Bundle) -> PyResult<Bound<'py, PyDict>> {
    let cx = &b.complexity;
    let out = PyDict::new_bound(py);
    out.set_item("eci", cx.eci.clone().into_pyarray_bound(py))?;
    out.set_item("pci", cx.pci.clone().into_pyarray_bound(py))?;
    out.set_item("diversity", cx.diversity.clone().into_pyarray_bound(py))?;
    out.set_item("ubiquity", cx.ubiquity.clone().into_pyarray_bound(py))?;
    out.set_item(
        "kept_countries",
        cx.kept_countries.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray_bound(py),
    )?;
    out.set_item(
        "kept_products",
        cx.kept_products.iter().map(|&x| x as i64).collect::<Vec<_>>().into_pyarray_bound(py),
    )?;
    out.set_item("proximity", mat_to_py(py, &b.proximity))?;
    out.set_item("density", mat_to_py(py, &b.density))?;
    out.set_item("eci_converged", cx.eci_converged)?;
    out.set_item("pci_converged", cx.pci_converged)?;
    Ok(out)
}

#[pyfunction]
fn rca<'py>(py: Python<'py>, exports: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
    mat_to_py(py, &core_rca(&to_matrix(&exports)))
}

#[pyfunction]
#[pyo3(signature = (exports, threshold=1.0, max_iters=100000, tol=1e-12))]
fn bundle_from_exports_py<'py>(
    py: Python<'py>,
    exports: PyReadonlyArray2<f64>,
    threshold: f64,
    max_iters: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let b = bundle_from_exports(&to_matrix(&exports), threshold, max_iters, tol)
        .ok_or_else(|| PyValueError::new_err("complexity undefined for input"))?;
    bundle_to_dict(py, &b)
}

#[pyfunction]
#[pyo3(signature = (rca, threshold=1.0, max_iters=100000, tol=1e-12))]
fn bundle_from_rca_py<'py>(
    py: Python<'py>,
    rca: PyReadonlyArray2<f64>,
    threshold: f64,
    max_iters: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let b = bundle_from_rca(&to_matrix(&rca), threshold, max_iters, tol)
        .ok_or_else(|| PyValueError::new_err("complexity undefined for input"))?;
    bundle_to_dict(py, &b)
}

#[pyfunction]
#[pyo3(signature = (m, max_iters=100000, tol=1e-12))]
fn bundle_from_m_py<'py>(
    py: Python<'py>,
    m: PyReadonlyArray2<f64>,
    max_iters: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let b = bundle_from_m(&to_matrix(&m), max_iters, tol)
        .ok_or_else(|| PyValueError::new_err("complexity undefined for input"))?;
    bundle_to_dict(py, &b)
}

#[pymodule]
fn _hidalgo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rca, m)?)?;
    m.add_function(wrap_pyfunction!(bundle_from_exports_py, m)?)?;
    m.add_function(wrap_pyfunction!(bundle_from_rca_py, m)?)?;
    m.add_function(wrap_pyfunction!(bundle_from_m_py, m)?)?;
    Ok(())
}
