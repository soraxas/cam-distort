use numpy::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray2};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python, pymodule, types::PyModule, wrap_pyfunction};

use crate::{ArgumentError, DistortionParams, brown_conrady};

impl From<ArgumentError> for PyErr {
    fn from(err: ArgumentError) -> PyErr {
        match err {
            ArgumentError::UnknownParameterLength(len, msg) => PyValueError::new_err(format!(
                "Unknown parameters length {}. Currently only support 14 for BrownConrady. {}",
                len, msg
            )),
        }
    }
}

#[pyclass(name = "DistortionParams")]
pub struct PyDistortionParams(pub DistortionParams);

#[pymethods]
impl PyDistortionParams {
    #[new]
    fn new(arr: PyReadonlyArray1<f32>) -> Result<Self, PyErr> {
        Ok(PyDistortionParams(DistortionParams::from_slice(
            arr.as_slice()?,
        )?))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Implement FromPyObject so numpy arrays are auto-converted
impl<'source> FromPyObject<'source> for PyDistortionParams {
    fn extract_bound(obj: &Bound<'source, PyAny>) -> PyResult<Self> {
        // Try converting to a numpy array
        let arr: PyReadonlyArray1<f32> = obj.extract()?;
        let slice = arr.as_slice()?;

        DistortionParams::from_slice(slice)
            .map(PyDistortionParams)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

fn check_is_2d_pts_array(pts: &PyReadwriteArray2<f32>) -> Result<(), PyErr> {
    // Validate shape
    let shape = pts.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input must be a (N, 2) array",
        ));
    }

    // Ensure it's contiguous in memory
    if !pts.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must be C-contiguous",
        ));
    }
    Ok(())
}

// wrapper of apply_distortion_np_broadcast
/// Apply distortion to a batch of points.
///
/// Parameters
/// ----------
/// pts : numpy.ndarray
///     Array of shape (N, 2) of points.
/// params : DistortionParams
///     Distortion parameters.
#[pyfunction(signature = (pts, params))]
fn apply_distortion<'py>(
    mut pts: PyReadwriteArray2<f32>,
    params: PyDistortionParams,
) -> PyResult<()> {
    check_is_2d_pts_array(&pts)?;

    let points_slice: &mut [[f32; 2]] = bytemuck::cast_slice_mut(pts.as_slice_mut()?);
    match &params.0 {
        DistortionParams::BrownConrady(params) => {
            brown_conrady::apply_distortion_parallel(points_slice, params);
        }
    }

    Ok(())
}

// wrapper of undistort_iterative_np_broadcast
/// Undistort a batch of points via iterative Newton method.
///
/// Parameters
/// ----------
/// pts : numpy.ndarray
///     Distorted points, shape (N, 2).
/// params : DistortionParams
///     Distortion parameters.
/// max_iter : int, optional
///     Maximum number of iterations.
/// tol : float, optional
///     Tolerance for convergence.
#[pyfunction(signature = (pts, params, max_iter = 60, tol = 1e-6, verbose = false))]
fn undistort_iterative<'py>(
    mut pts: PyReadwriteArray2<f32>,
    params: PyDistortionParams,
    max_iter: usize,
    tol: f32,
    verbose: bool,
) -> PyResult<()> {
    // Validate shape
    check_is_2d_pts_array(&pts)?;

    let points_slice: &mut [[f32; 2]] = bytemuck::cast_slice_mut(pts.as_slice_mut()?);
    match &params.0 {
        DistortionParams::BrownConrady(params) => {
            brown_conrady::undistort_iterative_parallel(
                points_slice,
                params,
                max_iter,
                tol,
                verbose,
            );
        }
    }
    Ok(())
}

/// The Python module
#[pymodule]
fn cam_distort<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistortionParams>()?;

    // register top-level wrappers
    m.add_function(wrap_pyfunction!(apply_distortion, m)?)?;
    m.add_function(wrap_pyfunction!(undistort_iterative, m)?)?;

    Ok(())
}
