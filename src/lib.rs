use numpy::ndarray::{Array1, Array2, ArrayView2, Ix2};
use numpy::prelude::*;
use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use rayon::prelude::*;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python, pymodule, types::PyModule, wrap_pyfunction};

/// Struct to hold distortion parameters
#[pyclass]
pub struct DistortionParams {
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
    pub k5: f32,
    pub k6: f32,
    pub p1: f32,
    pub p2: f32,
    pub s1: f32,
    pub s2: f32,
    pub s3: f32,
    pub s4: f32,
    pub cx: f32,
    pub cy: f32,
}

#[pymethods]
impl DistortionParams {
    #[new]
    fn new(
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        k5: f32,
        k6: f32,
        p1: f32,
        p2: f32,
        s1: f32,
        s2: f32,
        s3: f32,
        s4: f32,
        cx: f32,
        cy: f32,
    ) -> Self {
        DistortionParams {
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            p1,
            p2,
            s1,
            s2,
            s3,
            s4,
            cx,
            cy,
        }
    }

    #[staticmethod]
    fn from_array(params: PyReadonlyArray1<f32>) -> PyResult<Self> {
        let params = params.as_slice()?;
        if params.len() != 14 {
            return Err(PyValueError::new_err("Expected array of length 14"));
        }

        Ok(Self {
            k1: params[0],
            k2: params[1],
            k3: params[2],
            k4: params[3],
            k5: params[4],
            k6: params[5],
            p1: params[6],
            p2: params[7],
            s1: params[8],
            s2: params[9],
            s3: params[10],
            s4: params[11],
            cx: params[12],
            cy: params[13],
        })
    }
    fn __repr__(&self) -> String {
        format!(
            "DistortionParams(k1={:.3}, k2={:.3}, k3={:.3}, k4={:.3}, k5={:.3}, k6={:.3}, \
p1={:.3}, p2={:.3}, s1={:.3}, s2={:.3}, s3={:.3}, s4={:.3}, cx={:.3}, cy={:.3})",
            self.k1,
            self.k2,
            self.k3,
            self.k4,
            self.k5,
            self.k6,
            self.p1,
            self.p2,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
            self.cx,
            self.cy
        )
    }
}

/// Implement FromPyObject so numpy arrays are auto-converted
impl<'source> FromPyObject<'source> for DistortionParams {
    fn extract_bound(obj: &Bound<'source, PyAny>) -> PyResult<Self> {
        // Try converting to a numpy array
        let arr: PyReadonlyArray1<f32> = obj.extract()?;
        let slice = arr.as_slice()?;

        if slice.len() != 14 {
            return Err(PyValueError::new_err("Expected array of length 14"));
        }

        Ok(Self {
            k1: slice[0],
            k2: slice[1],
            k3: slice[2],
            k4: slice[3],
            k5: slice[4],
            k6: slice[5],
            p1: slice[6],
            p2: slice[7],
            s1: slice[8],
            s2: slice[9],
            s3: slice[10],
            s4: slice[11],
            cx: slice[12],
            cy: slice[13],
        })
    }
}

/// apply distortion to an ndarray 2D view, returning a new Array2
/// Apply distortion model to a batch of 2D points (vectorized).
pub fn apply_distortion_np_broadcast(
    pts: ArrayView2<'_, f32>, // shape (N, 2)
    params: &DistortionParams,
) -> Array2<f32> {
    let n = pts.shape()[0];

    // Shift by cx, cy (center of distortion)
    let x = &pts.column(0).to_owned() - params.cx;
    let y = &pts.column(1).to_owned() - params.cy;

    // Compute radius powers
    let r2 = &x * &x + &y * &y;
    let r4 = &r2 * &r2;
    let r6 = &r4 * &r2;

    // Radial distortion
    let radial = &Array1::<f32>::ones(n)
        + &r2 * params.k1
        + &r4 * params.k2
        + &r6 * params.k3
        + &(&r2 * &r4) * params.k4
        + &(&r4 * &r4) * params.k5
        + &(&r6 * &r2) * params.k6;

    // Tangential distortion
    let x_tang = 2.0 * params.p1 * &x * &y + params.p2 * (&r2 + 2.0 * &x * &x);
    let y_tang = params.p1 * (&r2 + 2.0 * &y * &y) + 2.0 * params.p2 * &x * &y;

    // Thin prism distortion
    let x_thin = &r2 * params.s1 + &r4 * params.s2;
    let y_thin = &r2 * params.s3 + &r4 * params.s4;

    // Apply distortion
    let x_dist = &x * &radial + &x_tang + &x_thin + params.cx;
    let y_dist = &y * &radial + &y_tang + &y_thin + params.cy;

    // Stack into (N, 2) output array
    let mut out = Array2::<f32>::zeros((n, 2));
    out.column_mut(0).assign(&x_dist);
    out.column_mut(1).assign(&y_dist);

    out
}

/// Undistort a batch of points via iterative Newton method.
///
/// Parameters
/// ----------
/// pts_dist : numpy.ndarray
///     Distorted points, shape (N, 2).
/// params : DistortionParams
///     Distortion parameters.
/// max_iter : int
///     Maximum number of iterations.
/// tol : float
///     Tolerance for convergence.
/// Returns
/// -------
/// numpy.ndarray
///     Undistorted points, shape (N, 2).
/// Rust helper: undistort points using iterative Newton; input is 2D view, returns new Array2
fn undistort_iterative_np_broadcast(
    pts: ArrayView2<'_, f32>,
    params: &DistortionParams,
    max_iter: usize,
    tol: f32,
    verbose: bool,
) -> Array2<f32> {
    let n = pts.shape()[0];

    // Split input into x/y components, subtract cx/cy
    let mut x = &pts.column(0).to_owned() - params.cx; // shape (N,)
    let mut y = &pts.column(1).to_owned() - params.cy; // shape (N,)

    let x_dist_rel = x.clone();
    let y_dist_rel = y.clone();

    for iter in 0..max_iter {
        // Radius powers
        let r2 = &x * &x + &y * &y;
        let r4 = &r2 * &r2;
        let r6 = &r4 * &r2;
        let r8 = &r4 * &r4;

        // Radial distortion
        let radial = &Array1::<f32>::ones(n)
            + &r2 * params.k1
            + &r4 * params.k2
            + &r6 * params.k3
            + &(&r2 * &r4) * params.k4
            + &(&r4 * &r4) * params.k5
            + &(&r6 * &r2) * params.k6;

        // Tangential distortion
        let x_tang = 2.0 * params.p1 * &x * &y + params.p2 * (&r2 + 2.0 * &x * &x);
        let y_tang = params.p1 * (&r2 + 2.0 * &y * &y) + 2.0 * params.p2 * &x * &y;

        // Thin prism distortion
        let x_thin = &r2 * params.s1 + &r4 * params.s2;
        let y_thin = &r2 * params.s3 + &r4 * params.s4;

        // Total distorted coordinates (relative)
        let fx = &x * &radial + &x_tang + &x_thin;
        let fy = &y * &radial + &y_tang + &y_thin;

        // Residuals
        let f1 = &fx - &x_dist_rel;
        let f2 = &fy - &y_dist_rel;

        let residual = (&f1 * &f1 + &f2 * &f2).mapv(f32::sqrt);
        let mean_residual = residual.mean().unwrap_or(0.0);
        if verbose {
            eprintln!("Iter {iter}: mean residual = {mean_residual:.3e}");
        }
        if mean_residual < tol {
            break;
        }

        // Derivatives
        let drad_dr2 = params.k1
            + 2.0 * params.k2 * &r2
            + 3.0 * params.k3 * &r4
            + 5.0 * params.k4 * &r4
            + 8.0 * params.k5 * &r6
            + 7.0 * params.k6 * &r8;

        let drad_dx = 2.0 * &x * &drad_dr2;
        let drad_dy = 2.0 * &y * &drad_dr2;

        // Jacobian terms (element-wise)
        let j11 = &radial
            + &x * &drad_dx
            + 6.0 * params.p2 * &x
            + 2.0 * params.p1 * &y
            + 2.0 * params.s1 * &x
            + 4.0 * params.s2 * &x * &r2;

        let j12 = &x * &drad_dy
            + 2.0 * params.p1 * &x
            + 2.0 * params.p2 * &y
            + 2.0 * params.s1 * &y
            + 4.0 * params.s2 * &y * &r2;

        let j21 = &y * &drad_dx
            + 2.0 * params.p2 * &y
            + 2.0 * params.p1 * &x
            + 2.0 * params.s3 * &x
            + 4.0 * params.s4 * &x * &r2;

        let j22 = &radial
            + &y * &drad_dy
            + 6.0 * params.p1 * &y
            + 2.0 * params.p2 * &x
            + 2.0 * params.s3 * &y
            + 4.0 * params.s4 * &y * &r2;

        // Determinant and inverse
        let det = &j11 * &j22 - &j12 * &j21;
        let det_safe = &det + 1e-12;

        let dx = (-&f1 * &j22 + &f2 * &j12) / &det_safe;
        let dy = (-&f2 * &j11 + &f1 * &j21) / &det_safe;

        x = &x + &dx;
        y = &y + &dy;
    }

    // Final undistorted points (add back cx, cy)
    let x_undist = &x + params.cx;
    let y_undist = &y + params.cy;

    // Stack into (N, 2)
    let mut out = Array2::<f32>::zeros((n, 2));
    out.column_mut(0).assign(&x_undist);
    out.column_mut(1).assign(&y_undist);
    out
}

#[rustfmt::skip]
/// Parallel undistortion for any mutable 2D container of [f32; 2] using Rayon.
///
/// Applies Newton-Raphson iteration per point, stops when residual < tol.
/// Modifies points in-place.
///
/// - `points`: mutable reference to 2D container of points `&mut [[f32; 2]]`
/// - `params`: distortion model parameters
/// - `max_iter`: max iterations per point
/// - `tol`: convergence tolerance
/// - `verbose`: if true, print per-point residual
pub fn undistort_iterative_parallel(
    points: &mut [[f32; 2]],
    params: &DistortionParams,
    max_iter: usize,
    tol: f32,
    verbose: bool,
) {
    points.par_iter_mut().for_each(|p| {

        // Initial guess: shift by cx, cy
        let mut x = p[0] - params.cx;
        let mut y = p[1] - params.cy;

        let x_dist = x;
        let y_dist = y;

        for iter in 0..max_iter {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let r8 = r4 * r4;

            let radial = 1.0
                + r2 * params.k1
                + r4 * params.k2
                + r6 * params.k3
                + r2 * r4 * params.k4
                + r4 * r4 * params.k5
                + r6 * r2 * params.k6;

            let x_tang = 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x);
            let y_tang = params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y;

            let x_thin = r2 * params.s1 + r4 * params.s2;
            let y_thin = r2 * params.s3 + r4 * params.s4;

            let fx = x * radial + x_tang + x_thin;
            let fy = y * radial + y_tang + y_thin;

            let f1 = fx - x_dist;
            let f2 = fy - y_dist;

            let residual = (f1 * f1 + f2 * f2).sqrt();


            if residual < tol {
                break;
            } else if iter == max_iter - 1 && verbose {
                eprintln!("Point ({:.3}, {:.3}): did not converge after {} iter, residual = {:.3e}", p[0], p[1], iter, residual);
            }

            // Derivatives
            let drad_dr2 = params.k1
                + 2.0 * params.k2 * r2
                + 3.0 * params.k3 * r4
                + 5.0 * params.k4 * r4
                + 8.0 * params.k5 * r6
                + 7.0 * params.k6 * r8;

            let drad_dx = 2.0 * x * drad_dr2;
            let drad_dy = 2.0 * y * drad_dr2;

            // Jacobian terms
            let j11 = radial
                + x * drad_dx
                + 6.0 * params.p2 * x
                + 2.0 * params.p1 * y
                + 2.0 * params.s1 * x
                + 4.0 * params.s2 * x * r2;

            let j12 = x * drad_dy
                + 2.0 * params.p1 * x
                + 2.0 * params.p2 * y
                + 2.0 * params.s1 * y
                + 4.0 * params.s2 * y * r2;

            let j21 = y * drad_dx
                + 2.0 * params.p2 * y
                + 2.0 * params.p1 * x
                + 2.0 * params.s3 * x
                + 4.0 * params.s4 * x * r2;

            let j22 = radial
                + y * drad_dy
                + 6.0 * params.p1 * y
                + 2.0 * params.p2 * x
                + 2.0 * params.s3 * y
                + 4.0 * params.s4 * y * r2;

            // Inverse Jacobian
            let det = j11 * j22 - j12 * j21;
            let det_safe = det + f32::EPSILON;


            let dx = (-f1 * j22 + f2 * j12) / det_safe;
            let dy = (-f2 * j11 + f1 * j21) / det_safe;

            x += dx;
            y += dy;
        }

        // Update the original point (add back cx, cy)
        p[0] = x + params.cx;
        p[1] = y + params.cy;
    });
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
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f32>,
    params: &DistortionParams,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_dyn = pts.as_array();
    let a2 = a_dyn
        .into_dimensionality::<Ix2>()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("pts must be a 2-D array"))?;
    let z = apply_distortion_np_broadcast(a2, params);
    Ok(z.into_pyarray(py))
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
    py: Python<'py>,
    pts: &Bound<'py, PyArray2<f32>>,
    params: &DistortionParams,
    max_iter: usize,
    tol: f32,
    verbose: bool,
) -> PyResult<()> {
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

    let n = shape[0];

    // Safety: We have validated the shape and contiguity of the input array.
    unsafe {
        let raw = pts.as_slice_mut()?.as_mut_ptr();
        let points_slice = std::slice::from_raw_parts_mut(raw as *mut [f32; 2], n);

        undistort_iterative_parallel(points_slice, params, max_iter, tol, verbose);
    }

    Ok(())
}

/// The Python module
#[pymodule]
fn cam_distort<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<DistortionParams>()?;

    // register top-level wrappers
    m.add_function(wrap_pyfunction!(apply_distortion, m)?)?;
    m.add_function(wrap_pyfunction!(undistort_iterative, m)?)?;

    Ok(())
}
