pub mod brown_conrady;

use brown_conrady::BrownConradyParams;
use thiserror::Error;

#[cfg(feature = "py")]
pub mod python;

#[derive(Debug, Clone)]
/// Struct to hold distortion parameters
pub enum DistortionParams {
    BrownConrady(BrownConradyParams),
}

#[derive(Debug, Error)]
pub enum ArgumentError {
    #[error("Invalid parameter length, given {0}. {1}")]
    UnknownParameterLength(usize, String),
}

impl DistortionParams {
    pub fn from_slice(slice: &[f32]) -> Result<Self, ArgumentError> {
        if slice.len() != 14 {
            return Err(ArgumentError::UnknownParameterLength(
                slice.len(),
                "Currently only support 14 for BrownConrady".to_string(),
            ));
        }

        Ok(Self::BrownConrady(BrownConradyParams {
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
        }))
    }
}
