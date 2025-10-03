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

pub trait DistortionParamsTrait {
    const NUM_PARAMS: usize;
}

#[derive(Debug, Error)]
pub enum ArgumentError {
    #[error("Invalid parameter length, given {0}. {1}")]
    UnknownParameterLength(usize, String),
}

impl DistortionParams {
    pub fn from_slice(slice: &[f32]) -> Result<Self, ArgumentError> {
        if let Ok(fixed) = <&[f32; BrownConradyParams::NUM_PARAMS]>::try_from(slice) {
            Ok(Self::BrownConrady(BrownConradyParams::from_slice(fixed)))
        } else {
            Err(ArgumentError::UnknownParameterLength(
                slice.len(),
                format!("Currently only support {} params for BrownConrady",
                BrownConradyParams::NUM_PARAMS,
            ),
            ))
        }
    }
}
