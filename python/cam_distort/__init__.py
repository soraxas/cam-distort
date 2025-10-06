import numpy as np

from .cam_distort import (
    DistortionParams,
    apply_distortion,
    apply_undistortion,
)

from cam_distort import cam_distort
from typing import Callable

__doc__ = cam_distort.__doc__
if hasattr(cam_distort, "__all__"):
    __all__ = cam_distort.__all__


DistortionParamsType = np.ndarray | DistortionParams

remap_pixel_docstring = """
{verb} an image using {verb} parameters

Args:
    img (np.ndarray): input image.
    params (np.ndarray): [k1, k2, ..., p1, p2, ..., cx_norm, cy_norm]

Returns:
    img (np.ndarray): {verb}ed output image (same size as input).
"""


def _remap_pixels_with_functor(
    image: np.ndarray,
    params: DistortionParamsType,
    functor: Callable[[np.ndarray, DistortionParamsType], np.ndarray],
    opencv_kwargs: dict | None = None,
):
    try:
        import cv2
    except ImportError as e:
        raise ImportError("Needs opencv2 for remapping image pixel") from e

    if opencv_kwargs is None:
        opencv_kwargs = dict(
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    h, w = image.shape[:2]

    scale = np.array([w, h])

    # Generate grid of normalized coordinates for each pixel
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([xv, yv], axis=-1).reshape(-1, 2)

    # Normalize grid coordinates
    coords_norm = coords / scale

    # Inverse mapping: compute distorted pixel locations that map to each undistorted location
    coords_norm = coords_norm.astype(np.float32)
    # cam_distort.apply_distortion(coords_norm, params)
    functor(coords_norm, params)
    # distorted_coords_norm = distortion_model(params, coords_norm)

    distorted_coords_norm = coords_norm

    # Convert back to pixel space
    distorted_coords_px = distorted_coords_norm * scale

    # Reshape for remapping
    map_x = distorted_coords_px[:, 0].reshape(h, w).astype(np.float32)
    map_y = distorted_coords_px[:, 1].reshape(h, w).astype(np.float32)

    # Remap the distorted image to get the undistorted version
    img_undistorted = cv2.remap(
        image,
        map_x,
        map_y,
        **opencv_kwargs,
    )

    return img_undistorted


def distort_image(image: np.ndarray, params: DistortionParamsType):
    f"""{remap_pixel_docstring.format(verb="distort")}"""
    return _remap_pixels_with_functor(image, params, apply_distortion)


def undistort_image(image: np.ndarray, params: DistortionParamsType):
    f"""{remap_pixel_docstring.format(verb="undistort")}"""
    return _remap_pixels_with_functor(image, params, apply_undistortion)


__all__ = [
    "DistortionParams",
    "apply_distortion",
    "apply_undistortion",
]
