import sys

import numpy as np
import cam_distort

from PIL import Image
import soraxas_toolbox as st

params = np.array(
    [
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # k1-k6
        -0.05,
        -0.03,  # p1, p2
        0.0,
        0.0,
        0.0,
        0.0,  # s1-s4
        0.5,
        0.5,  # cx, cy in normalized coordinates
    ],
    dtype=np.float32,
)

# Generate some test points
undist_pts = np.array(
    [
        [0.5, 0.5],
        [0.2, 0.8],
        [0.9, 0.1],
    ],
    dtype=np.float32,
)


print("== original points ==")
print(undist_pts)
print("== distored points ==")
cam_distort.apply_distortion(undist_pts, params)
print(undist_pts)
print("== undistored points ==")
cam_distort.apply_undistortion(undist_pts, params, verbose=True)
print(undist_pts)


if len(sys.argv) < 2:
    print("\n To test (un)distort image:\n")
    print(f"Usage: python {__file__} <img_fname.ext>")
    exit(1)

image_path = sys.argv[1]
image = np.array(Image.open(image_path))


print("\n== original ==\n")
st.image.display(image)
print("\n== distored ==\n")
image = cam_distort.distort_image(image, params)
st.image.display(image)
print("\n== undistored ==\n")
image = cam_distort.undistort_image(image, params)
st.image.display(image)
