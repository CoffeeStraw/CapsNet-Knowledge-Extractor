"""
Utilities functions.
Author: Antonio Strippoli
"""
import PIL.Image as pil
import PIL.ImageOps as pil_ops
import numpy as np


def img_transform(img_as_array, rotation, invert_colors):
    """
    Transform an image by applying rotation and color inversion.
    """
    img = pil.fromarray(img_as_array)

    # Rotate?
    if rotation > 0:
        img = img.rotate(rotation)

    # Invert colors?
    if invert_colors:
        img = pil_ops.invert(img)

    return np.array(img)
