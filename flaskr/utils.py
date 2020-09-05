"""
Utilities functions.
Author: Antonio Strippoli
"""
import PIL.Image as pil
import PIL.ImageOps as pil_ops
import numpy as np


def np_array_to_img_rgb(img, img_mode):
    """
    Convert an image to RGBA channel (if not already)
    """
    # Add channels (if not)
    if img_mode != "RGB":
        img = np.repeat(img[:, :] * 255.0, 3, axis=2)
    else:
        img *= 255.0

    # Add alpha channel and convert to image
    img = np.dstack((img, np.full(img.shape[:-1], 255.0))).astype("int8")
    return pil.fromarray(np.uint8(img))


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
