import warnings
import numpy as np
from skimage.measure import find_contours
from skimage import dtype_limits, filters
from skimage.filters import gaussian
import skimage.morphology as morph
from scipy.ndimage.filters import maximum_filter


def invert(
        img: np.ndarray
) -> np.ndarray:
    if img.dtype == "bool":
        return ~img
    else:
        return dtype_limits(img, clip_negative=False)[1] - img


def exclude_regions(
        label_image: np.ndarray,
        y_range_ex: tuple,
        x_range_ex: tuple,
        background: int = 0,
) -> np.ndarray:
    """Exclude regions in bottom and right overlap.

    Exclude all labelled regions that share pixels with the bottom or right overlap area that is defined by
    `y_range_ex` and `x_range_ex`.
    Also exclude all labelled regions that do not share any pixels with the tile area.

    Args:
        label_image: Label image
        y_range_ex: Range (start, end) on y axis with `start` denoting the first index within data region (non overlap)
            and `end` denoting the first overlap index after the data region.
        x_range_ex: Range (start, end) on x axis with `start` denoting the first index within data region (non overlap)
            and `end` denoting the first overlap index after the data region.
        background: Background label.

    Raises:
        Warnings:
            - If labelled region that belongs to tile touches upper or left border a warning is raised,
              as this indicates that the overlap is not sufficient to securely include all objects.

    Returns:
        Resulting label image
    """

    def f(*args):
        z = np.unique(np.concatenate([np.unique(a) for a in args]))
        return list(filter((np.array(background)).__ne__, z))

    ya, yb = y_range_ex
    xa, xb = x_range_ex

    if ((ya > 0 and np.any(~np.isclose(label_image[:yb, 0], background))) or
            (xa > 0 and np.any(~np.isclose(label_image[0, :xb], background)))):
        warnings.warn('Region that has to be processed in this tile has reached outer border. '
                      'Overlap may not be sufficient.')

    outer = f(
        label_image[:ya, :xb],  # top
        label_image[ya:yb, :xa]  # left
    )
    red = f(
        label_image[yb:],  # bottom
        label_image[:yb, xb:]  # right
    )
    green = f(
        label_image[ya:yb, xa],  # left
        label_image[ya, xa:xb]  # top
    )
    label_image[np.isin(label_image, red) | (np.isin(label_image, outer) & ~np.isin(label_image, green))] = background
    return label_image


def calc_min_max_cart(
        mask: np.ndarray,
        image: np.ndarray,
        n_iter: int = 1,
        sigma: float = 2.,
        footprint: int = 5
) -> np.ndarray:
    filtered = invert(image)
    for _ in range(n_iter):
        filtered = gaussian(filtered, sigma=sigma)

    filtered[np.logical_not(mask)] = 0
    fp = morph.disk(footprint)
    fp_zero = fp.copy()
    shape = np.array(fp.shape) // 2
    fp_zero[shape[0], shape[1]] = 0
    max_img = maximum_filter(filtered, footprint=fp, mode="constant", cval=-1)
    max_img_zero = maximum_filter(filtered, footprint=fp_zero, mode="constant", cval=-1)
    I = np.zeros_like(max_img, dtype=np.int8)
    I[(max_img != max_img_zero)] = -1
    return I


def mean_magnitude(img, mask=None, sigma=0.5, size=3):
    """Calculate the mean magnitude of the contour of a region.

    Args:
        img (array_like): Image containing the object for which the mean magnitude should be computed.
        mask (array_like): Mask of the object.
        sigma: sigma value for the gaussian smoothing.abs
    """
    gradient = np.gradient(filters.gaussian(img.astype("float"), sigma=sigma))
    mag = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    cnt = find_contours(mask, 0.5)[0].astype(int)
    mag = maximum_filter(mag, size=size)
    mean = mag[cnt[:, 0], cnt[:, 1]].mean()
    return mean
