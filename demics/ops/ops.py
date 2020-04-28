from .util import exclude_regions, calc_min_max_cart, mean_magnitude
from ..controller import Controller, LabelAggregate, LabelVisAggregate
from ..meta import Model
from ..util import pack_list, unpack_list
import pandas as pd
import numpy as np
import cv2
import gc
from skimage.morphology import label as sk_label
from skimage.measure import label as skm_label, regionprops
import skimage.morphology as morph
from scipy import ndimage as ndi

VERBOSE = False


class Op:
    def __init__(self, overlap, aggregate, tile_size, *args, verbose=VERBOSE, **kwargs):
        self.overlap = overlap
        self.aggregate = aggregate
        self.tile_size = tile_size
        self.args = args
        self.kwargs = kwargs
        self.con = Controller(verbose=verbose)

    def __call__(self, func):
        def wrapper(inputs, *args, overlap=None, tile_size=None, **kwargs):  # multi-input ops as tuples of inputs
            overlap = self.overlap if overlap is None else overlap
            tile_size = self.tile_size if tile_size is None else tile_size
            if overlap is None:
                raise ValueError('Overlap must be defined for this operation.')
            inputs, stat = pack_list(inputs)
            res = self.con(callback=func, i_objects=inputs, overlap=overlap, args=args,
                           kwargs=kwargs, aggregate=self.aggregate, tile_size=tile_size)
            return unpack_list(res, stat)

        return wrapper


class AtomicOp(Op):
    def __init__(self, aggregate=None, tile_size=None, *args, verbose=VERBOSE, **kwargs):
        super().__init__(overlap=0, aggregate=aggregate, tile_size=tile_size, *args, verbose=verbose, **kwargs)


class NonAtomicOp(Op):
    def __init__(self, overlap=None, aggregate=None, tile_size=None, *args, verbose=VERBOSE, **kwargs):
        super().__init__(overlap=overlap, aggregate=aggregate, tile_size=tile_size, *args, verbose=verbose, **kwargs)


class TfModel(NonAtomicOp):
    def __init__(self, overlap=None, aggregate=None, tile_size=1024, *args, verbose=VERBOSE, **kwargs):
        super().__init__(overlap=overlap, aggregate=aggregate, tile_size=tile_size, *args, verbose=verbose, **kwargs)
        self.model = None

    def __call__(self, func):
        def wrapper(inputs, model: str, overlap=None, gpu=True, *args,
                    **kwargs):  # multi-input ops work with tuples of inputs
            model_dir = model
            model = Model(model_dir)
            overlap = self.overlap if overlap is None else overlap
            if overlap is None:
                overlap = model.padding
            if overlap is None:
                raise ValueError('Overlap must be defined for this operation.')
            if gpu is False or self.con.has_gpu:
                inputs, stat = pack_list(inputs)
                if model.is_tf1():
                    res = self._handle_tf1(model, model_dir, func, inputs, overlap, gpu, args, kwargs)
                elif model.is_tf2():
                    raise NotImplementedError('Tf2 not supported yet.')
                else:
                    raise ValueError('Unknown framework')
                res = unpack_list(res, stat)
            else:
                res = None
            return res

        return wrapper

    def _handle_tf1(self, model, model_dir, func, inputs, overlap, gpu, args, kwargs):
        self._start_tf1(model.directory, gpu)
        kwargs.update({
            "model": model_dir,
            "_session": self.session,
            "_input_tensor": model.inputs_str,
            "_output_tensor": model.outputs_str,
            "_preprocessing": model.preprocessing,
        })
        res = self.con(callback=func, i_objects=inputs, overlap=overlap, args=args,
                       kwargs=kwargs, aggregate=self.aggregate, tile_size=self.tile_size,
                       gpu=gpu)
        if self.con.has_gpu:
            self._stop_tf1()
        return res

    def _start_tf1(self, checkpoint, gpu):
        try:
            from tensorflow import Session, saved_model, ConfigProto
            config = ConfigProto()
            config.gpu_options.allow_growth = True
            if gpu is False or self.con.has_gpu is False:
                config.gpu_options.visible_device_list = ''
            else:
                config.gpu_options.visible_device_list = str(self.con.local_gpu_rank)
            print(f"Node {self.con.node_rank} rank {self.con.rank_gpu} starts session with GPU-ID:",
                  config.gpu_options.visible_device_list)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Could not import from TensorFlow 1.X')

        self.session = Session(config=config)
        saved_model.loader.load(self.session, ['serve'], checkpoint)

    def _stop_tf1(self):
        if self.session is not None:
            self.session.close()
            self.session = None


@TfModel()
def tf_model(
        inputs: np.ndarray,
        model: str,
        **kwargs
) -> np.ndarray:
    _preprocessing = kwargs['_preprocessing']
    _input_tensor = kwargs['_input_tensor']
    _output_tensor = kwargs['_output_tensor']
    _session = kwargs['_session']

    # Ensure shape
    if inputs.ndim == 2:  # assuming no batch processing, as inputs is wrapped with list
        inputs = inputs[..., None]

    # Preprocessing
    if _preprocessing is not None:
        for prep in _preprocessing:
            if prep == '0..255_0..1':
                try:
                    inputs /= 255
                except TypeError:
                    inputs = inputs / 255
            elif prep == '0..255_-1..1':
                try:
                    inputs /= 127.5
                except TypeError:
                    inputs = inputs / 127.5
                inputs -= 1
            elif prep == 'mean_std':
                try:
                    inputs /= np.std(inputs)
                except TypeError:
                    inputs = inputs / 255
                inputs -= np.mean(inputs)

    # Ensure shape
    divisor = 32
    dims = 2
    pad_width = []
    for j, s in enumerate(inputs.shape):
        if j >= dims:
            pad_width.append([0, 0])
        else:
            d = int(np.ceil(s / divisor)) * divisor - s
            a = d // 2
            b = d - a
            pad_width.append([a, b])
    inputs = np.pad(inputs, pad_width=pad_width)
    res = _session.run(_input_tensor, {_output_tensor: [inputs]})[0]
    res = res[tuple([slice(a, -b if b > 0 else None) for a, b in pad_width[:dims]])]
    return res


@NonAtomicOp()
def adaptive_threshold(
        inputs: np.ndarray,
        smooth_diameter: int = 21,
        smooth_sigma_color: int = 20,
        smooth_sigma_space: int = 2,
        block_size: int = 31,
        constant: float = 30.,
        **kwargs
):
    if smooth_diameter > 0:
        inputs = cv2.bilateralFilter(inputs, smooth_diameter, smooth_sigma_color, smooth_sigma_space)
    return cv2.adaptiveThreshold(inputs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, constant)


@AtomicOp()
def threshold(
        inputs,
        thresh: float,
        vmax: int = 1,
        method=cv2.THRESH_BINARY,
        fill: bool = False,
        **kwargs
):
    t, res = cv2.threshold(
        src=inputs,
        thresh=thresh,
        maxval=vmax,
        type=method
    )
    if fill:
        res = ndi.binary_fill_holes(res)
    return res


@NonAtomicOp(aggregate=LabelVisAggregate)
def scramble(
        label_image: np.ndarray,
        vmin: int = 125,
        meta: dict = None,
        **kwargs
):
    if meta is not None and 'yrange_ex' in meta.keys() and 'xrange_ex' in meta.keys():
        label_image = exclude_regions(label_image, y_range_ex=meta['yrange_ex'], x_range_ex=meta['xrange_ex'])
    uniques = np.unique(label_image)
    res = np.zeros(label_image.shape[:2] + (3,), dtype=np.uint8) + LabelVisAggregate.back_color
    for u in uniques:
        if u == 0:
            continue
        selection = label_image == u
        res[selection] = list(
            zip(*[((label_image[selection] * np.random.randint(vmin, 255) % (255 - vmin)) + vmin) for _ in range(3)]))
    return res


@NonAtomicOp(aggregate=LabelAggregate)
def label(
        mask: np.ndarray,
        background: int = 0,
        connectivity: int = 2,
        meta: dict = None,
        **kwargs
):
    label_image = sk_label(mask, background=background, connectivity=connectivity)
    if meta is not None and 'yrange_ex' in meta.keys() and 'xrange_ex' in meta.keys():
        label_image = exclude_regions(label_image, y_range_ex=meta['yrange_ex'], x_range_ex=meta['xrange_ex'])
    return label_image


@NonAtomicOp(aggregate=LabelAggregate)
def watershed(
        mask: np.ndarray,
        image: np.ndarray,
        n_iter: int = 1,
        sigma: float = 2.,
        footprint: int = 5,
        meta: dict = None,
        **kwargs
) -> np.ndarray:
    if np.max(mask) == 0:
        labeled = np.zeros_like(image, dtype=np.uint32)
        return labeled
    mmcart = calc_min_max_cart(mask, image, n_iter=n_iter, sigma=sigma, footprint=footprint)
    markers = skm_label(mmcart)
    label_image = morph.watershed(image, markers=markers, mask=mask).astype(np.uint32)
    if meta is not None and 'yrange_ex' in meta.keys() and 'xrange_ex' in meta.keys():
        label_image = exclude_regions(label_image, y_range_ex=meta['yrange_ex'], x_range_ex=meta['xrange_ex'])
    return label_image


@NonAtomicOp()
def properties(
        # inputs,
        label_image: np.ndarray,
        image: np.ndarray,
        meta: dict = None,
        **kwargs
) -> pd.DataFrame:
    # label_image, image = inputs
    gc_counter = 0
    gc_max_props = 50
    x_start, x_end = meta["xrange"]
    y_start, y_end = meta["yrange"]
    x_padded_start = meta["xrange_padded"][0]
    y_padded_start = meta["yrange_padded"][0]
    regions = regionprops(label_image, image)
    boxes = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        y, x = props.centroid
        # ignore regions with centroid out of xrange / yrange - these belong to neighboring tile
        if x + x_padded_start < x_start or x + x_padded_start >= x_end:
            continue
        if y + y_padded_start < y_start or y + y_padded_start >= y_end:
            continue
        _minr = int(minr)
        _minc = int(minc)
        _maxr = int(maxr + 1)
        _maxc = int(maxc + 1)

        if _minr - 1 >= 0:
            _minr -= 1
        if _minc - 1 >= 0:
            _minc -= 1

        # additional features
        area = props.area
        perimeter = props.perimeter
        circularity = 0 if perimeter == 0 else 4. * np.pi * area / perimeter ** 2
        mean = props.mean_intensity
        diameter = props.equivalent_diameter
        labeled_patch = label_image[_minr:_maxr, _minc:_maxc].copy()
        labeled_patch[labeled_patch != props.label] = 0
        img_patch = image[_minr:_maxr, _minc:_maxc]
        magnitude = mean_magnitude(img_patch, labeled_patch)

        b = np.array(
            [props.label, x + x_padded_start, y + y_padded_start, minr + y_padded_start, minc + x_padded_start, maxr +
             y_padded_start, maxc + x_padded_start, area, perimeter, magnitude, circularity, mean, diameter],
            dtype=np.float32)
        boxes.append(b)

        # Invoke the garbage collector (TODO: Usage of float32 instead of float64, still keeping the GC invocations)
        gc_counter += 1
        if gc_counter >= gc_max_props:
            gc_counter = 0
            gc.collect()

    gc.collect()

    return pd.DataFrame(boxes, dtype=np.float32,
                        columns=["cell_label", "x", "y", "minr", "minc", "maxr", "maxc", "area",
                                 "perimeter", "magnitude", "circularity", "mean", "diameter"])
