"""I/O functionality.

This file defines I/O related functions and classes.
"""

from ..system.resolver import Resolver
from ..util import probe_shape, probe_size, Printable
from typing import Union, List, Tuple, Callable, Any
from pytiff import Tiff
from h5py import File as H5
from skimage import color
from PIL import Image
import pandas as pd
import numpy as np
import warnings
import imageio
import gc


class Tile:
    """
    Instances of this class represent a single tile.
    """

    def __init__(self, x, y, height, width, context_shape, padding=36):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.padding = padding
        self.context_shape = context_shape
        self.caret = 0

        # Assuming result has same shape as initial padded crop
        self.a_range = [self.y, min(self.context_shape[0], self.y + height)]
        self.a_range_pad = [max(0, self.a_range[0] - padding), min(self.context_shape[0], self.a_range[1] + padding)]
        self.a_range_extract = [self.a_range[0] - self.a_range_pad[0],
                                self.a_range[0] - self.a_range_pad[0] + self.a_range[1] - self.a_range[0]]

        self.b_range = [self.x, min(self.context_shape[1], self.x + width)]
        self.b_range_pad = [max(0, self.b_range[0] - padding),
                            min(self.context_shape[1], self.b_range[1] + padding)]  # to read from input with pad
        self.b_range_extract = [self.b_range[0] - self.b_range_pad[0],  # to crop from padded crop
                                self.b_range[0] - self.b_range_pad[0] + self.b_range[1] - self.b_range[0]]

        self.slices = (slice(*self.a_range), slice(*self.b_range))  # for writing results to result file
        self.slices_pad = (slice(*self.a_range_pad), slice(*self.b_range_pad))  # for reading inputs from input file
        self.slices_extract = (slice(*self.a_range_extract),
                               slice(*self.b_range_extract))  # for reading w/o padding from results


class TensorTile(Tile):

    def __init__(self, io_obj, x, y, height, width, context_shape, padding=36):
        super().__init__(x, y, height, width, context_shape, padding)
        self.obj = io_obj
        isli = isinstance(self.obj, (list, tuple))
        ob = list(self.obj) if isli else [self.obj]
        for i, o in enumerate(ob):
            assert isinstance(o, Tensor)
            if o.array_mode():
                ob[i] = Tensor(o[self.slices_pad])
        self.obj = type(self.obj)(ob) if isli else ob[0]
        self._results = None

    def _prep(self, state):
        if isinstance(self.obj, Tensor):
            inputs = [self.obj]
        elif isinstance(self.obj, (tuple, list)):
            inputs = self.obj
        else:
            raise ValueError(f"Could not handle input type: {type(self.obj)}.")
        for i in inputs:
            i.reveal(state)

    def pre_transport(self):
        """Call before pickle or MPI send"""
        self._prep(True)

    def post_transport(self):
        """Call after unpickle or MPI receive"""
        self._prep(False)

    def inputs(self, copy=False):
        if isinstance(self.obj, Tensor):
            inputs = (self.obj,)
            dtype = None
        elif isinstance(self.obj, (tuple, list)):
            inputs = self.obj
            dtype = type(self.obj)
        else:
            raise ValueError(f"Could not handle input type: {type(self.obj)}.")
        res = [(o if o.array_mode() else o.lazy_load(self.slices_pad)).numpy() for o in inputs]
        if copy:
            res = [np.copy(o) for o in res]
        if dtype is not None:
            res = dtype(res)
        return res

    def set_results(self, v: Union[np.ndarray, pd.DataFrame], include_overlap=False):
        if isinstance(v, np.ndarray):
            expected_size = (abs(np.subtract.reduce(self.a_range_pad[:2])),
                             abs(np.subtract.reduce(self.b_range_pad[:2])))
            if v.shape[:2] != expected_size:
                raise ValueError('Expected processed data to have the same size as input data, but found '
                                 f'{expected_size} for input and {v.shape[:2]} for result.')
            self._results = v if include_overlap else v[self.slices_extract]
        elif isinstance(v, pd.DataFrame):
            self._results = v
        else:
            raise ValueError(f'Could not handle data type of result: {str(type(v))}.')

    def results(self, discard=True):  # assuming result has same shape as inputs
        if discard:
            self.obj = None
        return self._results

    def handover(self, target, aggregate=None, with_overlap=False, non_value=None):
        slices = self.slices_pad if with_overlap else self.slices
        res = self.results()
        if res is None:
            raise ValueError('No results to hand over.')

        if isinstance(res, pd.DataFrame):
            assert target is None or isinstance(target, pd.DataFrame)
            target = pd.concat((target, res), ignore_index=True)
            if aggregate is not None:
                raise NotImplementedError('Aggregate not defined for pandas DataFrames')
        elif isinstance(res, np.ndarray):
            if aggregate is not None:
                res = aggregate.feed(res)

            if non_value is None:
                target[slices] = res
            else:
                selection = ~np.isclose(res, non_value)
                target[slices][selection] = res[selection]
        else:
            raise ValueError(f'Could not handle results.')
        return target


class Tiling:
    """
    Helper class for Tile handling.

    Notes:
        Currently assuming, that processing of crops does not change spatial dimensions.

    Examples:
        results = np.zeros_like(image)
        for tile in Tiling(shape=shape, tile_height=h, tile_width=w, padding=16):
            padded_crop = image[tile.slices_pad]
            processed_crop = process(padded_crop)
            results[tile.slices] = processed_crop[tile.slices_extract]
    """

    def __init__(self, shape, tile_height, tile_width, padding, worker_id=None, workers=None, tile_class=Tile):
        assert len(shape) >= 2
        self.shape = shape
        if None in (tile_height, tile_width):
            self.tile_height = shape[0]
            self.tile_width = shape[1]
        else:
            self.tile_height = tile_height
            self.tile_width = tile_width
        self.padding = padding
        self.worker_id = worker_id
        self.workers = workers
        self.caret = 0
        self.TileClass = tile_class
        w_tup = (worker_id, workers)
        if w_tup != (None, None) and None in w_tup:
            warnings.warn('Tiling: Either `worker_id` or `workers` was None while the other was set. '
                          'Note that tiling is only worker specific if both arguments are set.')
        self._tiles = []
        self._tile_kwargs = []
        counter = 0
        for a in range(0, self.shape[0], self.tile_height):
            for b in range(0, self.shape[1], self.tile_width):
                counter += 1
                if None in (worker_id, workers) or (counter % self.workers) == worker_id:
                    self._append(
                        x=b,
                        y=a,
                        height=self.tile_height,
                        width=self.tile_width,
                        context_shape=self.shape[:2],
                        padding=self.padding
                    )

    def _append(self, x, y, height, width, context_shape, padding):
        self._tile_kwargs.append({
            'x': x,
            'y': y,
            'height': height,
            'width': width,
            'context_shape': context_shape,
            'padding': padding
        })

    def __getitem__(self, item):
        return self.TileClass(**self._tile_kwargs[item])

    def __len__(self):
        return len(self._tile_kwargs)

    def __iter__(self):
        self.caret = 0
        return self

    def __next__(self):
        if self.caret < len(self):
            v = self[self.caret]
            self.caret += 1
            return v
        else:
            raise StopIteration


class TensorTiling(Tiling):

    def __init__(self, io_obj, shape, tile_height, tile_width, padding, worker_id=None, workers=None):
        self.io_obj = io_obj
        super().__init__(shape, tile_height, tile_width, padding, worker_id, workers, tile_class=TensorTile)

    def _append(self, x, y, height, width, context_shape, padding):
        self._tile_kwargs.append({
            'x': x,
            'y': y,
            'height': height,
            'width': width,
            'context_shape': context_shape,
            'padding': padding,
            'io_obj': self.io_obj
        })


color_conversions = {
    color.gray2rgb.__name__: (color.gray2rgb, {0: 3}),
    color.hed2rgb.__name__: (color.hed2rgb, {3: 3}),
    color.rgb2gray.__name__: (color.rgb2gray, {3: 0, 4: 0}),
    color.rgb2hed.__name__: (color.rgb2hed, {3: 3}),
    color.rgb2lab.__name__: (color.rgb2lab, {3: 3}),
    color.rgb2rgbcie.__name__: (color.rgb2rgbcie, {3: 3}),
    color.rgb2xyz.__name__: (color.rgb2xyz, {3: 3}),
    color.rgb2ycbcr.__name__: (color.rgb2ycbcr, {3: 3}),
    color.rgb2ydbdr.__name__: (color.rgb2ydbdr, {3: 3}),
    color.rgb2yiq.__name__: (color.rgb2yiq, {3: 3}),
    color.rgb2ypbpr.__name__: (color.rgb2ypbpr, {3: 3}),
    color.rgb2yuv.__name__: (color.rgb2yuv, {3: 3}),
    color.rgbcie2rgb.__name__: (color.rgbcie2rgb, {3: 3}),
    color.xyz2lab.__name__: (color.xyz2lab, {3: 3}),
    color.xyz2rgb.__name__: (color.xyz2rgb, {3: 3}),
    color.ycbcr2rgb.__name__: (color.ycbcr2rgb, {3: 3}),
    color.ydbdr2rgb.__name__: (color.ydbdr2rgb, {3: 3}),
    color.yiq2rgb.__name__: (color.yiq2rgb, {3: 3}),
    color.ypbpr2rgb.__name__: (color.ypbpr2rgb, {3: 3}),
    color.yuv2rgb.__name__: (color.yuv2rgb, {3: 3}),
}


# HANDLED_FUNCTIONS = {}


class Tensor(np.ndarray):
    """
    Tensor Class.

    Examples:
        i = Tensor.from_tif('mytif.tif')
        crop = i[:500, :500]

        i = Tensor.from_h5('myh5.h5', dataset='image')
        crop = i[:500, :500]

        i = Tensor.from_tifs('z-scan/scan_slice[0-9][0-9].tif')
        i.min_reduce()
        crop = i[:500, :500]

    Notes:
        Reading:
            From file:
                - limited_memory: read only from disk what is requested
                - unlimited_memory: read everything from disk on first read access and switch to ndarray mode
            From ndarray:
                - everything is in memory anyway, so always read from memory
        Writing:
            From hdf5 file with single dataset:
                - limited_memory: Write directly to disk
                - unlimited_memory: Always write to memory. Data is only flushed to disk on `self.close` or
                    `self.__exit__`
            From ndarray:
                - everything is in memory anyway, so write to ndarray. Call `self.to_h5` to flush.
            Others:
                - no write access
    """

    def __new__(
            cls,
            array,
            dtype=None,
            order=None,
            lazy_read=True,
            tile_size=1024,
            shape=None,
            *args,
            **kwargs
    ):
        assert isinstance(lazy_read, bool)
        if array is None:
            o = super(Tensor, cls).__new__(cls, *args, shape=(1,) if shape is None else shape, **kwargs)
            o._dummy = True
            o._intermediate_dummy = False
        else:
            o = np.asarray(array, dtype=dtype, order=order).view(Tensor)
            o._dummy = False
            o._intermediate_dummy = False
        o.__data = None
        o._block_trigger = False
        o.altered_type = False
        o._load = None
        o.filename = None  # used for single h5/tif files
        o.filenames = None  # used for multiple tif files
        o.dataset = None  # used for single h5 dataset
        o.datasets = None  # used for multiple h5 datasets
        o.src = None
        o._page = 0
        o._is_volume = None
        o._keep = False

        o.op_chain = []
        o.op_params = []
        o.op_shape = []

        o.lazy_read = lazy_read
        o.tile_size = tile_size
        o.__spatial_dims = 2  # number of spatial dimensions per input
        o.file_handles = {}

        o.inferred_shape = None
        o.inferred_ndim = None
        o.__inferred_dtype = None

        o._debug_verbose = False
        o._reveal = False
        return o

    def reveal(self, status):
        """Reveal actual data.

        When an instance is created on basis of data from files the data is usually loaded once it is first required.
        Before this instance contains dummy data. For some applications the actual properties of that dummy data
        is needed. This method can be used to reveal actual properties.

        Args:
            status: Whether to reveal actual instance properties. If True always return actual properties, like size,
                shape, ndim, etc. Otherwise use inferred properties if necessary.

        Returns:

        """
        self._reveal = status

    def _debug(self, *s):
        if getattr(self, '_debug_verbose', False):
            print(*s)

    def __array_finalize__(self, o):
        if o is None:
            return

        if isinstance(o, Tensor):
            self.__dict__.update(o.__dict__)

    def __reduce__(self):
        """Reduce instance. Required for proper pickling."""
        pickled_state = super().__reduce__()  # reduce tuple from parent
        new_state = pickled_state[2] + (self.__dict__,)  # custom reduce tuple with entire dict for custom setstate
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        """Set state. Required for proper pickling."""
        self.__dict__.update(state[-1])  # update dict from custom reduce tuple
        super().__setstate__(state[0:-1])  # update parent with parent reduce tuple

    @staticmethod
    def _convert_array(inputs, itype, otype):
        if isinstance(inputs, dict):
            res = {}
            for key, val in inputs.items():
                res[key] = Tensor._convert_array(val, itype, otype)
            return res
        elif isinstance(inputs, (tuple, list)):
            dtype_in = type(inputs)
            return dtype_in([Tensor._convert_array(i, itype, otype) for i in inputs])
        elif isinstance(inputs, itype) and inputs.__class__ == itype:
            return inputs.view(otype)
        else:
            return inputs

    def __str__(self):
        if self.array_mode():
            return super(Tensor, self).__str__()
        else:
            return f'Tensor with source: {Printable.p(self.src)}. Data not loaded yet.'

    def __repr__(self):
        if self.array_mode():
            return super(Tensor, self).__repr__()
        else:
            return f'Tensor with source: {Printable.p(self.src)}. Data not loaded yet.'

    def numpy(self):
        return self.view(np.ndarray)

    def __len__(self):
        s = self.shape
        if len(s) > 0:
            return s[0]
        return 0

    @property
    def shape(self):
        if (self.array_mode() and getattr(self, '_intermediate_dummy', False) is False) or getattr(self, '_reveal',
                                                                                                   False):
            return super().shape
        return self.inferred_shape

    @shape.setter
    def shape(self, shape):
        super().shape = shape

    @property
    def inferred_dtype(self):
        return self.__inferred_dtype

    @property
    def _mask(self):
        # Commonly used to test for MaskedArray. Assuming this is done in a context that requires data to be loaded.
        self._try_load_trigger()
        raise AttributeError('No such attribute')

    @inferred_dtype.setter
    def inferred_dtype(self, dtype):
        self.__inferred_dtype = dtype
        if not self.array_mode():
            self.dtype = dtype

    @property
    def ndim(self):
        if (self.array_mode() and getattr(self, '_intermediate_dummy', False) is False) or getattr(self, '_reveal',
                                                                                                   False):
            return super().ndim
        return self.inferred_ndim

    @ndim.setter
    def ndim(self, ndim):
        super().ndim = ndim

    @property
    def size(self):
        if (self.array_mode() and getattr(self, '_intermediate_dummy', False) is False) or getattr(self, '_reveal',
                                                                                                   False):
            return super().size
        return self.inferred_size

    @size.setter
    def size(self, size):
        super().size = size

    def min_reduce(self, axis=2, keepdims=False):
        self.callback_op(np.min, axis=axis, keepdims=keepdims,
                         shape_info={'axis': axis, 'dim': 1 if keepdims else 0})

    def max_reduce(self, axis=2, keepdims=False):
        self.callback_op(np.max, axis=axis, keepdims=keepdims,
                         shape_info={'axis': axis, 'dim': 1 if keepdims else 0})

    def mean_reduce(self, axis=2, keepdims=False):
        self.callback_op(np.mean, axis=axis, keepdims=keepdims,
                         shape_info={'axis': axis, 'dim': 1 if keepdims else 0})

    def median_reduce(self, axis=2, keepdims=False):
        self.callback_op(np.median, axis=axis, keepdims=keepdims,
                         shape_info={'axis': axis, 'dim': 1 if keepdims else 0})

    @staticmethod
    def _run_callback_op(inputs, callback, kwargs):
        res = callback(inputs, **kwargs)
        if isinstance(res, np.ndarray) and not isinstance(res, Tensor):
            res = res.view(Tensor)
        return res

    def _apply_callbacks(self, inputs):
        for c, kp, s in zip(self.op_chain, self.op_params, self.op_shape):
            self._debug("Apply callback", c)
            inputs = self._run_callback_op(inputs, c, kp)
        return inputs

    def callback_op(self, callback: Callable[[np.ndarray, Any], np.ndarray], shape_info: dict = None,
                    dtype_info=None, **kwargs):

        # Try to exec op
        if self.array_mode():
            self._run_callback_op(self, callback, kwargs)

        self.op_chain.append(callback)
        self.op_params.append(kwargs)
        self.op_shape.append(shape_info)

        # Update shape
        if shape_info is not None:
            axis, dim = shape_info['axis'], shape_info['dim']
            shape = list(self.shape)
            if dim == 0:
                del shape[axis]
            elif isinstance(dim, tuple):
                shape += list(dim)
            else:
                shape[axis] = dim
            if dim is None:
                warnings.warn('Op has undefined effect on shape!')
            self._set_shape(tuple(shape))

        if dtype_info is not None:
            self.dtype = dtype_info

        return self

    def convert_color(self, method):
        try:
            callback, s = color_conversions[method]
        except KeyError:
            raise ValueError(f'Unknown method: {method}')

        try:
            if 0 in s.keys():
                new_dim = s[0]
            else:
                new_dim = s[self.shape[-1]]
        except KeyError:
            raise ValueError('Input shape not compatible.')

        self.callback_op(callback, shape_info={'axis': -1, 'dim': new_dim})

    def _set_shape(self, shape):
        """Update inferred shape of this object.

        Args:
            shape: Tuple or list of new dimensions. If dimension is None the dimension inherits its value from the
                current shape.

        """
        assert isinstance(shape, (list, tuple))
        shape = list(shape)
        for i, s in enumerate(shape):
            if s is None:
                try:
                    shape[i] = self.inferred_shape[i]
                except IndexError:
                    raise ValueError('Invalid shape: Did not match with previous shape.')
        self.inferred_shape = tuple(shape)
        self.inferred_ndim = len(self.inferred_shape)
        self.inferred_size = np.multiply.reduce(self.inferred_shape)

    def array_mode(self):
        return getattr(self, '_dummy', False) is False

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        o = super().astype(dtype=dtype, order=order, casting=casting, subok=subok, copy=copy)
        o.altered_type = True
        return o

    def totype(self, dtype):
        self._totype(self, dtype)

    @staticmethod
    def _totype(a, dtype):
        """Converts data type of given ndarray.

        Convert data type of the given ndarray instance.
        In contrast to `array.astype(..)` this method does not create a new instance. However, it is possible that
        this function is required to temporarily copy the instance's data. This is especially relevant for very
        large arrays.

        Notes:
            - If the byte size of source and target data type differs a temporary copy of the intance's data is created.
              This can lead to OOM for large arrays.
            - If the byte size of source and target data type are equal the conversion is done via a view.

        Args:
            a: Input array.
            dtype: Target data type.

        """
        max_v = 64
        oldtype = a.dtype
        newtype = np.dtype(dtype)
        oi, ni = oldtype.itemsize, newtype.itemsize
        if oi == ni:
            # Same itemsize
            proxy = a.view(newtype)
            proxy[:] = a
            a.dtype = newtype
        elif oi < ni or oi > ni:
            # Different itemsize
            tmp = np.array(a.view(np.ndarray))
            try:
                # Try to populate new dtype
                a.dtype = newtype
            except ValueError:
                # Except target dtype is larger and size incompatible
                tmp_shape = np.array(a.shape)
                tmp_shape[-1] += max_v - (tmp_shape[-1] % max_v)
                a.resize(tmp_shape, refcheck=False)
                a.dtype = newtype
            a.resize(tmp.shape, refcheck=False)
            a[:] = tmp
        else:
            raise ValueError('Unexpected error.')

    def load(self):
        self._try_load_trigger()

    def _try_load_trigger(self):
        if not self.array_mode() and not self._block_trigger:
            try:
                if not callable(self._load):
                    raise ValueError('Instance not properly build.')
                self._debug("Load all triggered")
                self._dummy = False
                self._intermediate_dummy = True
                v = self._load()
                self._set_data(v)
                self._intermediate_dummy = False
            except ValueError as e:
                raise ValueError('This operation does not support lazy Tensors. Data must be loaded before this '
                                 'operation is applied.')

    def __getitem__(self, item):
        # self._debug("__getitem__", item, type(item), type(self), id(self), "array_mode:", self.array_mode())
        self._try_load_trigger()
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        self._debug("__setitem__", key, value)
        self._try_load_trigger()
        return super().__setitem__(key, value)

    def lazy_load(self, item):
        """Lazy load of specified data.

        Tries to load specified data without triggering a `load_all`.

        Args:
            item: Item to load

        Returns:
            Data
        """
        if self.array_mode():
            res = super().__getitem__(item)
        else:
            try:
                res = Tensor(self._load(item))
            except TypeError:
                raise TypeError('Load function of Tensor object not build properly.')
        return res

    @classmethod
    def empty(cls, shape, dtype=None, order='C'):
        return cls(np.empty(shape=shape, dtype=dtype, order=order))

    @classmethod
    def zeros(cls, shape, dtype=None, order='C'):
        return cls(np.zeros(shape=shape, dtype=dtype, order=order))

    @classmethod
    def array(cls, array):
        return cls(array)

    @classmethod
    def from_file(
            cls,
            src: str,
            n: int = -1,
            volume: bool = None,
            id_range: Union[List[int], Tuple[int], str] = None,
            prefix: str = '_Slice',
            numbers: int = 2,
            delimiter: str = ':',
            lazy_read=True,
            page: int = 0,
            keep: bool = False
    ):
        """

        Args:
            src: Source string. Can be filename, filename pattern, filename template or a string with
                delimiter notation denoting <h5 filename><delimiter><dataset name> where <dataset name>
                can again be the name, pattern or template of a dataset.
            n:
            volume:
            id_range:
            prefix:
            numbers:
            delimiter:
            lazy_read:
            page: Tif page. Only used if source is a tif file.

        Returns:

        """
        resolver = Resolver()
        single = volume if volume is None else not volume
        o = cls(lazy_read=lazy_read, array=None)
        o._page = page
        o._keep = keep
        f, d = resolver.mixed(
            src=src,
            n=n,
            single=single,
            id_range=id_range,
            prefix=prefix,
            numbers=numbers,
            delimiter=delimiter
        )
        if resolver.single_file():
            o.filename = o.src = f
            o._is_volume = False
        else:
            o.filenames = o.src = f
            o._is_volume = True
        o.src = f
        rd = resolver.single_dataset()
        if rd is True:
            o.dataset = d
        elif rd is False:
            o.datasets = d

        if resolver.all_h5():
            if resolver.single_dataset():
                o._load = o._load_h5_dataset
            else:
                o._load = o._load_h5_datasets
        elif resolver.all_tif():
            if resolver.single_file():
                o._load = o._load_tif
            else:
                o._load = o._load_tifs
        elif resolver.all_png() or resolver.all_jpg():
            if resolver.single_file():
                o._load = o._load_pil_img
            else:
                o._load = o._load_pil_imgs
        else:
            raise ValueError('Could not handle file formats.')
        o._load(explore=True)
        return o

    def _load_tif(
            self,
            item=None,
            filename=None,
            keep=None,
            explore=False,
            fractal=False
    ):
        """

        Notes:
            PyTiff is not thread safe!

        Args:
            item:
            filename:
            keep:
            explore:
            fractal:

        Returns:

        """
        # TODO: PyTiff does not properly handle exceeding bounds! Users might expect behaviour similar to numpy. PyTiff may show integer overflow and raise MemoryError as a consequence.
        # TODO: Fix this either in PyTiff or check for bounds here.
        filename = self.filename if filename is None else filename
        keep = self._keep if keep is None else keep
        fh_key = filename + f'::{str(self._page)}'
        res = None
        if keep:
            try:
                t = self.file_handles[fh_key]
            except KeyError:
                self.file_handles[fh_key] = t = Tiff(filename)
            t.set_page(self._page)
            if explore:
                self._update_meta(t)
            else:
                res = t[:] if item is None else t[item]
        else:
            with Tiff(filename) as t:
                t.set_page(self._page)
                if explore:
                    self._update_meta(t)
                else:
                    res = t[:] if item is None else t[item]

        if not fractal and not explore:
            res = self._apply_callbacks(res)

            if self._all_item(item):
                self._set_data(res)
        return res

    def _load_tifs(
            self,
            item=None,
            explore=False
    ):
        assert self.filenames is not None
        return self._multi_loader(
            loader_callback=lambda item, name: self._load_tif(item=item, filename=name, keep=False, explore=explore,
                                                              fractal=True),  # TODO: keep changed to False
            name_list=self.filenames,
            item=item,
            explore=explore
        )

    @staticmethod
    def _all_item(item):
        return item in (None, slice(None))

    def _load_pil_img(
            self,
            item=None,
            filename=None,
            keep=False,
            explore=False,
            fractal=False
    ):
        filename = self.filename if filename is None else filename

        def do(t):
            if explore:
                self._update_meta(t)
                res = None
            else:
                res = np.array(t)
                if item is not None:
                    res = res[item]
            return res

        if keep:
            try:
                t = self.file_handles[filename]
            except KeyError:
                self.file_handles[filename] = t = Image.open(filename, 'r')
            res = do(t)
        else:
            with Image.open(filename, 'r') as t:
                res = do(t)

        if not fractal and not explore:
            res = self._apply_callbacks(res)

            if self._all_item(item):
                self._set_data(res)
        return res

    def _load_pil_imgs(
            self,
            item=None,
            explore=False
    ):
        assert self.filenames is not None
        return self._multi_loader(
            loader_callback=lambda item, name: self._load_pil_img(item=item, filename=name, keep=True, explore=explore,
                                                                  fractal=True),
            name_list=self.filenames,
            item=item,
            explore=explore
        )

    def _update_meta(self, handle, warn=True):
        self._debug("Update meta", handle, warn)
        iitems = ['shape', 'dtype', 'size', 'ndim', 'nbytes', 'pages']
        oitems = ['inferred_shape', 'inferred_dtype', 'inferred_size', 'inferred_ndim', 'inferred_nbytes', 'pages']
        for i, o in zip(iitems, oitems):
            v = getattr(handle, i, None)

            # Special treatment
            if i == 'size' and isinstance(v, (tuple, list)):  # handle pytiff bug + PIL.Image
                v = np.multiply.reduce(v)
            elif i == 'shape' and Image.isImageType(handle):
                v = getattr(handle, 'size', None)
            elif i == 'pages' and v is not None and isinstance(v, (list, tuple)):
                v = len(v)

            if warn and hasattr(self, o):
                c = getattr(self, o)
                if None not in (c, v) and c != v:
                    warnings.warn(f'Encountered different {i}! Expected {c} but got {v}.')
            setattr(self, o, v)
            self._debug("    update", i, o, v, "result:", getattr(self, o))

        # Special treatment
        if self.inferred_ndim is None:
            self.inferred_ndim = len(self.shape)

    def _load_h5_dataset(
            self,
            item=None,
            filename=None,
            dataset=None,
            keep=None,
            explore=False,
            fractal=False
    ):
        """

        Notes:
            "The entire API is now believed to be thread-safe." (http://docs.h5py.org/en/stable/whatsnew/2.4.html)

        Args:
            item:
            filename:
            dataset:
            keep:
            explore:
            fractal:

        Returns:

        """
        filename = self.filename if filename is None else filename
        dataset = self.dataset if dataset is None else dataset
        keep = self._keep if keep is None else keep
        res = None

        if keep:
            try:
                h5 = self.file_handles[filename]
            except KeyError:
                self.file_handles[filename] = h5 = H5(filename, mode='r')
            t = h5[dataset]
            if explore:
                self._update_meta(t)
            else:
                res = t[:] if item is None else t[item]
        else:
            with H5(filename, mode='r') as h5:
                t = h5[dataset]
                if explore:
                    self._update_meta(t)
                else:
                    res = t[:] if item is None else t[item]

        if not fractal and not explore:
            res = self._apply_callbacks(res)

            if self._all_item(item):
                self._set_data(res)
        return res

    def _load_h5_datasets(
            self,
            item=None,
            explore=False
    ):
        return self._multi_loader(
            loader_callback=lambda item, name: self._load_h5_dataset(item=item, dataset=name,
                                                                     filename=self.filename,
                                                                     keep=True, explore=explore, fractal=True),
            name_list=self.datasets,
            item=item,
            explore=explore
        )

    def _multi_loader(
            self,
            loader_callback: Callable,
            name_list: List[str],
            item,
            explore=False
    ):
        """Helper function for multiple read operations.

        Supports read and reduce operations with multiple files, optionally with tiling.

        Notes:
            Tiling is not supported if `item` is set.

        Args:
            loader_callback: Callable with two arguments (item, name)
            name_list: List of names (filenames for tifs, dataset names for hdf5)
            item: Item specifier
            explore:

        Returns:

        """
        # Without tiling
        if self.tile_size is None or explore or len(
                self.op_chain) == 0 or item is not None:
            res = []
            for name in self.filenames:
                res.append(loader_callback(item=item, name=name))
            if explore:
                shape = list(self.shape)
                self._set_shape(
                    tuple(shape[:self.__spatial_dims] + [len(self.filenames)] + shape[self.__spatial_dims:]))
            else:
                stack = np.stack(res, axis=self.__spatial_dims)
                res = self._apply_callbacks(stack)

        # With tiling
        else:
            res = None
            tiles = Tiling(shape=self.shape, tile_height=self.tile_size, tile_width=self.tile_size, padding=0)
            for tile in tiles:
                stack = []
                for name in name_list:
                    stack.append(loader_callback(item=tile.slices_pad, name=name))
                stack = final_stack = np.stack(stack, axis=self.__spatial_dims)
                if len(self.op_chain) > 0:
                    final_stack = self._apply_callbacks(stack)
                    del stack
                if res is None:  # use non-spatial dimensions from actual result
                    res = np.empty(shape=self.shape[:self.__spatial_dims] + final_stack.shape[self.__spatial_dims:],
                                   dtype=self.dtype)
                res[tile.slices] = final_stack[tile.slices_extract]

        self.purge_file_handles()
        return res

    def close_file_handle(self, key):
        """Close buffered file handle.

        Ties to close file handle.

        Args:
            key:

        Returns:

        """
        try:
            val = self.file_handles[key]
            if hasattr(val, 'close'):
                val.close()
            elif hasattr(val, '__exit__'):
                val.__exit__()
        except KeyError:
            pass

    def purge_file_handles(self):
        """Close all cached file handles.

        Tries to call __exit__ or close method of each cached file handle.
        """
        for key in self.file_handles.keys():
            self.close_file_handle(key)
        self.file_handles = {}

    def to_h5(
            self,
            filename: str,
            dataset: str,
            **kwargs
    ):
        """Write data to hdf5 dataset.

        Args:
            filename: Name of hdf5 file.
            dataset: Name of hdf5 dataset.
            **kwargs: Keyword arguments for h5py's `create_dataset`

        References:
            - http://docs.h5py.org/en/stable/high/dataset.html#dataset-create

        """
        self.close_file_handle(filename)
        with H5(filename, mode='a') as h5:  # TODO: Add write params
            if dataset in h5:
                del h5[dataset]
            h5.create_dataset(name=dataset, data=self[:], **kwargs)

    def to_tif(
            self,
            filename: str,
            method='tile',
            tile_height: int = 240,
            tile_width: int = None,
            min_is_black: bool = True,
            compression: int = 1,
            planar_config: int = 1
    ):
        """Write data to tif file.

        Args:
            filename: Name of tif file.
            method: One of ('tile', 'scanline'). Determines which method is used for writing. 'scanline' is recommended
                for compatibility with common image viewers, 'tile' is recommended for large images.
            tile_height: Tile length. Only relevant for tile method.
            tile_width: Tile width. Only relevant for tile method.
            min_is_black: Whether minimum value represents black color. If False minimum values represent white color.
            compression: Compression level. Value 1 for no compression.
            planar_config: Defaults to 1, component values for each pixel are stored contiguously.
                2 says components are stored in component planes. Irrelevant for greyscale images.

        """
        if tile_width is None:
            tile_width = tile_height
        with Tiff(filename, 'w') as handle:
            handle.write(self[:], method=method, tile_width=tile_width, tile_length=tile_height,
                         photometric=int(min_is_black), compression=compression, planar_config=planar_config)

    def to_tifs(
            self,
            filename: str,
            method='tile',
            tile_height: int = 240,
            tile_width: int = None,
            min_is_black: bool = True,
            compression: int = 1,
            planar_config: int = 1
    ):
        if filename.endswith('.tif'):
            stub = filename[:-len('.tif')]
            ending = '.tif'
        elif filename.endswith('.tiff'):
            stub = filename[:-len('.tiff')]
            ending = '.tiff'
        else:
            stub = filename
            ending = '.tif'

        if tile_width is None:
            tile_width = tile_height
        dims = self.guess_spatial_dimensions()
        rs = self.shape[2:2 + dims]
        nums = [min(2, int(np.ceil(np.log10(i)))) for i in rs]
        for i in range(int(np.prod(rs))):
            indices = np.unravel_index(i)
            sel = (slice(None), slice(None)) + tuple((slice(j) for j in indices))
            suffix = '_'.join([f'%0{n}d' % index for n, index in zip(nums, indices)])
            with Tiff(stub + '_Slice' + suffix + ending, 'w') as handle:
                handle.write(self.lazy_load(sel), method=method, tile_width=tile_width, tile_length=tile_height,
                             photometric=int(min_is_black), compression=compression, planar_config=planar_config)

    def to_pyr_tif(
            self,
            filename,
            method='tile',
            tile_height=256,
            tile_width=None,
            min_is_black: bool = True,
            compression: int = 1,
            planar_config: int = 1,
            max_pages=9,
            min_size: int = 42,
            dtype=np.uint8
    ):
        """Write data to pyramid tif.

        Writes paged tif file. Page 0 is largest, each page reduces image size by half.
        This can be viewed as the reversed DZI standard.

        References:
            http://schemas.microsoft.com/deepzoom/2008

        Args:
            filename: Name of tif file.
            method: One of ('tile', 'scanline'). Determines which method is used for writing. 'scanline' is recommended
                for compatibility with common image viewers, 'tile' is recommended for large images.
            tile_height: Tile length. Only relevant for tile method.
            tile_width: Tile width. Only relevant for tile method.
            min_is_black: Whether minimum value represents black color. If False minimum values represent white color.
            compression: Compression level. Value 1 for no compression.
            planar_config: Defaults to 1, component values for each pixel are stored contiguously.
                2 says components are stored in component planes. Irrelevant for greyscale images.
            max_pages: Maximum number of pages. If image size is less than `min_size` at any dimension no further pages
                are generated.
            min_size: Minimal allowed spatial image dimension size.
            dtype: Data type

        Returns:
            None
        """
        from cv2 import pyrDown

        if tile_width is None:
            tile_width = tile_height
        inputs = self[:]
        with Tiff(filename, 'w', bigtiff=True) as o:
            for page in range(max_pages):
                if inputs.dtype != dtype:
                    inputs = inputs.astype(dtype)
                    gc.collect()
                o.write(inputs, method=method, tile_width=tile_width, tile_length=tile_height,
                        photometric=int(min_is_black), compression=compression, planar_config=planar_config)
                if np.any((np.array(inputs.shape[:2]) * .5) < min_size):
                    break
                if page + 1 >= max_pages:
                    inputs = None
                else:
                    inputs = pyrDown(inputs)
                gc.collect()

    def to_image(
            self,
            filename: str,
            **kwargs
    ):
        """Write data to image file.

        Args:
            filename: Name of image file.
            kwargs: Additional arguments for imageio's `imwrite` function.

        """
        imageio.imwrite(filename, self[:], **kwargs)

    def _set_data(self, v: np.ndarray):
        if v.dtype != super().dtype and not self.altered_type:
            self.totype(v.dtype)
        self.resize(v.shape, refcheck=False)
        self[:] = v

    @staticmethod
    def tiling(
            obj,
            tile_height: int,
            tile_width: int = None,
            padding=0,
            worker_id=None,
            workers=None
    ):
        if tile_width is None:
            tile_width = tile_height
        if isinstance(obj, (tuple, list)):
            sizes = [probe_size(o, spatial_dims=2) for o in obj]
            shape = probe_shape(obj)
            if len(set(sizes)) != 1:
                raise ValueError(f'All Tensor objects are expected to have the same shape, but found: {sizes}')
        else:
            shape = obj.shape
        return TensorTiling(obj, shape, tile_height, tile_width, padding=padding, worker_id=worker_id,
                            workers=workers)

    def is_volume(self):
        if self._is_volume is None:
            return self.guess_spatial_dimensions() == 3
        return self._is_volume

    def guess_spatial_dimensions(self):
        if None in (self.shape, self.ndim):
            return None

        sd = self.ndim
        if self.shape[-1] in (1, 3, 4):
            sd -= 1
        return sd


# Add color conversion methods to class for convenience
for c in color_conversions.keys():
    setattr(Tensor, c, lambda self, x=c: self.convert_color(x))

TR = ['__abs__', '__add__', '__and__', '__bool__', '__complex__', '__contains__', '__delitem__', '__dir__',
      '__divmod__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__gt__', '__iadd__',
      '__iand__', '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__', '__imul__', '__index__',
      '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__',
      '__ixor__', '__le__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__',
      '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__rfloordiv__', '__rlshift__',
      '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
      '__rtruediv__', '__rxor__', '__sub__', '__truediv__', '__xor__', 'all', 'any', 'argmax', 'argmin',
      'argpartition', 'argsort', 'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate', 'cumprod',
      'cumsum', 'diagonal', 'dot', 'dump', 'dumps', 'fill', 'flatten', 'getfield', 'item', 'itemset', 'max',
      'mean', 'min', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'repeat',
      'reshape', 'round', 'searchsorted', 'setfield', 'setflags', 'sort', 'squeeze', 'std', 'sum',
      'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view']

for tr in TR:
    def interim(self: Tensor, *a, t=tr, **kw):
        # print("INTERIM", t)
        # print("       ", a, t, kw)
        self._try_load_trigger()
        # o = getattr(super(Tensor, self), t)(*a, **kw)
        # print("       ", type(o))
        # if isinstance(o, Tensor):
        #     print("       ", o.array_mode())
        # return o
        return getattr(super(Tensor, self), t)(*a, **kw)


    setattr(Tensor, tr, interim)


def tensor(p_object, dtype=None):
    return Tensor(p_object, dtype=dtype)
