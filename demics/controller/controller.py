from ..tensor import Tensor, TensorTile
from ..util import pack_list, unpack_list, probe_size, Mock
from ..environment import Environment
from typing import List, Callable, Union, Tuple
import numpy as np
import sys


class Controller:
    EMPTY = Mock()
    _instance = None

    def __init__(
            self,
            verbose=False
    ):
        super().__init__()
        self.verbose = verbose
        if Controller._instance is None:
            Controller._instance = Environment(verbose=self.verbose)
        self.env = Controller._instance
        self.writer_rank = 0

    def log(self, *s):
        if self.verbose:
            print(*s, flush=True)

    @staticmethod
    def _handle_outputs(outputs, tile: TensorTile, size, aggregate=None):
        data = tile.results()
        if outputs is None:
            if isinstance(data, np.ndarray):
                outputs = Tensor.zeros(shape=size + data.shape[2:], dtype=data.dtype)
        if aggregate is None:
            non_value = None
            with_overlap = False
        else:
            non_value = aggregate.non_value
            with_overlap = aggregate.return_overlap
        outputs = tile.handover(outputs, aggregate, non_value=non_value, with_overlap=with_overlap)
        return outputs

    def _step(self, payload, root, callback, return_overlap, gpu, args=None, kwargs=None):
        comm = self.env.comm_gpu if gpu else self.env.comm
        rank = self.env.rank_gpu if gpu else self.env.rank
        ranks = self.env.ranks_gpu if gpu else self.env.ranks
        args, kwargs = [] if args is None else args, {} if kwargs is None else kwargs

        if comm is None:
            assert len(payload) == 1
            tile = payload[0]
        else:
            if gpu is False or self.env.has_gpu:
                tile = self.env.scatter(comm, payload, root, rank, ranks)
                # tile = comm.scatter(payload, root=root)
            else:
                tile = None  # idle
        if tile is ...:
            return tile
        if tile is not None:
            current_inputs = tile.inputs(copy=root == rank)
            self.log(">>", [(c.shape, c.dtype) for c in current_inputs])
            result = callback(*current_inputs, *args, **kwargs, meta={
                'xrange': tile.b_range,
                'yrange': tile.a_range,
                'xrange_padded': tile.b_range_pad,
                'yrange_padded': tile.a_range_pad,
                'xrange_ex': tile.b_range_extract,
                'yrange_ex': tile.a_range_extract,
            })
            self.log("<<", str(type(result)) + ' ' + str(
                         (result.shape, result.dtype) if isinstance(result, np.ndarray) else ''))
            tile.set_results(result, include_overlap=return_overlap)
        if comm is None:
            return [tile]
        else:
            return self.env.gather(comm, tile, root, rank, ranks, empty=self.EMPTY)

    def work(self, i_objects, overlap, callback, gpu, args=None, kwargs=None, aio=False, aggregate=None,
             tile_size=None):
        """

        Args:
            i_objects:
            overlap:
            callback:
            aio: If True force single worker to perform operation in a single step.
            aggregate:

        Returns:

        """
        comm = self.env.comm_gpu if gpu else self.env.comm
        rank = self.env.rank_gpu if gpu else self.env.rank
        root = self.writer_rank
        if self.env.one_file_one_node:
            i_objects = i_objects[self.env.node_rank::self.env.node_ranks]
        tile_height = None if aio else tile_size
        master = self.env.rank == root
        ranks = 1 if aio else (self.env.ranks_gpu if gpu else self.env.ranks)
        return_overlap = False if aggregate is None else aggregate.return_overlap

        if hasattr(callback, 'start') and callable(callback.start):
            callback.start()

        if master:
            final_outputs = []
            if gpu and self.env.has_gpu is False:
                raise ValueError('GPU is required, but no GPU available.')
            for i, inputs in enumerate(i_objects):
                outputs = None
                cur_aggregate = None if aggregate is None else aggregate()
                size = probe_size(inputs)
                tiling = Tensor.tiling(inputs, tile_height=tile_height, padding=overlap)
                while True:
                    payload = []
                    try:
                        for _ in range(ranks):
                            payload.append(next(tiling))
                            payload[-1].pre_transport()
                    except StopIteration:
                        if len(payload) == 0:
                            break
                    payload += [None] * (ranks - len(payload))
                    step = self._step(payload, root=self.writer_rank, callback=callback, return_overlap=return_overlap,
                                      gpu=gpu, args=args, kwargs=kwargs)
                    for p in payload:
                        if isinstance(p, TensorTile):
                            p.post_transport()
                    for tile in step:
                        assert tile is None or isinstance(tile, TensorTile)
                        if tile is not None:
                            outputs = self._handle_outputs(outputs, tile, size, cur_aggregate)
                final_outputs.append(outputs)
            if comm is not None:
                self.log("sending signal to finish")
                self.env.scatter(comm, [...] * ranks, root, rank, ranks)
        else:
            while True:
                step = self._step(self.EMPTY, root=self.writer_rank, callback=callback, return_overlap=return_overlap,
                                  gpu=gpu, args=args, kwargs=kwargs)
                if step is ...:
                    self.log("finishing")
                    break
            final_outputs = self.EMPTY
        if hasattr(callback, 'stop') and callable(callback.stop):
            callback.stop()
        return final_outputs

    def __call__(
            self,
            callback: Callable,
            i_objects: Union[Tensor, Tuple[Tensor], List[Tensor], List[Tuple[Tensor]]],
            overlap: int,
            gpu: bool = False,
            tile_size=None,
            aggregate=None,
            args=None,
            kwargs=None,
    ):
        """

        Args:
            callback:
            i_objects: Input objects. Can be single input object, list of input objects,
                a tuple of input objects for a single operation with multiple inputs or a list of
                tuples of input objects.
            overlap:
            *args:
            **kwargs:

        Returns:

        """

        if gpu and self.env.has_gpu is False:
            return self.EMPTY

        i_objects, stat = pack_list(i_objects)
        self.i_objects = i_objects

        if len(self.i_objects) == 0 and self.env.rank == self.writer_rank:
            raise ValueError(f'Missing input. (Rank {self.env.rank})')

        if self.env.one_file_one_node:
            self.i_objects = self.i_objects[self.env.node_rank::self.env.node_ranks]

        self.padding = overlap
        res = self.work(
            i_objects=i_objects,
            overlap=overlap,
            callback=callback,
            aggregate=aggregate,
            gpu=gpu,
            args=args,
            kwargs=kwargs,
            tile_size=tile_size
        )
        res = unpack_list(res, stat)
        return res
