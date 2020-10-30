from ..system import System
from ..util import StdOut
import numpy as np
import sys

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = False


class Environment:
    def __init__(self, use_mpi=True, one_file_one_node=False, verbose=False, use_tagging=True):
        self.use_mpi = use_mpi and MPI
        self.one_file_one_node = one_file_one_node
        self.__gpus = None
        self.system = System()

        # Resolve MPI params
        if self.use_mpi:
            self.comm_world = MPI.COMM_WORLD
            self.world_rank = self.comm_world.Get_rank()
            self.world_ranks = self.comm_world.Get_size()
            self.host = MPI.Get_processor_name()
            self.hosts = list(np.sort(np.unique(self.comm_world.allgather(self.host))))
            self.node_ranks = len(list(self.hosts))
            self.node_rank = next(i for i, h in enumerate(self.hosts) if h == self.host)
            self.comm_local = self.comm_world.Split(color=self.node_rank, key=self.world_rank)
            self.local_rank = self.comm_local.Get_rank()
            self.local_ranks = self.comm_local.Get_size()

            self.has_gpu = self.local_rank < self.system.num_devices
            gpu_color = int(self.has_gpu)
            self.comm_local_gpu = self.comm_local.Split(color=gpu_color, key=self.local_rank)
            self.local_gpu_rank = self.comm_local_gpu.Get_rank()
            self.local_gpu_ranks = self.comm_local_gpu.Get_size()
            self.comm_world_gpu = self.comm_world.Split(color=gpu_color, key=self.world_rank)
            self.world_gpu_rank = self.comm_world_gpu.Get_rank()
            self.world_gpu_ranks = self.comm_world_gpu.Get_size()

            if use_tagging and self.world_ranks > 1:
                sys.stdout = StdOut(sys.stdout, prefix=f'(rank: {self.world_rank})  ')
        else:
            self.comm_world = None
            self.world_rank = 0
            self.world_ranks = 1
            self.host = None
            self.hosts = 1
            self.node_ranks = 1
            self.node_rank = 0
            self.comm_local = None
            self.local_rank = 0
            self.local_ranks = 1
            self.has_gpu = self.local_rank < self.system.num_devices
            self.comm_world_gpu = None
            self.world_gpu_rank = 0
            self.world_gpu_ranks = 1

        if self.one_file_one_node:
            self.comm = self.comm_local
            self.rank = self.local_rank
            self.ranks = self.local_ranks
            self.comm_gpu = self.comm_local_gpu  # use along with self.has_gpu, as this is the result of a Split
            self.rank_gpu = self.local_gpu_rank
            self.ranks_gpu = self.local_gpu_ranks
        else:
            self.comm = self.comm_world
            self.rank = self.world_rank
            self.ranks = self.world_ranks
            self.comm_gpu = self.comm_world_gpu
            self.rank_gpu = self.world_gpu_rank
            self.ranks_gpu = self.world_gpu_ranks

        if verbose:
            w = 13
            print('\n'.join(['MPI_SETUP('] + [f'    {(k + " " * (w - len(k)))}: {v}' for k, v in {
                'host': self.host,
                'world_rank': self.world_rank,
                'world_ranks': self.world_ranks,
                'comm_world': self.comm_world,
                'local_rank': self.local_rank,
                'local_ranks': self.local_ranks,
                'comm_local': self.comm_local,
                'rank': self.rank,
                'ranks': self.ranks,
                'comm': self.comm,
                'has_gpu': self.has_gpu,
                'rank_gpu': self.rank_gpu,
                'ranks_gpu': self.ranks_gpu,
                'comm_gpu': self.comm_gpu,
                'node_rank': self.node_rank,
                'node_ranks': self.node_ranks,
            }.items()] + [')']))
            sys.stdout.flush()

    @staticmethod
    def scatter(comm, data, root, rank, ranks):
        """Scatter with single message passing. Use when comm.scatter msg size is exceeded."""
        if rank == root:
            r = list(range(ranks))
            del r[root]
            for i in r:
                comm.send(data[i], dest=i, tag=0)
            data = data[root]
        else:
            data = comm.recv(source=root, tag=0)
        return data

    @staticmethod
    def gather(comm, data, root, rank, ranks, empty=()):
        """Gather with single message passing. Use when comm.gather msg size is exceeded."""
        if rank == root:
            result = ranks * [None]
            result[root] = data
            for i in range(ranks - 1):
                status = MPI.Status()
                data_recv = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
                result[status.Get_source()] = data_recv
            return result
        else:
            comm.send(data, dest=root, tag=0)
        return empty
