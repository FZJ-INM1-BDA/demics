from numba import cuda
import psutil
from ..util import Printable, Bytes


class Device(Printable):
    def __init__(
            self,
            dev: cuda.cudadrv.devices._DeviceContextManager
    ):
        self.name: str = dev.name.decode()
        self.id: int = dev.id
        self.compute_capability = dev.compute_capability


class System(Printable):
    def __init__(self):

        # GPU
        devices: cuda.cudadrv.devices._DeviceList = cuda.list_devices()
        try:
            self.num_devices = len(devices)
            self.devices = [Device(d) for d in devices]
        except cuda.CudaSupportError:
            self.num_devices = 0

        # CPU
        self.logical_cpus = psutil.cpu_count(logical=True)
        self.physical_cpus = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq(percpu=False)
        self.min_cpu_freq = freq.min
        self.max_cpu_freq = freq.max

        # Memory
        self.virtual_memory = Bytes(psutil.virtual_memory().total)
        self.swap_memory = Bytes(psutil.swap_memory().total)
