from numba import cuda
import psutil
from ..util import Printable, Bytes
import platform
import socket


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
        # System
        plat = platform.uname()
        self.system = plat.system
        self.node = plat.node
        self.ip = self.get_ip()

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

    @staticmethod
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            i = s.getsockname()[0]
        except Exception:
            i = '127.0.0.1'
        finally:
            s.close()
        return i

