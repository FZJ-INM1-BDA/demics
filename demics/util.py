import _io
import numpy as np
from collections.abc import Iterable


def pack_list(v, indicator=True):
    d = isinstance(v, list)
    if not d:
        if isinstance(v, tuple) and all((isinstance(v_, list) for v_ in v)):
            v = list(zip(*v))
        else:
            v = [v]
    if indicator:
        return v, d
    return v


def unpack_list(v, indicator=False):
    if indicator or len(v) != 1:
        return v
    return v[0]


def probe_shape(inputs):
    if isinstance(inputs, (list, tuple)):
        return probe_size(inputs[0])
    elif hasattr(inputs, 'shape'):
        return inputs.shape
    else:
        raise ValueError('Did not understood structure of inputs.')


def probe_size(inputs, spatial_dims=2):
    return probe_shape(inputs)[:spatial_dims]


class Printable:
    """Standalone class that makes child classes printable.

    Examples:
        >>> class A(Printable):
        >>>     def __init__(self):
        >>>         self.name = 'Name'
        >>>         self.value = 42
        >>>         self.dictionary = {'key0': 4, 'key1': 2}
        >>>         self.array = np.random.randint(0, 42, (2, 2))
        >>>         self.mylist = list(range(5))
        >>>         self.myset = set(range(50))
        >>> a = A()
        >>> a
        A(
          (name): Name,
          (value): 42,
          (dictionary): dict({'key0': 4, 'key1': 2}),
          (array): ndarray(
            [[17 39]
             [ 5 36]]
          ),
          (mylist): list([0, 1, 2, 3, 4]),
          (myset): set(
            (0): 0,
            (1): 1,
            ...,
            (49): 49,
          ),
        )

    """

    @staticmethod
    def p(inputs, indent=0, n=2, always_type=False):
        if isinstance(inputs, (int, float, str)) and not always_type:
            return str(inputs)

        kw = {'always_type': always_type}
        string = f'{type(inputs).__name__}('
        ni = indent + 1
        ind = " " * indent * n
        nind = " " * ni * n
        if isinstance(inputs, np.ndarray):
            string += '\n'
            s = str(inputs)
            s = nind + s.replace('\n', f'\n{nind}')
            string += s
            string += f'\n{" " * indent * n})'

        elif hasattr(inputs, '__dict__'):
            string += '\n'
            for key, val in inputs.__dict__.items():
                if key.startswith('_'):
                    continue
                s = Printable.p(val, **kw).replace('\n', f'\n{nind}')
                string += f'{nind}({key}): {s},\n'
            string += f'{" " * indent * n})'

        elif isinstance(inputs, dict):
            string += '\n'
            le = len(inputs)
            if le < 150:
                for key, val in inputs.items():
                    s = Printable.p(val, **kw).replace('\n', f'\n{nind}')
                    string += f'{nind}({key}): {s},\n'
            else:
                items = iter(inputs.items())
                for i in range(min(2, le)):
                    key, v = next(items)
                    s = Printable.p(v, **kw).replace('\n', f'\n{nind}')
                    string += f'{nind}({key}): {s},\n'
                if le > 3:
                    s = '...'
                    string += f'{nind}{s},\n'
                if le > 2:
                    *_, (key, v) = iter(items)
                    s = Printable.p(v, **kw).replace('\n', f'\n{nind}')
                    string += f'{nind}({key}): {s},\n'
                string += f'{nind})'
            string += f'{" " * indent * n})'

        elif isinstance(inputs, Iterable) and hasattr(inputs, '__getitem__') and isinstance(inputs, str) is False:
            s = str(inputs)
            if len(s) <= 64:
                if isinstance(inputs, (tuple, list)):
                    string = s
                else:
                    string += s + ')'
            else:
                string += '\n'
                le = len(inputs)
                oversize = le > 64
                for i in range(min(2, le) if oversize else le):
                    s = Printable.p(inputs[i], **kw).replace('\n', f'\n{nind}')
                    string += f'{nind}({i}): {s},\n'
                if oversize:
                    if le > 3:
                        s = '...'
                        string += f'{nind}{s},\n'
                    if le > 2:
                        s = Printable.p(inputs[le - 1], **kw).replace('\n', f'\n{nind}')
                        string += f'{nind}({le - 1}): {s},\n'
                string += f'{ind})'
        else:
            s = str(inputs)
            if len(s) > 64:
                s = s[:24] + '...'

            string += s + ')'
        return string

    def __str__(self):
        return self.p(self)

    def __repr__(self):
        return self.__str__()


class Bytes(int):
    def __init__(self, b: int):
        super().__init__()
        self.bytes = b

    def __str__(self):
        n = np.log2(int(self)) if self > 0 else 0
        plan = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB', 'BB']
        s = None
        for i, tag in enumerate(plan):
            if n < (i + 1) * 10 or i == len(plan) - 1:
                s = str(np.round(float(self) / (2 ** (10 * i)), 2)) + ' ' + tag
                break
        return s

    def __repr__(self):
        return str(self)


class Mock:
    def __getattr__(self, item):
        return self

    def __len__(self):
        return 0

    def __int__(self):
        return 0

    def __bool__(self):
        return False

    def __float__(self):
        return 0.


for func_name in {'__call__', '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__',
                  '__lt__', '__gt__', '__le__', '__ge__', '__eq__', '__ne__', '__isub__', '__iadd__', '__imul__',
                  '__idiv__', '__ifloordiv__', '__imod__', '__ipow__', '__neg__', '__pos__', '__invert__'}:
    setattr(Mock, func_name, lambda self, *args, **kwargs: self)


class StdOut:
    def __init__(self, context: _io.TextIOWrapper, prefix):
        self.context = context
        self.prefix = prefix
        self.initial = True

    def write(self, string: str):
        if len(string.strip()) > 0:
            if self.initial:
                string = self.prefix + string
                self.initial = False
            string = string.replace('\n', '\n' + ' ' * len(self.prefix))
        if string.strip(' ').endswith('\n'):
            self.initial = True
        self.context.write(string)

    def flush(self):
        self.context.flush()
