import re
import os
from h5py import File as H5
from typing import Union, List, Tuple, Callable


class Resolver:
    """Stateful class for matching of names, patterns and templates of files and hdf5 datasets."""

    def __init__(self, check_existence=True):
        self._filenames = None
        self._datasets = None
        self._single_file = None
        self._single_ds = None
        self.check_existence = check_existence

    def single_file(self):
        return self._single_file

    def single_dataset(self):
        return self._single_ds

    def _all_ext(self, extensions: list):
        if self._filenames is None:
            return None
        return all([any(f.endswith(ex) for ex in extensions) for f in self._filenames])

    def all_h5(self):
        return self._all_ext(['h5', 'hdf5'])

    def all_tif(self):
        return self._all_ext(['tif', 'tiff'])

    def all_png(self):
        return self._all_ext(['png'])

    def all_jpg(self):
        return self._all_ext(['jpg', 'jpeg'])

    @staticmethod
    def _list_match(string_list, pattern):
        prog, matched = re.compile(pattern), []
        for f in string_list:
            if prog.match(f) is not None:
                matched.append(f)
        return matched

    @staticmethod
    def _parse_value_string(values):
        if isinstance(values, str):
            res = []
            for v in values.split(','):
                v = v.strip()
                # Range
                if '-' in v:
                    ab = v.split('-')
                    assert len(ab) == 2
                    a, b = ab
                    res += list(range(int(a), int(b) + 1))
                # Index
                else:
                    res.append(int(v))
        else:
            res = list(values)
            assert all([isinstance(v, int) for v in values])
        return list(set(res))

    @staticmethod
    def _repl_match(template, values, prefix='_Slice', numbers=2):
        values = Resolver._parse_value_string(values)
        ns = str(numbers)
        return [re.sub(prefix + '[0-9]{' + ns + '}', (prefix + '%0' + ns + 'd') % v, template) for v in values]

    @staticmethod
    def _resolve_file_pattern(pattern):
        dirname, pattern = os.path.dirname(pattern), os.path.basename(pattern)
        if len(dirname) == 0:
            dirname = '.'
        return [os.path.join(dirname, f) for f in Resolver._list_match(os.listdir(dirname), pattern)]

    def _resolve_h5_pattern(self, pattern):
        res = []
        for filename in self._filenames:
            with H5(filename, 'r') as h5:
                keys = list(h5.keys())
            res.append(Resolver._list_match(keys, pattern))
        return res

    @staticmethod
    def _get(sources: list, single):
        if sources is None:
            res = None
        elif single is True or (single is not False and len(sources) == 1):
            res = sources[0]
        else:
            res = sources
        return res

    @staticmethod
    def _split_file_dataset(src: str, delimiter: str):
        """Split strings in two substrings on first occurrence of delimiter.

        Args:
            src: Source string.
            delimiter: Delimiter string.

        Returns:
            Tuple of two strings.
        """
        s = src.split(delimiter)
        prefix = s[0]
        postfix = src[len(prefix) + len(delimiter):]
        return prefix, postfix

    def get_filenames(self):
        return self._get(self._filenames, self._single_file)

    def get_datasets(self):
        res = []
        for d in self._datasets:
            res.append(self._get(d, self._single_ds))
        return self._get(res, self._single_file)

    def _resolve(
            self,
            src: str,
            callback: Callable,
            n: int = -1,
            single: bool = None,
            id_range: Union[List[int], Tuple[int], str] = None,
            prefix: str = '_Slice',
            numbers: int = 2,
            sort: bool = True
    ):
        # Pattern matching or single file
        if id_range is None:
            # Single file
            if single is True:
                filenames = [src]
            # Implicit single file or volume
            else:
                filenames = callback(src)
        # Templating
        else:
            filenames = self._repl_match(template=src, values=id_range, prefix=prefix, numbers=numbers)

        num = len(filenames)
        if num == 0:
            raise FileNotFoundError(f'No file found for src={str(src)}.')

        if 0 < n != num:
            raise ValueError(f'Found {num} files, but asserted n={n}.')

        if single is True and num > 1:
            raise ValueError(f'The parameter single was set to {single}, but found {num} source files. '
                             f'Use single to indicate, that a single source is asserted to exist and returned.')
        if sort and isinstance(filenames, list):
            filenames.sort()
        return filenames

    def files(
            self,
            src: str,
            n: int = -1,
            single: bool = None,
            id_range: Union[List[int], Tuple[int], str] = None,
            prefix: str = '_Slice',
            numbers: int = 2
    ):
        self._filenames = self._resolve(
            src=src,
            callback=self._resolve_file_pattern,
            n=n,
            single=single,
            id_range=id_range,
            prefix=prefix,
            numbers=numbers,
        )
        self._single_file = single or (single is not False and len(self._filenames) == 1)

        if self.check_existence:
            for f in self._filenames:
                if not os.path.isfile(f):
                    raise FileNotFoundError(f'File does not exist: {f}.')

        return self.get_filenames()

    def dataset(
            self,
            src: str,
            n: int = -1,
            single: bool = None,
            id_range: Union[List[int], Tuple[int], str] = None,
            prefix: str = '_Slice',
            numbers: int = 2
    ):
        if self._filenames is None or len(self._filenames) == 0:
            raise ValueError('File names need to be resolved first.')

        self._datasets = self._resolve(
            src=src,
            callback=self._resolve_h5_pattern,
            n=n,
            single=single,
            id_range=id_range,
            prefix=prefix,
            numbers=numbers,
        )
        self._single_ds = single or (single is not False and len(self._datasets) == 1)

        if self.check_existence:
            for f, ds in zip(self._filenames, self._datasets):
                with H5(f, 'r') as h5:
                    for h in ds:
                        if h not in h5:
                            raise FileNotFoundError(f'The file {f} does not contain: {h}.')

        return self.get_datasets()

    def mixed(
            self,
            src: str,
            n: int = -1,
            single: bool = None,
            id_range: Union[List[int], Tuple[int], str] = None,
            prefix: str = '_Slice',
            numbers: int = 2,
            delimiter: str = ':'
    ):
        """

        Notes:
            It is assumed that file names do not contain the delimiter.

        Args:
            src:
            n:
            single:
            id_range:
            prefix:
            numbers:
            delimiter:

        Returns:

        """
        # Src denotes H5 file with dataset(s)
        if delimiter in src:
            src, ds = self._split_file_dataset(src, delimiter)
            files = self.files(
                src=src,
                n=1,
                single=True,
            )
            if self.all_h5() is False:
                raise ValueError('Expected all files to be hdf5 files.')

            datasets = self.dataset(
                src=ds,
                n=n,
                single=single,
                id_range=id_range,
                prefix=prefix,
                numbers=numbers,
            )

        # Src denotes file(s)
        else:
            files = self.files(
                src=src,
                n=n,
                single=single,
                id_range=id_range,
                prefix=prefix,
                numbers=numbers,
            )
            datasets = None
            if self.all_tif() is False and self.all_jpg() is False and self.all_png() is False:
                raise ValueError('Expected all files to be image files with the same format.')
        return files, datasets
