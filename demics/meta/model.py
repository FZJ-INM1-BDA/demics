import warnings
import json
import os


class Model:
    def __init__(
            self,
            directory: str
    ):
        # Defaults
        self.preprocessing = []
        self.padding = 0
        self.inputs_str = None
        self.outputs_str = None
        self.framework = None
        self.version = None

        self.directory = directory
        assert os.path.isdir(self.directory)
        self.meta_filename = os.path.join(self.directory, 'meta.json')

        if not os.path.isfile(self.meta_filename):
            warnings.warn(f'Could not find meta file for model {self.meta_filename}.')
        else:
            with open(self.meta_filename, 'r') as fp:
                self.meta = json.load(fp)
            self.parse(self.meta)

    def parse(self, json_dict: dict):
        try:
            self.framework = json_dict['framework']
            self.version = json_dict['version']
            self.preprocessing = json_dict['preprocessing']
            self.padding = json_dict['padding']
            self.inputs_str = json_dict['inputs_str']
            self.outputs_str = json_dict['outputs_str']
        except KeyError:
            raise ValueError(f'Error during parsing of model meta file "{self.meta_filename}". There are possibly '
                             f'missing items.')

    def is_tf(self):
        return self.framework.lower() == "tensorflow"

    def is_tf1(self):
        return self.is_tf() and self.version.startswith('1.')

    def is_tf2(self):
        return self.is_tf() and self.version.startswith('2.')

    def is_pytorch(self):
        return self.framework.lower() == "pytorch"
