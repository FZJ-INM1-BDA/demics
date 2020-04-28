from abc import abstractmethod
import numpy as np


class Aggregate:
    return_overlap = False

    def __init__(self, initial_state, non_value=None):
        self.state = initial_state
        self.non_value = non_value

    @abstractmethod
    def feed(self, inputs: np.ndarray) -> np.ndarray:
        pass


class LabelAggregate(Aggregate):
    return_overlap = True

    def __init__(self, initial_state=0, background=0):
        super().__init__(initial_state, non_value=background)
        self.background = background

    def feed(self, inputs: np.ndarray):
        inputs[inputs > self.background] += self.state  # alter instance
        self.state = np.max(inputs)
        return inputs


class LabelVisAggregate(LabelAggregate):
    back_color = [0, 0, 0]

    def __init__(self):
        super().__init__(initial_state=0, background=LabelVisAggregate.back_color)

    def feed(self, inputs: np.ndarray):
        return inputs
