from flwr.common import NDArrays

import numpy as np

from abc import ABC, abstractmethod


class ModelUtils(ABC):
    @abstractmethod
    def get_model_parameters(self, model) -> NDArrays:
        pass

    @abstractmethod
    def set_model_params(self, model, params: NDArrays):
        pass

    @abstractmethod
    def set_initial_params(self, model, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def evaluate(self, model, parameters: NDArrays, config, x_test: np.ndarray, y_test: np.ndarray):
        pass
