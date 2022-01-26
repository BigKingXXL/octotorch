import logging
import json
from typing import Callable, List, Union, Optional
import numpy as np
import torch.nn as nn
from .utils import copy_state_dict

from quantization import Quantization
from quantization.linear_quantization import LinearQuantization

class OctoTorch:
    def __init__(
        self,
        model: nn.Module,
        steps: List[int] = [32, 16, 8, 6, 4, 2],
        qmethod: Optional[Union[List[Quantization], Quantization]] = LinearQuantization,
        score_func: Optional[Callable[[nn.Module], Union[int, float, np.number]]] = None,
        allow_layers: List[str] = None,
        exclude_layers: Optional[List[str]] = None
    ):
        """Initialize the OctoTorch module.

        Args:
            model (nn.Module): Model to be quantized.
            steps (List[int]): Different bit widths for the quantization to test.
            method (Quantization): Quantization method(s) to test. Can be a list of quantiuation methods.
            score_func (Optional[Callable[[nn.Module], Union[int, float, np.number]]], optional): 
                Function to score the model. If not set it will call `model.score()`.
                Expects a numeric value as return value.
                The return value is used to evaluate quantized models.
                Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.allow_layers = allow_layers
        self.exclude_layers = exclude_layers
        self.qmethods = qmethod if isinstance(qmethod, list) else [qmethod]
        self.bits = steps
        self._model = model
        if score_func is not None:
            self.score_func = score_func
        else:
            self.score_func: Callable[[nn.Module], Union[int, float, np.number]] = lambda model: model.score()

    def quantize(self):
        self.logger.info("starting quantization")
        model_weigths = self._model.state_dict()
        results = {}
        for method in self.qmethods:
            self.logger.info(f"quantizing with {method.__class__.__name__}")
            for bit_width in self.bits:
                # Get an initial result for quantizing each layer on its own.
                for key, weight in model_weigths:
                    if self.exclude_layers is not None:
                        if any(map(lambda x: x in key, self.exclude_layers)):
                            continue
                    if self.allow_layers is not None:
                        if not any(map(lambda x: x in key, self.allow_layers)):
                            continue
                    quantized = method.quantize(weight, bit_width)
                    state_dict = copy_state_dict(model_weigths)
                    state_dict[key] = quantized
                    self._model.load_state_dict(state_dict)
                    score = self.score_func(self._model)

                    results["single"][key] = {
                        "score": score,
                        "quantized": {
                            key: {
                                "method": method.__class__.__name__,
                                "bitwidth": bit_width
                            }
                        }
                    }
        with open("quantization_results.json", "w") as file_handle:
            json.dump(results, file_handle)
