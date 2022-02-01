from concurrent.futures import thread
import logging
import json
from pyexpat import model
import sys
from typing import Any, Callable, Dict, List, Union, Optional
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .utils import copy_state_dict

from .quantization import AciqQuantization, Quantization, LinearQuantization, LogarithmicQuantization

class OctoTorch:
    def __init__(
        self,
        model: nn.Module,
        steps: List[int] = [32, 16, 8, 6, 4, 2, 1],
        qmethod: Optional[Union[List[Quantization], Quantization]] = [AciqQuantization()],# [LinearQuantization(), LogarithmicQuantization(), AciqQuantization()],
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
        assert qmethod is not None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
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
        self._model.eval()
        self.logger.info("starting quantization")
        model_weigths = copy_state_dict(self._model.state_dict())
        results = {}
        initial_score = self.score_func(self._model)
        for method in tqdm(self.qmethods, leave=False):
            self.logger.debug(f"quantizing with {method.name}")
            keys_to_try = {x: True for x in model_weigths.keys()}
            for bit_width in tqdm(self.bits, leave=False):
                if not method.supports_bit(bit_width):
                    continue
                # Get an initial result for quantizing each layer on its own.
                for key, weight in tqdm(model_weigths.items(), leave=False):
                    if not keys_to_try[key]:
                        continue
                    if self.exclude_layers is not None:
                        if any(map(lambda x: x in key, self.exclude_layers)):
                            continue
                    if self.allow_layers is not None:
                        if not any(map(lambda x: x in key, self.allow_layers)):
                            continue
                    quantized = method.quantize(weight.clone(), bit_width)
                    error = abs(weight - quantized).sum().item()
                    state_dict = copy_state_dict(model_weigths)
                    state_dict[key] = quantized
                    self._model.load_state_dict(state_dict)
                    score = self.score_func(self._model)
                    if "single" not in results:
                        results["single"] = {}
                    if key not in results["single"]:
                        results["single"][key] = []
                    results["single"][key].append({
                        "score": score,
                        "score_error": initial_score - score,
                        "quantized": {
                            "method": method,
                            "method_name": method.name,
                            "size_difference": weight.numel() * 32 - weight.numel() * bit_width,
                            "bitwidth": bit_width,
                            "weight_error": error,
                            "method_options": method.get_setting_dict()
                        }
                    })
                    if score <= 0.85 * initial_score:
                        keys_to_try[key] = False
                    write_json(results)
        # Find the best ones (sorted) and take them greedy
        all_results = {}
        for key, value in results.items():
            get_best(value, all_results, 0.85 * initial_score)
        all_items = list(all_results.items())
        all_items.sort(key=lambda el: el[1]["score"] * el[1]["quantized"]["size_difference"])
        chosen_keys = []
        quantized_state_dict = copy_state_dict(model_weigths)
        for key, value in all_items:
            quantized = method.quantize(quantized_state_dict[key], bit_width)
            # error = abs(weight - quantized).sum().item()
            quantized_state_dict[key] = quantized
            self._model.load_state_dict(quantized_state_dict)
            score = self.score_func(self._model)
            if score <= 0.85 * initial_score:
                return
            chosen_keys.append((key, value))
            if "multi" not in results:
                results["multi"] = []
            results["multi"].append({
                "score": score,
                "score_error": initial_score - score,
                "quantizations": list(map(lambda el: {
                    "key": el[0],
                    "method": el[1]["quantization"]["method"],
                    "bitwith": el[1]["quantization"]["bitwith"],
                    "method_options": el[1]["quantization"]["method_options"],
                }, chosen_keys))
            })
            write_json(results)
        return results

def get_best(key: str, result_list: Dict[str, Dict[str, Any]], out_list: List, score_thresh = 0):
    filtered = list(filter(lambda el: el["score"] >= score_thresh, result_list))
    if len(filtered) == 0:
        return
    filtered.sort(key=lambda el: el["score"] * el["quantized"]["size_difference"], reverse=True)
    out_list[key] = filtered[0]

def write_json(dict: Dict):
    with open("quantization_results.json", "w") as file_handle:
        json.dump(dict, file_handle, indent=4, default=vars)
