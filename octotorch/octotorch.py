from concurrent.futures import thread
import logging
import json
import math
import sys
from typing import Any, Callable, Dict, List, Union, Optional
from cv2 import FlannBasedMatcher
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .utils import copy_state_dict

from .quantization import AciqQuantization, Quantization, LinearQuantization, LogarithmicQuantization, PwlqQuantization

class OctoTorch:
    def __init__(
        self,
        model: nn.Module,
        steps: Optional[List[int]] = [32, 16, 8, 6, 4, 2, 1],
        qmethod: Optional[Union[List[Quantization], Quantization]] = [PwlqQuantization(), AciqQuantization()],# [LinearQuantization(), LogarithmicQuantization(), AciqQuantization()],
        score_func: Optional[Callable[[nn.Module], Union[int, float, np.number]]] = None,
        allow_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
        thresh: Optional[float] = 0.9,
        oreder_weighted: Optional[bool] = True
    ):
        """Initialize the OctoTorch module.

        Args:
            model (`nn.Module`): Model to be quantized.
            steps (`List[int]`, optional, default: `[32, 16, 8, 6, 4, 2, 1]`): Different bit widths for the quantization to test.
            method (`Quantization`): Quantization method(s) to test. Can be a list of quantiuation methods.
            score_func (`Callable[[nn.Module], Union[int, float, np.number]]`, optional, default: `None`): 
                Function to score the model. If not set it will call `model.score()`.
                Expects a numeric value as return value.
                The return value is used to evaluate quantized models.
                Defaults to `None`.
            allow_layers (`List[str]`, optional, default: `None`): List of strings which when not `None` is a whitelist of keys which must
                be included in parameter names to be quantized.
            exclude_layers (`List[str]`, optional, default: `None`): List of strings which when included in parameter name exclude the parameter
                from quantization.
            thresh (`float`, optional, default: `0.9`): Relative threshold of the initial score of the model that must be archived in order
                to keep a quantized layer.
            oreder_weighted (`bool`, optional, default: `False`): Whether to order single layer quantization results using the weight difference
                or not before starting the greedy selection.
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
        self.thresh = thresh
        self.oreder_weighted = oreder_weighted
        if score_func is not None:
            self.score_func = score_func
        else:
            self.score_func: Callable[[nn.Module], Union[int, float, np.number]] = lambda model: model.score()

    def quantize(self):
        device = next(self._model.parameters()).device
        self._model.eval()
        self.logger.info("starting quantization")
        model_weigths = copy_state_dict(self._model.state_dict())
        results = {}
        initial_score = self.score_func(self._model)
        self.logger.info("testing single layer quantization")
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
                    try:
                        quantized = method.quantize(weight.clone(), bit_width)
                    except:
                        self.logger.warning(f"Failed quantizing at: {key}")
                        continue
                    error = abs(weight - quantized).sum().item()
                    state_dict = copy_state_dict(model_weigths)
                    state_dict[key] = quantized
                    self._model.load_state_dict(state_dict)
                    self._model.to(device)
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
                            "size_difference": weight.numel() * 32 - weight.numel() * bit_width,
                            "bitwidth": bit_width,
                            "weight_error": error if error != math.nan else -1.0,
                            "method_options": method.get_setting_dict()
                        }
                    })
                    if score <= self.thresh * initial_score:
                        keys_to_try[key] = False
                    write_json(results)
        # Find the best ones (sorted) and take them greedy
        all_results = {}
        for key in results["single"].keys():
            get_best(key, results["single"], all_results, self.thresh * initial_score, self.oreder_weighted)
        all_items = list(all_results.items())
        all_items.sort(key=lambda el: el[1]["score"] * el[1]["quantized"]["size_difference"])
        chosen_keys = []
        quantized_state_dict = copy_state_dict(model_weigths)
        best_scores = []
        print(all_items)
        self.logger.info("greedy searching for optimal multi layer quantization")
        for key, value in tqdm(all_items, leave=False):
            method = value["quantized"]["method"]
            bit_width = value["quantized"]["bitwidth"]
            quantized = method.quantize(quantized_state_dict[key], bit_width)
            quantized_state_dict[key] = quantized
            self._model.load_state_dict(quantized_state_dict)
            self._model.to(device)
            score = self.score_func(self._model)
            if score <= self.thresh * initial_score:
                break
            best_scores.append(score)
            chosen_keys.append((key, value))
            if "multi" not in results:
                results["multi"] = []
            results["multi"].append({
                "score": score,
                "score_error": initial_score - score,
                "size_difference": sum([x[1]["quantized"]["size_difference"] for x in chosen_keys]),
                "quantizations": list(map(lambda el: {
                    "key": el[0],
                    "method": el[1]["quantized"]["method"],
                    "size_difference": el[1]["quantized"]["size_difference"],
                    "bitwith": el[1]["quantized"]["bitwidth"],
                    "weight_error": el[1]["quantized"]["weight_error"],
                    "method_options": el[1]["quantized"]["method_options"],
                }, chosen_keys))
            })
            write_json(results)
        self.logger.info(f"Final score is: {best_scores[-1]}")
        return results

def get_best(
    key: str,
    result_dict: Dict[str, Dict[str, Any]],
    out_dict: Dict[str, Dict[str, Any]],
    score_thresh = 0,
    oreder_weighted=True
):
    results_of_key = result_dict[key]
    filtered = list(filter(lambda el: el["score"] >= score_thresh, results_of_key))
    if len(filtered) == 0:
        return
    filtered.sort(key=lambda el: el["score"] * (el["quantized"]["size_difference"] if oreder_weighted else 1), reverse=True)
    out_dict[key] = filtered[0]

def write_json(dict: Dict):
    with open("quantization_results.json", "w") as file_handle:
        json.dump(dict, file_handle, indent=4, default=lambda x: x.name)
