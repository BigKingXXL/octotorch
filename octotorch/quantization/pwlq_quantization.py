from typing import Any, Dict
import torch
from . import Quantization
from PWLQ.quant_weis import quant_weights
from types import SimpleNamespace
from enum import Enum

class BreakPointApproach(Enum):
    LAPLACE="laplace"
    NORM="norm"
    SEARCH="search"
    def __str__(self):
        return str(self.value)

class PwlqQuantization(Quantization):
    def __init__(
        self,
        bias_correction=True,
        scale_bits=0.0,
        break_point_approach: BreakPointApproach=BreakPointApproach.NORM,
        approximate=True
    ):
        self.bias_correction = bias_correction
        self.scale_bits = scale_bits
        self.break_point_approach = str(break_point_approach)
        self.approximate = approximate

    def get_setting_dict(self) -> Dict[str, Any]:
        return {
            "scale_bits": self.scale_bits,
            "break_point": self.break_point_approach,
            "approximate": self.approximate,
            "bias_corr": self.bias_correction
        }

    def quantize(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Implements the Logarithmic Quantization proposed in
        Miyashita, D., Lee, E. H., Murmann, B. (2016). Convolutional Neural Networks using Logarithmic Data Representation.

        Args:
            tensor (torch.Tensor): The tensor to be quantized.
            bits (int): bitwidth
            full_scale_range (int): The range of the full scale.

        Returns:
            torch.Tensor: The quantized tensor.
        """
        args = {
            "bits": bits,
            "wei_quant_scheme": "pw-2",
            "wei_bits": bits,
            **self.get_setting_dict()
        }
        return quant_weights(tensor, SimpleNamespace(**args))[0]

    @property
    def name(self) -> str:
        return "PwlqQuantization"
