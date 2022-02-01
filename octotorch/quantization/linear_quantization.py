from typing import Any, Dict
import torch
from . import Quantization

class LinearQuantization(Quantization):
    def __init__(self, full_scale_range: int = 7):
        self.full_scale_range = full_scale_range
    
    def get_setting_dict(self) -> Dict[str, Any]:
        return {
            "full_scale_range": self.full_scale_range
        }

    def quantize(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Implements the Linear Quantization proposed in
        Miyashita, D., Lee, E. H., Murmann, B. (2016). Convolutional Neural Networks using Logarithmic Data Representation.

        Args:
            tensor (torch.Tensor): The tensor to be quantized.
            bits (int): bitwidth

        Returns:
            torch.Tensor: The quantized tensor.
        """
        step = 2.0 ** (self.full_scale_range - bits)
        return tensor \
            .div(step) \
            .round_() \
            .mul_(step) \
            .clip_(0, 2.0 ** self.full_scale_range)
    
    @property
    def name(self) -> str:
        return "LinearQuantization"
