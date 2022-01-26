import torch
from . import Quantization

class LogarithmicQuantization(Quantization):
    def __init__(self, full_scale_range: int = 7):
        self.full_scale_range = full_scale_range

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
        return torch.pow(2, (tensor \
            .abs_() \
            .log2_() \
            .round_() \
            .clip_(self.full_scale_range - (2**bits), self.full_scale_range))) \
            .mul_(tensor.not_equal(0))

    @property
    def name(self) -> str:
        return "LogarithmicQuantization"
