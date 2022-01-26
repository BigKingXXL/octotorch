from abc import ABC, abstractmethod
import torch

class Quantization(ABC):
    """Abstract base class for all Quantization implementations.
    """
    @abstractmethod
    def quantize(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
        
