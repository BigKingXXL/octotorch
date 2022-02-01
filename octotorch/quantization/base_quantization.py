from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict
import torch

class Quantization(ABC):
    """Abstract base class for all Quantization implementations.
    """
    @abstractmethod
    def quantize(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def get_setting_dict(self) -> Dict[str, Any]:
        raise NotImplementedError
    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError
    def __dict__(self):
        return self.get_setting_dict() + {"name": self.name}
    def supports_bit(self, bit: int) -> bool:
        return True
