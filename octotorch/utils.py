from typing import OrderedDict
import torch

def copy_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return {key: tensor.clone() for key, tensor in state_dict.items()}
