from typing import List, Dict

import torch
from torch import nn as nn


class EncoderBase(nn.Module):
    def __init__(self,
                 output_size: int,
                 max_variable_num_on_memory: int):
        nn.Module.__init__(self)

        self.output_size = output_size
        self.max_variable_num_on_memory = max_variable_num_on_memory

    def encode(self, examples: List) -> Dict:
        raise NotImplementedError


ContextEncoding = Dict[str, torch.Tensor]
COLUMN_TYPES = ['string', 'date', 'number', 'num1', 'num2']