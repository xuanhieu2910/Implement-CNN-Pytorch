from typing import Optional

import torch.nn as nn
from torch import Tensor


class Losser(nn.CrossEntropyLoss):
    def __init__(self,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = "mean",
                 label_smoothing: float = 0.0):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
