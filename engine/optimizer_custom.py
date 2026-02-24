from torch.optim import SGD
from torch.optim.optimizer import ParamsT


class OptimizerCustom(SGD):
    def __init__(self, params: ParamsT, learning_rate: float):
        super().__init__(params = params, lr=learning_rate)
