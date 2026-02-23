from torch import nn
import torch

class SimpleNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, device: torch.device = "cuda:0" if torch.cuda.is_available() else "cpu"):
        super(SimpleNN, self).__init__()
        self.layers_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=hidden_dim, device=device),
            nn.Linear(in_features=hidden_dim, out_features=output_dim, device=device),
        )

    def forward(self, x):
        return self.layers_stack(x)
