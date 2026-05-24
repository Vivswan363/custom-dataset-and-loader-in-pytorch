import torch.nn as nn


def build_model(input_dim: int = 79, output_dim: int = 1):
    return nn.Linear(input_dim, output_dim)