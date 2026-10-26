from dataclasses import dataclass

import torch


@dataclass
class Octave:
    layers: torch.Tensor # [n_layers, rows, cols]

    def __repr__(self):
        return f"n_layers, rows, colls: {self.layers.shape}"