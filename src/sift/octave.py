from dataclasses import dataclass

import torch


@dataclass
class Octave:
    layers: list[torch.Tensor]

    def __repr__(self):
        return f"n_layers: {len(self.layers)}, shape: {self.layers[0].shape}"