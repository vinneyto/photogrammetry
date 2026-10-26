from dataclasses import dataclass
import math
from pathlib import Path

import torch

from octave import Octave
from sift_params import SIFTParams
from util import gaussian_blur


def build_octaves(img: torch.Tensor, params: SIFTParams) -> list[Octave]:
    s = params.n_octave_layers
    sigma0 = params.sigma
    sigma_nominal = 1.0 if params.upscale_first else 0.5
    n_layers = s + 3

    rows, cols = img.shape[-2], img.shape[-1]

    n_octaves = max(1, round(math.log2(min(rows, cols))) - 3)
    sigma_base = math.sqrt(sigma0 ** 2 - sigma_nominal ** 2)

    base = gaussian_blur(img, sigma_base)

    octaves = []
    for octave_idx in range(n_octaves):
        layers = [base.clone()]

        for i in range(1, n_layers):
            sigma_layer = sigma0 * (2.0 ** (i / s))
            sigma_layer = math.sqrt(sigma_layer ** 2 - sigma0 ** 2)
            layers.append(gaussian_blur(layers[0], sigma_layer))

        if octave_idx + 1 < n_octaves:
            base = layers[s][::2, ::2]

        octaves.append(Octave(torch.stack(layers)))

    return octaves


def build_dog(octaves: list[Octave]):
    dog: list[Octave] = []

    for octave in octaves:
        dog_layers: list[torch.Tensor] = []
        for i in range(0, len(octave.layers) - 1):
            dog_layers.append(octave.layers[i+1] - octave.layers[i])
        dog.append(Octave(layers=torch.stack(dog_layers)))
    return dog


def find_scale_space_extrema(dog: list[Octave], params: SIFTParams):
    s = params.n_octave_layers
    sigma0 = params.sigma
    contrast_threshold = params.contrast_threshold
    edge_threshold = params.edge_threshold

    thresh = contrast_threshold / s

    first_octave = -1 if params.upscale_first else 0
