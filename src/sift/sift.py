from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import torch

from util import gaussian_blur

@dataclass
class SIFTParams:
    nfeatures = 0
    n_octave_layers = 3
    contrast_threshold = 0.04
    edge_threshold = 10
    sigma = 1.6

    orient_peak_ratio = 0.8
    orient_nbins = 36
    upscale_first = True

    enable_subpixel_localization = True
    enable_edge_like_filtering = True
    enable_orientation_bin_interpolation = True
    enable_orientation_gaussian_weighting = True
    enable_orientation_subpixel_localization = True
    enable_descriptor_gaussian_weighting = True
    enable_descriptor_bin_interpolation = True


@dataclass
class Octave:
    layers: list[torch.Tensor]

    def __repr__(self):
        return f"n_layers: {len(self.layers)}, shape: {self.layers[0].shape}"


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
            base = layers[s][:, ::2, ::2]

        octaves.append(Octave(layers))

    return octaves


def read_octaves(path: Path | str):
    # with gzip.open(path, 'rt', encoding='utf-8') as f:
    #     print(f.read())
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    num_octaves = int(fs.getNode("num_octaves").real())
    octaves: list[Octave] = []

    for o in range(num_octaves):
        layers: list[torch.Tensor] = []
        num_layers = int(fs.getNode(f"octave_{o}_num_layers").real())
        for l in range(num_layers):
            layer = fs.getNode(f"octave_{o}_layer_{l}").mat()
            layer = torch.from_numpy(layer).unsqueeze(0)
            layers.append(layer)
        octaves.append(Octave(layers))

    fs.release()

    return octaves