from pathlib import Path
import cv2
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt

from octave import Octave


def gaussian_blur(img, sigma):
    blurred = cv2.GaussianBlur(
        img.squeeze(0).numpy(),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return torch.from_numpy(blurred).unsqueeze(0)


def draw_octaves_grid(octaves: list[Octave], normalize=False):
    for octave in octaves:
        batch = torch.stack([t for t in octave.layers])

        grid = vutils.make_grid(batch, nrow=3, padding=2, normalize=normalize)
        grid = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 7))
        plt.title(f"{octave}")
        plt.imshow(grid)
        plt.axis("off")
        plt.show()


def assert_octaves(octaves: list[Octave], ref_octaves: list[Octave]):
    assert len(ref_octaves) == len(octaves)
    for ref_octave, octave in zip(ref_octaves, octaves):
        assert len(ref_octave.layers) == len(octave.layers)
        for ref_layer, layer in zip(ref_octave.layers, octave.layers):
            assert torch.allclose(ref_layer, layer, atol=1e-5, rtol=1e-5)
