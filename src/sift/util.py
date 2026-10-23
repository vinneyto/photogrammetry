from pathlib import Path
import cv2
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt


def gaussian_blur(img, sigma):
    blurred = cv2.GaussianBlur(
        img.squeeze(0).numpy(),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return torch.from_numpy(blurred).unsqueeze(0)


def draw_octaves_grid(octaves):
    for octave in octaves:
        batch = torch.stack([t for t in octave.layers])

        grid = vutils.make_grid(batch, nrow=3, padding=2, normalize=False)
        grid = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 7))
        plt.title(f"{octave}")
        plt.imshow(grid)
        plt.axis("off")
        plt.show()
