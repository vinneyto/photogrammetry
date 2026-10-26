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
    return torch.from_numpy(blurred)


def draw_octaves_grid(octaves: list[torch.Tensor], normalize=False):
    for i, octave in enumerate(octaves):
        batch = octave.unsqueeze(1)

        grid = vutils.make_grid(batch, nrow=3, padding=2, normalize=normalize)
        grid = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 7))
        plt.title(f"octave: {i}, shape: {octave.shape}")
        plt.imshow(grid)
        plt.axis("off")
        plt.show()


def assert_octaves(octaves: list[torch.Tensor], ref_octaves: list[torch.Tensor]):
    assert len(ref_octaves) == len(octaves)
    for ref_octave, octave in zip(ref_octaves, octaves):
        assert ref_octave.shape == octave.shape 
        assert torch.allclose(ref_octave, octave, atol=1e-5, rtol=1e-5)



def read_octaves(path: Path | str) -> list[torch.Tensor]:
    # with gzip.open(path, 'rt', encoding='utf-8') as f:
    #     print(f.read())
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    num_octaves = int(fs.getNode("num_octaves").real())
    octaves: list[torch.Tensor] = []

    for o in range(num_octaves):
        layers: list[torch.Tensor] = []
        num_layers = int(fs.getNode(f"octave_{o}_num_layers").real())
        for l in range(num_layers):
            layer = fs.getNode(f"octave_{o}_layer_{l}").mat()
            layer = torch.from_numpy(layer)
            layers.append(layer)
        octaves.append(torch.stack(layers))

    fs.release()

    return octaves