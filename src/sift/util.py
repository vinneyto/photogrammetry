import math
from pathlib import Path
import cv2
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt

from keypoint import KeyPoint


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


def draw_keypoints(image: torch.Tensor, keypoints: list[KeyPoint]):
    image = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
    image = image.squeeze()

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image, cmap="gray", origin="upper")

    octaves = sorted({kp.octave for kp in keypoints})

    for i, octave_id in enumerate(octaves):
        points = [kp for kp in keypoints if kp.octave == octave_id]

        color = plt.cm.plasma(1 - i / max(len(octaves) - 1, 1))

        ax.scatter(
            [kp.x for kp in points],
            [kp.y for kp in points],
            s=25,
            color=color,
            edgecolors="black",
            linewidths=0.3,
            label=f"Octave {octave_id}: {len(points)}",
        )

    ax.legend()
    ax.set_title(f"SIFT: {len(keypoints)} points")
    ax.axis("off")
    plt.tight_layout()
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


def read_keypoints(path: Path | str) -> list[KeyPoint]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    node = fs.getNode("keypoints");

    keypoints:list[KeyPoint] = []

    for i in range(node.size()):
        kp_cv2 = node.at(i)

        # float x, float y, float size, float angle=-1, float response=0, int octave=0, int class_id=-1

        x = kp_cv2.at(0).real()
        y = kp_cv2.at(1).real()
        size = kp_cv2.at(2).real()
        _ = kp_cv2.at(3).real() # angle
        response = kp_cv2.at(4).real()
        octave = int(kp_cv2.at(5).real())
        class_id = int(kp_cv2.at(6).real())

        kp = KeyPoint(
            x=x, y=y, size=size, octave=octave, class_id=class_id, response=response
        )

        keypoints.append(kp)

    fs.release()

    return keypoints


def assert_keypoints(keypoints: list[KeyPoint], ref_keypoints: list[KeyPoint]):
    assert len(keypoints) == len(ref_keypoints)

    for kp, ref_kp in zip(keypoints, ref_keypoints):
        assert math.isclose(kp.x, ref_kp.x, rel_tol=1e-5, abs_tol=1e-3), f"kp.x = {kp.x}, ref_kp.x = {ref_kp.x}"
        assert math.isclose(kp.y, ref_kp.y, rel_tol=1e-5, abs_tol=1e-3)
        assert math.isclose(kp.size, ref_kp.size, rel_tol=1e-5, abs_tol=1e-3)
        assert kp.octave == ref_kp.octave
        assert kp.class_id == ref_kp.class_id
        assert math.isclose(kp.response, ref_kp.response, rel_tol=1e-5, abs_tol=1e-3)

    