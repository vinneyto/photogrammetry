import math
from pathlib import Path
import numpy as np
import cv2
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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


def draw_orientation_weights(img, weight, xi, yi):
    image = img.detach().cpu().numpy()
    weights = weight.detach().cpu().numpy()

    h, w = weights.shape
    x_min = xi - w // 2
    y_min = yi - h // 2

    tile = image[y_min:y_min + h, x_min:x_min + w]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].add_patch(Rectangle(
        (x_min - 0.5, y_min - 0.5),
        w, h,
        edgecolor="red",
        facecolor="none",
        linewidth=1.5,
    ))
    axes[0].plot(xi, yi, "r+", markersize=8)
    axes[0].set_title(f"Keypoint ({xi}, {yi})")

    axes[1].imshow(tile, cmap="gray")
    overlay = axes[1].imshow(
        weights,
        cmap="magma",
        alpha=0.65,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].plot(w // 2, h // 2, "c+", markersize=8)
    axes[1].set_title("Gaussian weights")
    fig.colorbar(overlay, ax=axes[1], label="Weight")

    plt.tight_layout()
    plt.show()


class OrientationDebugger:
    def __init__(self, image, n=6):
        self.image = self._numpy(image).squeeze()
        if self.image.ndim != 2:
            raise ValueError("Expected grayscale image [H, W] or [1, H, W]")

        self.n = n
        self.points = []

    @staticmethod
    def _numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.array(value, copy=True)

    def add(
        self,
        histogram,
        img_tile,
        xi,
        yi,
        radius,
        octave,
        layer=None,
        label=None,
    ):
        if len(self.points) >= self.n:
            return

        self.points.append({
            "hist": self._numpy(histogram).reshape(-1),
            "tile": self._numpy(img_tile).squeeze(),
            "xi": xi,
            "yi": yi,
            "radius": radius,
            "octave": octave,
            "layer": layer,
            "label": str(
                label if label is not None else len(self.points)
            ),
        })

    @staticmethod
    def _draw_histogram(ax, hist, x, y, length, color):
        angles = np.arange(hist.size) * (2 * np.pi / hist.size)

        peak = hist.max()
        lengths = (
            hist / peak * length
            if peak > 0
            else np.zeros_like(hist)
        )

        ax.quiver(
            np.full(hist.size, x),
            np.full(hist.size, y),
            lengths * np.cos(angles),
            lengths * np.sin(angles),
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="tail",
            width=0.004,
        )
        ax.plot(x, y, "+", color=color, markersize=7)

    def draw(self, columns=3):
        if not self.points:
            return

        columns = max(1, min(columns, len(self.points)))
        rows = (len(self.points) + columns - 1) // columns

        fig = plt.figure(
            figsize=(8 + 3.5 * columns, max(6, 3.5 * rows)),
            constrained_layout=True,
        )
        outer = fig.add_gridspec(
            1, 2, width_ratios=[1, 1.3]
        )

        overview = fig.add_subplot(outer[0, 0])
        overview.imshow(
            self.image, cmap="gray", origin="upper"
        )
        overview.set_title("Keypoints and tile bounds")

        grid = outer[0, 1].subgridspec(rows, columns)
        colors = plt.get_cmap("tab10")

        for i, point in enumerate(self.points):
            color = colors(i % 10)
            hist = point["hist"]
            tile = point["tile"]
            radius = point["radius"]
            label = point["label"]

            scale = 2.0 ** point["octave"]
            x = point["xi"] * scale
            y = point["yi"] * scale

            h, w = tile.shape

            # Рамка тайла в координатах исходного изображения
            overview.add_patch(Rectangle(
                (
                    x - (radius + 0.5) * scale,
                    y - (radius + 0.5) * scale,
                ),
                w * scale,
                h * scale,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
            ))

            self._draw_histogram(
                overview, hist, x, y,
                radius * scale, color,
            )

            overview.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color=color,
                fontweight="bold",
                bbox=dict(
                    facecolor="black",
                    alpha=0.6,
                    edgecolor="none",
                ),
            )

            # Справа — реальный тайл соответствующего слоя
            ax = fig.add_subplot(
                grid[i // columns, i % columns]
            )
            ax.imshow(
                tile,
                cmap="gray",
                origin="upper",
                interpolation="nearest",
            )

            ax.add_patch(Rectangle(
                (-0.5, -0.5),
                w, h,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
                clip_on=False,
            ))

            self._draw_histogram(
                ax, hist, radius, radius,
                radius, color,
            )

            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_title(
                f"Point {label} · octave {point['octave']}"
                f" · layer {point['layer']}"
            )

        # Не даём стрелкам и рамкам менять границы обзора
        h, w = self.image.shape
        overview.set_xlim(-0.5, w - 0.5)
        overview.set_ylim(h - 0.5, -0.5)

        plt.show()
        # return fig


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
        angle = kp_cv2.at(3).real()
        response = kp_cv2.at(4).real()
        octave = int(kp_cv2.at(5).real())
        class_id = int(kp_cv2.at(6).real())

        kp = KeyPoint(
            x=x, y=y, size=size, octave=octave, class_id=class_id, response=response, angle=angle
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
        assert math.isclose(kp.angle, ref_kp.angle, rel_tol=1e-5, abs_tol=1e-3)

    