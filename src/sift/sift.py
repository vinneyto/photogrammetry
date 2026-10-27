from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from sift_params import SIFTParams
from util import gaussian_blur


@dataclass
class KeyPoint:
    x: float
    y: float
    size: float
    octave: int
    class_id: int
    response: float


def build_octaves(img: torch.Tensor, params: SIFTParams) -> list[torch.Tensor]: # [[n_layers, rows, cols]]
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

        octaves.append(torch.stack(layers))

    return octaves


def build_dog(octaves: list[torch.Tensor]):
    dog: list[torch.Tensor] = []

    for octave in octaves:
        dog_layers: list[torch.Tensor] = []
        for i in range(0, len(octave) - 1):
            dog_layers.append(octave[i+1] - octave[i])
        dog.append(torch.stack(dog_layers))
    return dog


def build_gradient_kernel(device=None):
    dn = torch.tensor([-0.5, 0, 0.5], device=device, dtype=torch.float32)

    grad_kernel = torch.zeros((3, 1, 3, 3, 3), device=device, dtype=torch.float32) # [grad_size(s,y,x), chann_in(gray_color), kernel_size x 3]
    grad_kernel[0, 0, :, 1, 1] = dn
    grad_kernel[1, 0, 1, :, 1] = dn
    grad_kernel[2, 0, 1, 1, :] = dn

    return grad_kernel


def build_hessian_kernel(device=None):
    dn = torch.tensor([-0.5, 0, 0.5], device=device, dtype=torch.float32)
    dnn = torch.tensor([1, -2, 1], device=device, dtype=torch.float32)

    # [(dss, dsy, dsx, dsy, dyy, dyx, dsx, dyx, dxx), channel, s, y, x]
    hessian_kernel = torch.zeros(
        (9, 1, 3, 3, 3), device=device, dtype=torch.float32
    )

    mixed = torch.outer(dn, dn)

    hessian_kernel[0, 0, :, 1, 1] = dnn    # dss
    hessian_kernel[1, 0, :, :, 1] = mixed  # dsy
    hessian_kernel[2, 0, :, 1, :] = mixed  # dsx

    hessian_kernel[3, 0, :, :, 1] = mixed  # dsy
    hessian_kernel[4, 0, 1, :, 1] = dnn    # dyy
    hessian_kernel[5, 0, 1, :, :] = mixed  # dyx

    hessian_kernel[6, 0, :, 1, :] = mixed  # dsx
    hessian_kernel[7, 0, 1, :, :] = mixed  # dyx
    hessian_kernel[8, 0, 1, 1, :] = dnn    # dxx

    return hessian_kernel


def find_scale_space_extrema(dog: list[torch.Tensor], params: SIFTParams):
    s = params.n_octave_layers
    sigma0 = params.sigma
    contrast_threshold = params.contrast_threshold
    edge_threshold = params.edge_threshold

    thresh = contrast_threshold / s

    first_octave = -1 if params.upscale_first else 0

    grad_kernel = build_gradient_kernel(device=dog[0].device)
    hessian_kernal = build_hessian_kernel(device=dog[0].device)

    for octave in dog:
        grad = F.conv3d(octave[None, None, ...], weight=grad_kernel) # [1, xyz, n_layer, rows, cols]
        hessian = F.conv3d(octave[None, None, ...], weight=hessian_kernal) # [1, mat3, n_layer, rows, cols]

        g = grad.movedim(1, -1) # [1, n_layer, rows, cols, xyz]
        H = hessian.movedim(1, -1).unflatten(-1, (3, 3)) # [1, n_layer, rows, cols, mat_rows, mat_cols]

        return torch.linalg.solve_ex(H, -g.unsqueeze(-1))
        # offset = offset.squeeze(-1)
