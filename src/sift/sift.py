from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from sift_params import SIFTParams
from keypoint import KeyPoint
from util import gaussian_blur


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


def find_scale_space_extrema(dog: list[torch.Tensor], params: SIFTParams) -> list[KeyPoint]:
    s = params.n_octave_layers
    sigma0 = params.sigma
    contrast_threshold = params.contrast_threshold
    edge_threshold = params.edge_threshold
    enable_subpixel_localization = params.enable_subpixel_localization

    thresh = contrast_threshold / s

    first_octave = -1 if params.upscale_first else 0

    border = 5

    grad_kernel = build_gradient_kernel(device=dog[0].device)
    hessian_kernel = build_hessian_kernel(device=dog[0].device)

    keypoints: list[KeyPoint] = []

    for i, octave in enumerate(dog):
        real_octave = i + first_octave
        
        # check for threshold
        center = octave[1:-1, 1:-1, 1:-1]
        contrast_mask = center.abs() > thresh * 0.5

        # check for local min/max
        S, rows, cols = octave.shape

        is_max = torch.ones_like(center, dtype=torch.bool)
        is_min = torch.ones_like(center, dtype=torch.bool)

        for ds in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if ds == dy == dx == 0:
                        continue

                    neighbor = octave[
                        1 + ds : S - 1 + ds,
                        1 + dy : rows - 1 + dy,
                        1 + dx : cols - 1 + dx,
                    ]

                    is_max &= center > neighbor
                    is_min &= center < neighbor
        
        extrema_mask = is_max | is_min

        # solve linear systems (finding local extrema)
        grad = F.conv3d(octave[None, None, ...], weight=grad_kernel)
        hessian = F.conv3d(octave[None, None, ...], weight=hessian_kernel)

        g0 = grad.movedim(1, -1)[0]  # [S, rows, cols, 3]
        H0 = hessian.movedim(1, -1).unflatten(-1, (3, 3))[0]

        S, rows, cols = g0.shape[:3]
        device = g0.device

        ss, yy, xx = torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing="ij",
        )

        coords = torch.stack([ss, yy, xx], dim=-1).reshape(-1, 3)

        lower = torch.tensor([0, border - 1, border - 1], device=device)
        upper = torch.tensor([S, rows - border + 1, cols - border + 1], device=device)

        candidate_mask = contrast_mask & extrema_mask

        coords = coords[candidate_mask.reshape(-1)]
        inside = ((coords >= lower) & (coords < upper)).all(dim=-1)
        coords = coords[inside]

        ready_coords = []
        ready_offsets = []

        # 5 attempts to have local offsets less then 0.5
        for _ in range(5):
            if coords.shape[0] == 0:
                break

            si, yi, xi = coords.unbind(dim=-1)

            offset, info = torch.linalg.solve_ex(
                H0[si, yi, xi],
                -g0[si, yi, xi].unsqueeze(-1),
            )
            offset = offset.squeeze(-1)  # [N_active, syx]

            solved = (info == 0) & torch.isfinite(offset).all(dim=-1)
            ready = solved & (offset.abs() < 0.5).all(dim=-1)
            move = solved & ~ready

            ready_coords.append(coords[ready])
            ready_offsets.append(offset[ready])

            coords = coords[move] + offset[move].round().long()

            inside = ((coords >= lower) & (coords < upper)).all(dim=-1)
            coords = coords[inside]

        ready_coords = (
            torch.cat(ready_coords)
            if ready_coords
            else torch.empty((0, 3), dtype=torch.long, device=device)
        ) # [K, syx]

        ready_offsets = (
            torch.cat(ready_offsets)
            if ready_offsets
            else g0.new_empty((0, 3))
        ) # [K, syx]

        # finding floating point response (px response + offset)
        si, yi, xi = ready_coords.unbind(dim=-1)
        g_ready = g0[si, yi, xi]
        center_ready = center[si, yi, xi]

        dot = (ready_offsets * g_ready).sum(dim=-1)
        response_optimized = center_ready + 0.5 * dot
        response_mask = response_optimized.abs() > thresh

        ready_coords = ready_coords[response_mask]
        ready_offsets = ready_offsets[response_mask]
        response_optimized = response_optimized[response_mask]

        # checking for enable_edge_like_filtering
        if params.enable_edge_like_filtering:
            si, yi, xi = ready_coords.unbind(dim=-1)
            H0_ready = H0[si, yi, xi]
            # dxx dxy, dxy, dyy
            dxx = H0_ready[:, 2, 2]
            dxy = H0_ready[:, 1, 2]
            dyy = H0_ready[:, 1, 1]

            trace = dxx + dyy
            det = dxx * dyy - dxy * dxy
            det_mask = det > 0

            ready_coords = ready_coords[det_mask]
            ready_offsets = ready_offsets[det_mask]
            response_optimized = response_optimized[det_mask]
            trace = trace[det_mask]
            det = det[det_mask]

            r = edge_threshold
            edge_mask = (trace * trace / det) <= (r + 1) ** 2 / r

            ready_coords = ready_coords[edge_mask]
            ready_offsets = ready_offsets[edge_mask]
            response_optimized = response_optimized[edge_mask]

        # upscale coords
        scale = 2 ** real_octave
        upscaled_coords = ready_coords.to(ready_offsets.dtype) + 1
        layers = ready_coords[:, 0] + 1

        if enable_subpixel_localization:
            upscaled_coords += ready_offsets

        upscaled_coords[:, 1:] *= scale

        kp_sigma = sigma0 * 2 ** (upscaled_coords[:, 0] / s) * scale

        for coord, response, sigma, layer in zip(
            upscaled_coords, response_optimized, kp_sigma, layers
        ):
            point = KeyPoint(
                x=coord[2].item(),
                y=coord[1].item(),
                size=2.0 * sigma.item(),
                octave=real_octave,
                class_id=int(layer.item()),
                response=response.abs().item()
            )
            keypoints.append(point)

    return keypoints
