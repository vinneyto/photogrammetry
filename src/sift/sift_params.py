from dataclasses import dataclass


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