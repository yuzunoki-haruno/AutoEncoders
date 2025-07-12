import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml  # type:ignore
from numpy.typing import NDArray
from skimage import metrics
from sklearn.manifold import TSNE
from torch import Tensor

UC_MAX = 255


def read_params_from_yaml(filename: Union[str, Path]) -> Dict[str, Any]:
    with open(filename, "r") as file:
        params = yaml.safe_load(file)
    return params


def dict_to_csv(
    filename: Union[str, Path], dictionary: Dict[str, Any]
) -> None:
    with open(filename, "w") as file:
        writer = csv.writer(file)
        for key in dictionary:
            writer.writerow([key, dictionary[key]])


def save_loss_curve(
    directory: Union[str, Path], loss_curve: List[float]
) -> None:
    path = Path(directory)
    # plot
    filename = path / "loss_curve.png"
    fig, ax = plt.subplots()
    ax.plot(loss_curve)
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(filename)
    # to csv
    filename = path / "loss_curve.csv"
    pd.Series(loss_curve, name="loss").to_csv(filename)


def reconstruct_embeddings(
    filename: Union[str, Path], x: Tensor, y: Tensor
) -> None:
    z = torch.abs(x - y)
    grid_x = torchvision.utils.make_grid(x, padding=5, pad_value=1)
    grid_y = torchvision.utils.make_grid(y, padding=5, pad_value=1)
    grid_z = torchvision.utils.make_grid(z, padding=5, pad_value=1)
    torchvision.utils.save_image(
        [grid_x, grid_y, grid_z], filename, nrow=3, padding=5, pad_value=1
    )


def plot_embeddings(
    filename: Union[str, Path], vector: List[List], labels: List
) -> None:
    x = np.array(vector)
    y = np.array(labels)
    label_name = sorted(set(labels))
    z = TSNE(n_components=2, max_iter=500).fit_transform(x)
    fig, ax = plt.subplots()
    for label in label_name:
        mask = y == label
        ax.scatter(z[mask, 0], z[mask, 1], label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)


def ssim_loss(input: Tensor, target: Tensor, shape: Tuple) -> float:
    ssim = 0.0
    channel, _, _ = shape
    x = _tendor_to_image(input, shape)
    y = _tendor_to_image(target, shape)
    for x_, y_ in zip(x, y):
        ssim += _calc_ssim(x_, y_, channel)
    ssim /= min(x.shape[0], y.shape[0])
    return ssim


def _calc_ssim(x: NDArray, y: NDArray, channel: int) -> float:
    if channel == 3:
        ssim_r = metrics.structural_similarity(x[0], y[0], data_range=UC_MAX)
        ssim_g = metrics.structural_similarity(x[1], y[1], data_range=UC_MAX)
        ssim_b = metrics.structural_similarity(x[2], y[2], data_range=UC_MAX)
        assert isinstance(ssim_r, float), "The array shape is invalid."
        assert isinstance(ssim_g, float), "The array shape is invalid."
        assert isinstance(ssim_b, float), "The array shape is invalid."
        return (ssim_r + ssim_g + ssim_b) / 3.0
    else:
        ssim = metrics.structural_similarity(x[0], y[0], data_range=UC_MAX)
        assert isinstance(ssim, float), "The array shape is invalid."
        return ssim


def psnr_loss(input: Tensor, target: Tensor, shape: Tuple) -> float:
    ssim = 0.0
    channel, _, _ = shape
    x = _tendor_to_image(input, shape)
    y = _tendor_to_image(target, shape)
    for x_, y_ in zip(x, y):
        ssim += _calc_psnr(x_, y_, channel)
    ssim /= min(x.shape[0], y.shape[0])
    return ssim


def _calc_psnr(x: NDArray, y: NDArray, channel: int) -> float:
    if channel == 3:
        ssim_r = metrics.peak_signal_noise_ratio(x[0], y[0], data_range=UC_MAX)
        ssim_g = metrics.peak_signal_noise_ratio(x[1], y[1], data_range=UC_MAX)
        ssim_b = metrics.peak_signal_noise_ratio(x[2], y[2], data_range=UC_MAX)
        return (ssim_r + ssim_g + ssim_b) / 3.0
    else:
        return metrics.peak_signal_noise_ratio(x[0], y[0], data_range=UC_MAX)


def _tendor_to_image(input: Tensor, shape: Tuple) -> NDArray:
    channel, height, width = shape
    image = input.view(-1, channel, height, width).detach().cpu().numpy()
    image *= UC_MAX
    return image.astype(np.uint8)
