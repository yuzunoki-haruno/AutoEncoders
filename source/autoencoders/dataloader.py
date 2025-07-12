from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from torch.utils.data import DataLoader
from torchvision.datasets import (
    CIFAR10,
    MNIST,
    FashionMNIST,
    ImageFolder,
)
from torchvision.transforms import Compose


def get_dataloader(
    filename: Union[str, Path], transform: Compose, params: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, List[str]]:
    path = Path(filename)
    name, root = path.name, path.parent
    if name == "MNIST":
        train_dataset = MNIST(root, train=True, transform=transform)
        test_dataset = MNIST(root, train=False, transform=transform)
    elif name == "FashionMNIST":
        train_dataset = FashionMNIST(root, train=True, transform=transform)
        test_dataset = FashionMNIST(root, train=False, transform=transform)
    elif name == "CIFAR10":
        train_dataset = CIFAR10(path, train=True, transform=transform)
        test_dataset = CIFAR10(path, train=False, transform=transform)
    else:
        train_dataset = ImageFolder(path / "train", transform=transform)
        test_dataset = ImageFolder(path / "test", transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )
    return train_dataloader, test_dataloader, train_dataset.classes


def get_default_shape(filename: Union[str, Path]) -> Dict[str, int]:
    path = Path(filename)
    name = path.name
    if name == "MNIST":
        channel, height, width = 1, 28, 28
    elif name == "FashionMNIST":
        channel, height, width = 1, 28, 28
    elif name == "CIFAR10":
        channel, height, width = 3, 32, 32
    else:
        channel, height, width = 1, 256, 256
    return {"channel": channel, "height": height, "width": width}
