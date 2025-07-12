from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Self, Tuple, Union

from torch import Tensor, nn, no_grad
from torch.utils.data import DataLoader


class _BaseBlock(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(_BaseBlock, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...


class _BaseAutoEncoder(nn.Module, metaclass=ABCMeta):

    def __init__(self) -> None:
        super(_BaseAutoEncoder, self).__init__()

    @abstractmethod
    def encode(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor: ...

    @abstractmethod
    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    @abstractmethod
    def save(self, filename: Union[str, Path]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, filename: Union[str, Path]) -> Self: ...

    @abstractmethod
    def step(self, dataloader: DataLoader, device: str) -> float: ...

    def train_step(self, dataloader: DataLoader, device: str) -> float:
        self.train()
        return self.step(dataloader, device)

    def test_step(self, dataloader: DataLoader, device: str) -> float:
        self.eval()
        with no_grad():
            return self.step(dataloader, device)

    def reparameterize(self, avg: Tensor, var: Tensor) -> Tensor:
        raise NotImplementedError
