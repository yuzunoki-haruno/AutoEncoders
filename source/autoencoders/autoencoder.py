from pathlib import Path
from typing import Any, Dict, Self, Union

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from ._base_class import _BaseAutoEncoder, _BaseBlock


class EncoderBlock(_BaseBlock):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(input_size, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        h = self.batch_norm(self.layer(x))
        return torch.nn.functional.leaky_relu(h)


class DecoderBlock(_BaseBlock):
    def __init__(self, embedding_size: int, input_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(embedding_size, input_size)
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, x: Tensor) -> Tensor:
        h = self.batch_norm(self.layer(x))
        return torch.sigmoid(h)


class AutoEncoder(_BaseAutoEncoder):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.encoder_block = EncoderBlock(input_size, embedding_size)
        self.decoder_block = DecoderBlock(embedding_size, input_size)
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCELoss()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_block(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder_block(z)

    def forward(self, x: Tensor) -> Tensor:
        y = self.encode(x)
        return self.decode(y)

    def save(self, filename: Union[str, Path]) -> None:
        self.cpu()
        model: Dict[str, Any] = dict()
        model["name"] = self.__class__.__name__
        model["input_size"] = self.input_size
        model["embedding_size"] = self.embedding_size
        model["state_dict"] = self.state_dict()
        torch.save(model, filename)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        params = torch.load(filename, weights_only=True)
        model = cls(params["input_size"], params["embedding_size"])
        model.load_state_dict(params["state_dict"])
        return model

    def step(self, dataloader: DataLoader, device: str) -> float:
        self.to(device)
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            y = self(x)
            loss = self.criterion(y, x)
            try:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            except RuntimeError:
                pass
            total_loss += loss.item()
        return total_loss / len(dataloader)
