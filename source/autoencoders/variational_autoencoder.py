from pathlib import Path
from typing import Dict, Self, Tuple, Union, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from ._base_class import _BaseAutoEncoder, _BaseBlock


class EncoderBlock(_BaseBlock):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(input_size, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        h = self.layer(x)
        return F.leaky_relu(h)


class DecoderBlock(_BaseBlock):
    def __init__(self, embedding_size: int, input_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(embedding_size, input_size)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.layer(x))


class VariationalAutoEncoder(_BaseAutoEncoder):
    def __init__(
        self, input_size: int, embedding_size: int, latent_channel: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.latent_channel = latent_channel
        self.encoder_block = EncoderBlock(input_size, embedding_size)
        self.decoder_block = DecoderBlock(embedding_size, input_size)
        self.latent_block = nn.Linear(latent_channel, embedding_size)
        self.avg = nn.Linear(embedding_size, latent_channel)
        self.var = nn.Linear(embedding_size, latent_channel)

        self.optimizer = optim.Adam(self.parameters())

    def encode(self, x: Tensor) -> Tuple[Tensor, ...]:
        h = self.encoder_block(x)
        avg = self.avg(h)
        var = F.softplus(self.var(h))
        return avg, var

    def decode(self, z: Tensor) -> Tensor:
        z = torch.nn.functional.leaky_relu(self.latent_block(z))
        y = self.decoder_block(z)
        return y

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        avg, var = self.encode(x)
        z = self.reparameterize(avg, var)
        y = self.decode(z)
        return y, avg, var

    def reparameterize(self, avg: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn_like(avg)
        return avg + torch.sqrt(var) * eps

    def criterion(
        self, x: Tensor, y: Tensor, avg: Tensor, var: Tensor
    ) -> Tensor:
        bce = F.binary_cross_entropy(y, x, reduction="sum")
        kld = -torch.sum(1 + torch.log(var) - avg**2 - var) / 2
        return bce + kld

    def save(self, filename: Path | str) -> None:
        self.cpu()
        model: Dict[str, Any] = dict()
        model["name"] = self.__class__.__name__
        model["input_size"] = self.input_size
        model["embedding_size"] = self.embedding_size
        model["latent_channel"] = self.latent_channel
        model["model_state_dict"] = self.state_dict()
        torch.save(model, filename)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        params = torch.load(filename, weights_only=True)
        model = cls(
            params["input_size"],
            params["embedding_size"],
            params["latent_channel"],
        )
        model.load_state_dict(params["model_state_dict"])
        return model

    def step(self, dataloader: DataLoader, device: str) -> float:
        self.to(device)
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            pred, avg, var = self(x)
            loss = self.criterion(x, pred, avg, var)
            try:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            except RuntimeError:
                pass
            total_loss += loss.item()
        return total_loss / len(dataloader)
