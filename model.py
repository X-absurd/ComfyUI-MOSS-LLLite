"""MOSS tile residual architecture, shared with diffusion-pipe/models/moss_tile_control.py."""

import torch
from torch import nn
import torch.nn.functional as F


FORMAT = "moss_tile_residual"
VERSION = "1"
PREFIX = "moss_tile."


class MossTileConditioner(nn.Module):
    def __init__(self, latent_channels, condition_dim, *, operations=nn, device=None, dtype=None):
        super().__init__()
        self.conv1 = operations.Conv2d(latent_channels, condition_dim, 3, padding=1, device=device, dtype=dtype)
        self.conv2 = operations.Conv2d(condition_dim, condition_dim, 3, padding=1, device=device, dtype=dtype)
        self.proj = operations.Conv2d(condition_dim, condition_dim, 1, device=device, dtype=dtype)

    def forward(self, latent):
        x = F.silu(self.conv1(latent))
        x = self.proj(F.silu(self.conv2(x)))
        return x.flatten(2).transpose(1, 2)


class MossTileResidual(nn.Module):
    def __init__(self, model_dim, condition_dim, rank, *, operations=nn, device=None, dtype=None):
        super().__init__()
        self.down = operations.Linear(model_dim, rank, device=device, dtype=dtype)
        self.mid = operations.Linear(rank, rank, device=device, dtype=dtype)
        self.cond_to_film = operations.Linear(condition_dim, 2 * rank, device=device, dtype=dtype)
        self.up = operations.Linear(rank, model_dim, device=device, dtype=dtype)

    def forward(self, x, condition, strength=1.0):
        count = condition.shape[1]
        image = x[:, :count]
        hidden = F.silu(self.down(F.layer_norm(image, (image.shape[-1],), eps=1e-6)))
        gamma, beta = self.cond_to_film(condition).chunk(2, dim=-1)
        hidden = F.silu(self.mid(hidden) * (1 + gamma) + beta)
        image = image + self.up(hidden) * strength
        return torch.cat((image, x[:, count:]), dim=1)


class MossTileControl(nn.Module):
    def __init__(self, model_dim, num_blocks, rank, condition_dim=128, latent_channels=128, *, operations=nn, device=None, dtype=None):
        super().__init__()
        self.config = dict(model_dim=model_dim, num_blocks=num_blocks, rank=rank,
                           condition_dim=condition_dim, latent_channels=latent_channels)
        if any(type(value) is not int or value <= 0 for value in self.config.values()):
            raise ValueError("MOSS tile dimensions must be positive integers")
        self.conditioner = MossTileConditioner(latent_channels, condition_dim, operations=operations, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            MossTileResidual(model_dim, condition_dim, rank, operations=operations, device=device, dtype=dtype)
            for _ in range(num_blocks)
        ])

    def metadata(self):
        return {"format": FORMAT, "moss_tile.version": VERSION,
                **{f"moss_tile.{key}": str(value) for key, value in self.config.items()}}


def config_from_metadata(metadata):
    if metadata is None or metadata.get("format") != FORMAT or metadata.get("moss_tile.version") != VERSION:
        raise ValueError("Expected a MOSS tile residual v1 checkpoint; legacy concat/LoRA weights cannot be loaded here")
    return {key: int(metadata[f"moss_tile.{key}"])
            for key in ("model_dim", "num_blocks", "rank", "condition_dim", "latent_channels")}
