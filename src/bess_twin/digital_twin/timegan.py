"""TimeGAN model for synthetic time series generation."""

import torch
import torch.nn as nn


class TimeGANEmbedding(nn.Module):
    """Embedding network."""

    def __init__(self, dim_input: int, dim_hidden: int, dim_embedding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_embedding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeGANGenerator(nn.Module):
    """Generator network."""

    def __init__(self, dim_latent: int, dim_hidden: int, dim_output: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_latent, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TimeGANDiscriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, dim_input: int, dim_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeGAN(nn.Module):
    """Simplified TimeGAN model."""

    def __init__(self, seq_len: int, dim_input: int, dim_latent: int = 12, 
                 dim_hidden: int = 24, dim_embedding: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.dim_input = dim_input
        self.dim_latent = dim_latent

        self.embedding = TimeGANEmbedding(dim_input, dim_hidden, dim_embedding)
        self.generator = TimeGANGenerator(dim_latent, dim_hidden, dim_embedding)
        self.discriminator = TimeGANDiscriminator(dim_input, dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        e = self.embedding(x)
        return e

    def generate(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """Generate synthetic sequences."""
        z = torch.randn(batch_size, self.seq_len, self.dim_latent, device=device)
        z_flat = z.view(-1, self.dim_latent)
        fake = self.generator(z_flat)
        return fake.view(batch_size, self.seq_len, -1)
