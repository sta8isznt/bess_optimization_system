import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Core Modules
# =========================

class RNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.fc(h)


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = RNNBlock(input_dim, hidden_dim, hidden_dim)

    def forward(self, x):
        return self.net(x)


class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = RNNBlock(hidden_dim, hidden_dim, output_dim)

    def forward(self, h):
        return self.net(h)


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim):
        super().__init__()
        self.net = RNNBlock(noise_dim, hidden_dim, hidden_dim)

    def forward(self, z):
        return self.net(z)


class Supervisor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = RNNBlock(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, h):
        return self.net(h)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        h, _ = self.rnn(h)
        y = self.fc(h)
        return y


# =========================
# TimeGAN Wrapper
# =========================

class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim):
        super().__init__()

        self.embedder = Embedder(input_dim, hidden_dim)
        self.recovery = Recovery(hidden_dim, input_dim)

        self.generator = Generator(noise_dim, hidden_dim)
        self.supervisor = Supervisor(hidden_dim)

        self.discriminator = Discriminator(hidden_dim)

    def forward(self, x, z):
        # Real embedding
        h = self.embedder(x)
        x_tilde = self.recovery(h)

        # Synthetic path
        h_hat = self.generator(z)
        h_hat_sup = self.supervisor(h_hat)
        x_hat = self.recovery(h_hat_sup)

        return x_tilde, x_hat, h, h_hat_sup