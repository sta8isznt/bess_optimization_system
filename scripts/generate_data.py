"""Train TimeGAN on user-provided time series and generate augmented samples."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.bess_twin.utils.config import load_digital_twin_config
from src.bess_twin.utils.seed import set_seed
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def _windows_from_series(series: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Convert a single series [T, F] into overlapping windows [N, seq_len, F]."""
    if series.ndim != 2:
        raise ValueError(f"Expected a 2D array [T, F], got shape {series.shape}")

    t, f = series.shape
    if t < seq_len:
        return np.zeros((0, seq_len, f), dtype=series.dtype)

    starts = range(0, t - seq_len + 1, stride)
    return np.stack([series[s:s + seq_len] for s in starts], axis=0)


def _load_user_data(input_path: str) -> np.ndarray:
    """Load user-provided time series from .npy or .csv files.

    Returns:
        Array shaped [N, T, F] or [T, F].
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")

    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        data = df.select_dtypes(include=[np.number]).to_numpy()
    else:
        raise ValueError("Unsupported input format. Use .npy or .csv")

    if data.ndim == 1:
        data = data[:, None]

    return data


def _prepare_training_windows(data: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Accept either a single series [T, F] or a batch [N, T, F] and window it."""
    if data.ndim == 2:
        windows = _windows_from_series(data, seq_len=seq_len, stride=stride)
    elif data.ndim == 3:
        window_list = [_windows_from_series(series, seq_len=seq_len, stride=stride) for series in data]
        window_list = [w for w in window_list if w.shape[0] > 0]
        windows = np.concatenate(window_list, axis=0) if window_list else np.zeros((0, seq_len, data.shape[-1]), dtype=data.dtype)
    else:
        raise ValueError(f"Expected input shape [T, F] or [N, T, F], got {data.shape}")

    if windows.shape[0] == 0:
        raise ValueError(
            f"No training windows could be created. Input length must be >= seq_len ({seq_len})."
        )

    return windows


def augment_timeseries_from_input(
    config: dict,
    input_path: str,
    output_name: str = "augmented_timeseries.npy",
    train_gan: bool = True,
    gan_epochs: int = 10,
    augment_factor: float = 1.0,
):
    """Train TimeGAN only on user-provided time series and generate augmented samples."""
    tg_config = config["digital_twin"]["timegan"]
    set_seed(config.get("seed", 42))

    from src.time_gan.gan import TimeGAN
    from src.time_gan.trainer import generate, train_embedder, train_joint, train_supervisor

    raw_data = _load_user_data(input_path)
    seq_len = tg_config.get("seq_len", config.get("window_size", 128))
    stride = tg_config.get("stride", max(1, seq_len // 2))
    windows = _prepare_training_windows(raw_data, seq_len=seq_len, stride=stride)

    logger.info(f"Loaded input data {raw_data.shape} and created {windows.shape[0]} windows for training")

    X = torch.tensor(windows, dtype=torch.float32)
    input_dim = X.shape[-1]
    hidden = tg_config.get("dim_hidden", 24)
    noise_dim = tg_config.get("dim_latent", 12)
    batch_size = tg_config.get("batch_size", 32)
    device = torch.device(config.get("device", "cpu"))

    model = TimeGAN(input_dim, hidden, noise_dim).to(device)

    ae_params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    sup_params = list(model.supervisor.parameters())
    gen_params = list(model.generator.parameters())
    disc_params = list(model.discriminator.parameters())

    optim_ae = torch.optim.Adam(ae_params, lr=tg_config.get("learning_rate", 1e-3))
    optim_sup = torch.optim.Adam(sup_params, lr=tg_config.get("learning_rate", 1e-3))
    optim_G = torch.optim.Adam(gen_params + sup_params, lr=tg_config.get("learning_rate", 1e-3))
    optim_D = torch.optim.Adam(disc_params, lr=tg_config.get("learning_rate", 1e-3))

    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if train_gan:
        pretrain_ae = tg_config.get("pretrain_ae_epochs", 5)
        pretrain_sup = tg_config.get("pretrain_sup_epochs", 5)

        logger.info("Starting autoencoder pretraining")
        for epoch in range(pretrain_ae):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(device)
                epoch_loss += train_embedder(model, x, optim_ae)
            logger.info(f"AE pretrain {epoch + 1}/{pretrain_ae} - loss={epoch_loss / len(dataloader):.4f}")

        logger.info("Starting supervisor pretraining")
        for epoch in range(pretrain_sup):
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(device)
                epoch_loss += train_supervisor(model, x, optim_sup)
            logger.info(f"Supervisor pretrain {epoch + 1}/{pretrain_sup} - loss={epoch_loss / len(dataloader):.4f}")

        logger.info("Starting joint TimeGAN training")
        for epoch in range(gan_epochs):
            epoch_g = 0.0
            epoch_d = 0.0
            for batch in dataloader:
                x = batch[0].to(device)
                z = torch.randn(x.shape[0], x.shape[1], noise_dim, device=device)
                loss_g, loss_d = train_joint(model, x, z, optim_G, optim_D)
                epoch_g += loss_g
                epoch_d += loss_d

            logger.info(
                f"Joint train {epoch + 1}/{gan_epochs} - G_loss={epoch_g / len(dataloader):.4f}, D_loss={epoch_d / len(dataloader):.4f}"
            )

    num_aug = int(X.shape[0] * augment_factor)
    if num_aug <= 0:
        logger.info("augment_factor <= 0, skipping augmentation generation")
        return windows

    logger.info(f"Generating {num_aug} augmented windows from the trained GAN")
    model.eval()
    augmented = []
    with torch.no_grad():
        produced = 0
        while produced < num_aug:
            b = min(batch_size, num_aug - produced)
            x_hat = generate(model, b, seq_len, noise_dim, device=device)
            augmented.append(x_hat.cpu().numpy())
            produced += b

    augmented = np.concatenate(augmented, axis=0)

    output_dir = Path(config["paths"]["data_synthetic"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / output_name, augmented)
    logger.info(f"Saved augmented data to {output_dir / output_name}")

    return windows, augmented


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True, help="Path to user-provided .npy or .csv time series")
    parser.add_argument("--output-name", default="augmented_timeseries.npy", help="Filename for augmented output")
    parser.add_argument("--skip-training", action="store_true", help="Skip GAN training and only generate from current weights")
    parser.add_argument("--gan-epochs", type=int, default=10)
    parser.add_argument("--augment-factor", type=float, default=1.0, help="Augmented windows per training window ratio")
    args = parser.parse_args()

    config = load_digital_twin_config()
    augment_timeseries_from_input(
        config,
        input_path=args.input_data,
        output_name=args.output_name,
        train_gan=not args.skip_training,
        gan_epochs=args.gan_epochs,
        augment_factor=args.augment_factor,
    )
