# gan.py

"""
Ce module implémente :
  • ConvGenerator (1D), ConvCritic (WGAN-GP)
  • gradient_penalty (WGAN-GP)
  • train_gan : entraîne le WGAN-GP sur des segments réels extraits de sp500_price_volume.csv
    vs segments synthétiques générés par generate_pricer_paths_for_ticker
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

import config
from pricer_path import generate_pricer_paths_for_ticker

# ─────────────────────────── ConvGenerator 1D ───────────────────────────
class ConvGenerator(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, hidden: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.hidden = hidden

        # projetter le bruit → (hidden × seq_len//4)
        self.fc = nn.Linear(latent_dim, hidden * (seq_len // 4))

        # upsample ×2 puis ×2
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(hidden, hidden // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, hidden // 2), nn.ReLU(True),

            nn.ConvTranspose1d(hidden // 2, hidden // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, hidden // 4), nn.ReLU(True),
        )

        # raffinage après concaténation
        self.refine = nn.Sequential(
            nn.Conv1d(1 + hidden // 4, hidden // 4, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden // 4), nn.ReLU(True),

            nn.Conv1d(hidden // 4, 1, kernel_size=3, padding=1)
        )

    def forward(self, pricer_paths: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # pricer_paths: (B,1,seq_len), noise: (B, latent_dim)
        B, _, L = pricer_paths.shape

        # 1) projeter + reshape
        x = self.fc(noise).view(B, self.hidden, L // 4)

        # 2) déconv pour remonter vers environ seq_len
        x = self.deconv(x)  # → (B, hidden//4, some_length)

        # 3) ajuster la longueur spatiale à exactement seq_len
        current = x.size(2)
        if current != L:
            diff = L - current
            if diff > 0:
                # pad right with zeros
                x = F.pad(x, (0, diff))
            else:
                # crop to L
                x = x[:, :, :L]

        # 4) concat avec l’entrée et raffiner
        x = torch.cat([pricer_paths, x], dim=1)  # (B, 1+hidden//4, L)
        return self.refine(x)                    # (B, 1, L)


# ─────────────────────────── ConvCritic 1D ───────────────────────────
class Critic(nn.Module):
    def __init__(self, seq_len: int, in_ch: int = 1, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, hidden * 2), nn.LeakyReLU(0.2, True),

            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, hidden * 4), nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear((seq_len // 8) * (hidden * 4), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


# ─────────────────────────── Gradient Penalty ───────────────────────────
def gradient_penalty(D: nn.Module,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     device: torch.device,
                     λ: float = 10.0) -> torch.Tensor:
    B, C, L = real.shape
    alpha = torch.rand(B, 1, 1, device=device)
    x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    y_hat = D(x_hat)
    grads = grad(
        outputs=y_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(y_hat, device=device),
        create_graph=True
    )[0]
    grads = grads.view(B, -1)
    gp = λ * ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ─────────────────────────── BOUCLE D'ENTRAÎNEMENT WGAN-GP ───────────────────────────

# Charger une fois pour toutes PX_LAST date×ticker
df_px_last = (
    pd.read_csv(config.CSV_PATH, sep=';', skiprows=2,
                header=[0, 1], index_col=0, parse_dates=True)
      .xs('PX_LAST', axis=1, level=1)
      .ffill().bfill()
)
tickers = df_px_last.columns.tolist()

def train_gan(pricer_name: str,
              start_date: str,
              end_date: str,
              seq_len: int,
              device: torch.device,
              num_epochs: int,
              lr: float,
              betas: tuple,
              lambda_gp: float) -> dict:
    """
    Entraîne un WGAN-GP sur un segment real vs synthétique à chaque itération.
    """

    # 1) Modèles & optimiseurs
    latent_dim = 100
    G = ConvGenerator(latent_dim, seq_len).to(device)
    D = Critic(seq_len).to(device)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)

    # 2) Scaler global
    all_flat = df_px_last.values.flatten().reshape(-1, 1)
    scaler = MinMaxScaler((0, 1)).fit(all_flat)

    G_losses, D_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # → tirer un ticker et un segment aléatoire
        ticker = random.choice(tickers)
        series = df_px_last[ticker].values
        if len(series) < seq_len:
            continue
        start_idx = random.randint(0, len(series) - seq_len)
        real_seg = series[start_idx : start_idx + seq_len]

        # normaliser + mise en forme
        real_np = scaler.transform(real_seg.reshape(-1,1)).reshape(seq_len)
        real = torch.tensor(real_np, dtype=torch.float32, device=device)
        real = real.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)

        # — Critic updates —
        for _ in range(config.WGAN_N_CRITIC):
            fake_np = generate_pricer_paths_for_ticker(
                pricer_name, ticker,
                start_date, end_date,
                seq_len, n_paths=1
            )[0]
            fake_np = scaler.transform(fake_np.reshape(-1,1)).reshape(seq_len)
            fake = torch.tensor(fake_np, dtype=torch.float32, device=device)
            fake = fake.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)

            D_real = D(real)
            D_fake = D(G(fake, torch.randn(1, latent_dim, device=device)))

            gp = gradient_penalty(D, real, fake, device, λ=lambda_gp)
            loss_D = D_fake.mean() - D_real.mean() + gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # — Generator update —
        fake_np = generate_pricer_paths_for_ticker(
            pricer_name, ticker,
            start_date, end_date,
            seq_len, n_paths=1
        )[0]
        fake_np = scaler.transform(fake_np.reshape(-1,1)).reshape(seq_len)
        fake = torch.tensor(fake_np, dtype=torch.float32, device=device)
        fake = fake.unsqueeze(0).unsqueeze(0)

        z = torch.randn(1, latent_dim, device=device)
        fake_G = G(fake, z)
        loss_G = -D(fake_G).mean()

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
        print(f"[Epoch {epoch}/{num_epochs}] D_loss {loss_D.item():.4f}  G_loss {loss_G.item():.4f}")

    # 3) Wasserstein final
    with torch.no_grad():
        real_final = real
        fake_np = generate_pricer_paths_for_ticker(
            pricer_name, tickers[0],
            start_date, end_date, seq_len, n_paths=1
        )[0]
        fake_np = scaler.transform(fake_np.reshape(-1,1)).reshape(seq_len)
        fake_final = torch.tensor(fake_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        wass = (D(fake_final).mean() - D(real_final).mean()).item()

    return {
        "pricer_name":       pricer_name,
        "start_date":        start_date,
        "end_date":          end_date,
        "seq_len":           seq_len,
        "num_epochs":        num_epochs,
        "generator_losses":  G_losses,
        "critic_losses":     D_losses,
        "final_wasserstein": wass
    }
