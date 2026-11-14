# main2.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pricer_path import generate_pricer_paths, load_historical_data_from_csv

def parse_args():
    p = argparse.ArgumentParser(description="Classifieur Réel vs Synthétique")
    p.add_argument("--pricers",  type=str, required=True, help="Nom du pricer (ex: BlackScholes)")
    p.add_argument("--start",   type=str, required=True, help="Date début (YYYY-MM-DD)")
    p.add_argument("--end",     type=str, required=True, help="Date fin (YYYY-MM-DD)")
    p.add_argument("--seq_len", type=int, default=100,      help="Longueur des fenêtres")
    p.add_argument("--n_real",  type=int, default=5000,     help="Nombre de fenêtres réelles")
    p.add_argument("--n_synth", type=int, default=5000,     help="Nombre de fenêtres synthétiques")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs",     type=int, default=20)
    return p.parse_args()

class CNNClassifier(nn.Module):
    def __init__(self, seq_len, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden*2, 5, padding=2),     nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((seq_len//4)*(hidden*2), 64),         nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def load_real_windows(start, end, seq_len, n_windows):
    df = load_historical_data_from_csv()
    df = df[(df["Date"]>=start)&(df["Date"]<=end)]
    prices = df["Close"].values.astype(np.float32)
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(prices.reshape(-1,1)).flatten()
    windows = []
    for i in range(len(scaled)-seq_len+1):
        windows.append(scaled[i:i+seq_len])
    # sample
    idx = np.random.choice(len(windows), size=n_windows, replace=False)
    return np.stack([windows[i] for i in idx])

def load_synth_windows(pricer, start, end, seq_len, n_paths):
    synth = generate_pricer_paths(pricer, start, end, seq_len, n_paths)
    # les pricer_paths sont en échelle brute, on normalise globalement comme les réels
    flat = synth.flatten().reshape(-1,1)
    scaler = MinMaxScaler((0,1))
    scaler.fit(flat)
    scaled = scaler.transform(flat).reshape(n_paths, seq_len)
    return scaled

def main():
    args = parse_args()
    os.makedirs("results_classifier", exist_ok=True)

    # 1) charger données
    real = load_real_windows(args.start, args.end, args.seq_len, args.n_real)
    synth= load_synth_windows(args.pricer, args.start, args.end, args.seq_len, args.n_synth)

    # 2) construire dataset
    X = np.concatenate([real, synth], axis=0)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))], axis=0)
    # shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # to torch
    X = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,seq_len)
    y = torch.from_numpy(y).float()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 3) modèle, perte, opti
    model = CNNClassifier(args.seq_len).to("cpu")
    crit  = nn.BCEWithLogitsLoss()
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    # 4) entraînement
    for ep in range(1, args.epochs+1):
        total_loss, correct, total = 0., 0, 0
        model.train()
        for xb, yb in dl:
            logits = model(xb)
            loss   = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total   += xb.size(0)
        print(f"[Epoch {ep}/{args.epochs}] Loss={(total_loss/total):.4f}  Acc={(correct/total):.4f}")

    # 5) évaluation finale
    print("\n→ Enregistrement du modèle et des résultats…")
    torch.save(model.state_dict(), f"results_classifier/{args.pricer}_clf.pth")

if __name__ == "__main__":
    main()
