import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.tsa.stattools import adfuller, acf
from tabulate import tabulate

from pricer_path import generate_pricer_paths, load_historical_data_from_csv
load_historical_data = load_historical_data_from_csv

def evaluate_synthetic_vs_real(
    pricer_name: str,
    start_date: str,
    end_date: str,
    seq_len: int = 100,
    n_paths: int = 1000,
    batch_size: int = None
) -> pd.DataFrame:
    """
    Compare des trajectoires "réelles" extraites de la fenêtre [start_date, end_date]
    et des trajectoires synthétiques générées par generate_pricer_paths.
    Si batch_size est donné, on échantillonne aléatoirement batch_size fenêtres réelles.

    Retourne un DataFrame résumé des métriques.
    """
    # 1) Charger et normaliser la série historique
    scaler = MinMaxScaler()
    df = load_historical_data()  # colonnes ['Date','Close']
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    prices = df["Close"].values.astype(np.float32)
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # 2) Construire toutes les fenêtres réelles de longueur seq_len
    real_windows = [
        scaled[i : i + seq_len]
        for i in range(len(scaled) - seq_len + 1)
    ]
    # 2b) Échantillonnage si batch_size spécifié
    if batch_size is not None and batch_size < len(real_windows):
        import random
        real_windows = random.sample(real_windows, batch_size)

    real_flat = np.concatenate(real_windows)

    # 3) Générer trajectoires synthétiques et normaliser avec le même scaler
    synth_paths = generate_pricer_paths(
        pricer_name, start_date, end_date, seq_len, n_paths
    )  # shape = (n_paths, seq_len)
    synth_flat = scaler.transform(
        synth_paths.flatten().reshape(-1, 1)
    ).flatten()

    # 4) Calcul des métriques statistiques
    ks_p    = ks_2samp(real_flat, synth_flat).pvalue
    wass    = wasserstein_distance(real_flat, synth_flat)
    adf_real  = adfuller(np.diff(real_flat))[1]
    adf_synth = adfuller(np.diff(synth_flat))[1]

    # 5) Affichage texte
    print(f"\n[Statistical evaluation for {pricer_name}]")
    summary_df = pd.DataFrame({
        "Metric": ["KS p-value", "Wasserstein", "ADF p-val (real)", "ADF p-val (synth)"],
        "Value":  [ks_p, wass, adf_real, adf_synth]
    })
    print(summary_df)

    # 6) Comparaison des rendements (différences)
    diff_real  = np.diff(real_flat)
    diff_synth = np.diff(synth_flat)
    comparaison = [
        ["Mean",   np.mean(diff_real),  np.mean(diff_synth)],
        ["Std Dev",np.std(diff_real),   np.std(diff_synth)],
        ["Min",    np.min(diff_real),   np.min(diff_synth)],
        ["Max",    np.max(diff_real),   np.max(diff_synth)],
    ]

    print("\n📊 Résumé statistique global :")
    print(tabulate(
        summary_df.values, headers=summary_df.columns, floatfmt=".6f"
    ))

    print("\n📈 Comparaison des rendements normalisés (real vs synth):")
    print(tabulate(
        comparaison, headers=["Stat", "Real", "Synthetic"], floatfmt=".6f"
    ))

    return summary_df
