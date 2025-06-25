# config.py

import os

# ─────────────────── CHEMIN VERS LE FICHIER EXCEL ───────────────────
# On suppose que le fichier Excel est dans le même dossier que main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILENAME = "sp500_price_volume.csv"
EXCEL_FILENAME2 = "sp500_price_volume_sorted.csv"
EXCEL_PATH = os.path.join(BASE_DIR, EXCEL_FILENAME)
CSV_PATH = EXCEL_PATH
HIST_TICKER = "MMM"
EXCEL_EXIT = os.path.join(BASE_DIR, EXCEL_FILENAME2)
# ─────────────────── HYPERPARAMÈTRES PAR DÉFAUT ───────────────────
# Fenêtre par défaut pour estimer la volatilité/historique (en jours)
DEFAULT_HIST_WINDOW = 252  # ≈ 1 an de bourse

# Nombre de pas (steps) par trajectoire GAN (sera égal à la “durée” demandée)
DEFAULT_SEQ_LEN = 100

# Hyperparamètres WGAN‐GP
WGAN_LR       = 1e-4
WGAN_BETAS    = (0.5, 0.9)
WGAN_LAMBDA_GP= 10.0
WGAN_N_CRITIC = 5

# Nombre de trajectoires “fake” générées par batch
DEFAULT_BATCH_SIZE = 64

# Nombre maximal d’époques pour l’entraînement du GAN
DEFAULT_NUM_EPOCHS = 50

# Seuil de stabilité du GAN : si la perte du Générateur varie de moins de EPS sur
# les N_STEPS dernières itérations, on considère le GAN stabilisé.
STABILITY_EPS   = 1e-3
STABILITY_STEPS = 10

# Liste des pricers disponibles (doivent correspondre aux clés implémentées dans pricer_path.py)
# Liste des pricers disponibles
AVAILABLE_PRICERS = [
    "BlackScholes",
    "Merton",
    "MonteCarlo",
    "BinomialCRR",
    "Trinomial",
    "ExplicitFD",
    "HestonMC",
    "CarrMadan",
    "AntitheticMC",
    "BjerksundStensland"
]