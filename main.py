# main.py

import os
import argparse
import pandas as pd
import torch

import config
from gan import train_gan
from stats_evaluation import evaluate_synthetic_vs_real

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraîne un WGAN-GP et exporte un fichier CSV détaillé."
    )
    p.add_argument("--pricers", nargs="+", required=True,
                   help=f"Pricers disponibles : {config.AVAILABLE_PRICERS}")
    p.add_argument("--start", type=str, required=True,
                   help="Date début (YYYY-MM-DD)")
    p.add_argument("--end", type=str, required=True,
                   help="Date fin (YYYY-MM-DD)")
    p.add_argument("--seq_len", type=int, default=config.DEFAULT_SEQ_LEN,
                   help=f"Longueur des trajectoires (défaut={config.DEFAULT_SEQ_LEN})")
    p.add_argument("--epochs", type=int, default=config.DEFAULT_NUM_EPOCHS,
                   help=f"Époques GAN (défaut={config.DEFAULT_NUM_EPOCHS})")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Répertoire de sortie (défaut=./results)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for pr in args.pricers:
        if pr not in config.AVAILABLE_PRICERS:
            raise ValueError(f"Pricer inconnu : {pr}")

        print(f"\n→ PRICER={pr} | PÉRIODE=[{args.start}–{args.end}] | SEQ_LEN={args.seq_len}")

        # 1) Statistical evaluation
        df_stats = evaluate_synthetic_vs_real(
            pricer_name=pr,
            start_date=args.start,
            end_date=args.end,
            seq_len=args.seq_len,
            n_paths=1000
        )

        # 2) Train GAN
        results = train_gan(
            pricer_name=pr,
            start_date=args.start,
            end_date=args.end,
            seq_len=args.seq_len,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_epochs=args.epochs,
            lr=config.WGAN_LR,
            betas=config.WGAN_BETAS,
            lambda_gp=config.WGAN_LAMBDA_GP
        )
        G_losses   = results["generator_losses"]
        D_losses   = results["critic_losses"]
        final_wass = results["final_wasserstein"]

        # 3) Build losses DataFrame
        df_losses = pd.DataFrame({
            "epoch": list(range(1, args.epochs + 1)),
            "D_loss": D_losses,
            "G_loss": G_losses
        })

        # 4) Extract summary values
        ks_pv      = df_stats.loc[0, "Value"]
        wass_stats = df_stats.loc[1, "Value"]
        adf_real   = df_stats.loc[2, "Value"]
        adf_synth  = df_stats.loc[3, "Value"]

        summary_df = pd.DataFrame({
            "Metric": [
                "KS p-value",
                "Wasserstein (stats)",
                "ADF p-val (real)",
                "ADF p-val (synth)",
                "Wasserstein (GAN final)"
            ],
            "Value": [
                ks_pv,
                wass_stats,
                adf_real,
                adf_synth,
                final_wass
            ]
        })

        # 5) Write a single CSV with all three blocks
        out_file = os.path.join(args.output_dir, f"{pr}_detail_results.csv")
        with open(out_file, "w", newline="") as f:
            f.write("### Summary ###\n")
            summary_df.to_csv(f, index=False)
            f.write("\n### Stats ###\n")
            df_stats.to_csv(f, index=False)
            f.write("\n### Losses ###\n")
            df_losses.to_csv(f, index=False)

        print(f"  ✓ Résultats détaillés sauvegardés dans {out_file}")

    print("\n→ FIN.")

if __name__ == "__main__":
    main()
