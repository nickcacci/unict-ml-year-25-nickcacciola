"""main.py.

Punto di ingresso principale da riga di comando (CLI) per il progetto GTSRB.

Permette di:
    - Valutare i modelli pre-addestrati sul Test Set e rigenerare metriche e grafici (--mode eval).
    - Addestrare la rete neurale MiniAlexNet da zero con checkpointing e log (--mode train).
    - Avviare la web app dimostrativa Streamlit per il testing interattivo (--mode demo).

Esempi di utilizzo:
    python main.py --mode eval
    python main.py --mode train --epochs 10 --batch_size 64
    python main.py --mode demo
"""

import argparse
import os
import subprocess
import sys

# Forza encoding UTF-8 su terminali Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.dataset import CustomGTSRB, download_and_extract_dataset
from src.metrics import calculate_metrics
from src.models import MiniAlexNet, cnn_inference
from src.train import train_cnn
from src.utils import GTSRB_CLASSES, verify_torch_gpu


def get_transforms(size=(48, 48)):
    """Restituisce le trasformazioni standard per immagini GTSRB."""
    return v2.Compose([
        v2.Resize(size=size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def evaluate_pipeline(data_dir: str, checkpoint_path: str, batch_size: int, device: torch.device):
    """Esegue la valutazione completa di MiniAlexNet sul Test Set,

    stampa la tabella delle metriche e salva i grafici in media/ e results/.
    """
    print("\n=======================================================")
    print("[GTSRB] VALUTAZIONE MODELLO (TEST SET)")
    print("=======================================================\n")

    test_csv_path = os.path.join(data_dir, "Test.csv")
    if not os.path.exists(test_csv_path):
        print(f"[ERRORE] File di annotazione '{test_csv_path}' non trovato!")
        print("   Assicurati di aver scaricato ed estratto il dataset nella cartella 'data/'.")
        return

    if not os.path.exists(checkpoint_path):
        print(f"[ERRORE] Checkpoint '{checkpoint_path}' non trovato!")
        print("   Esegui prima il training con: python main.py --mode train")
        return

    print(f"--> Caricamento Test Set da: {test_csv_path}")
    transform = get_transforms(size=(48, 48))
    ds_test = CustomGTSRB.from_csv(data_dir, "Test.csv", transform=transform)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"--> Numero campioni di test: {len(ds_test)}")

    print(f"--> Inizializzazione MiniAlexNet su dispositivo: {device}")
    model = MiniAlexNet()

    # Esecuzione inferenza
    y_true, y_pred, y_probs = cnn_inference(
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Calcolo metriche
    metrics = calculate_metrics(y_true, y_pred, y_probs, model_name="MiniAlexNet")
    df_metrics = pd.DataFrame([metrics])

    print("\n" + "=" * 60)
    print("[METRICHE] FINALI DI VALUTAZIONE (TEST SET)")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.2f}%")
        else:
            print(f"  {k:<30}: {v}")
    print("=" * 60 + "\n")

    # Salvataggio CSV risultati
    os.makedirs("results", exist_ok=True)
    csv_results_path = os.path.join("results", "final_model_comparison.csv")
    df_metrics.to_csv(csv_results_path, index=False)
    print(f"[SALVATAGGIO] Metriche salvate in: {csv_results_path}")

    # Generazione e salvataggio della Confusion Matrix in media/
    os.makedirs("media", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matrice di Confusione Normalizzata - MiniAlexNet (Test Set)", fontsize=14, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(GTSRB_CLASSES))
    plt.xticks(tick_marks, [f"{i}" for i in range(len(GTSRB_CLASSES))], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [f"{i}" for i in range(len(GTSRB_CLASSES))], fontsize=8)
    plt.xlabel("Classe Predetta", fontsize=11, fontweight="bold")
    plt.ylabel("Classe Reale (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join("media", "confusion_matrix_minialexnet.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[PLOT] Matrice di confusione salvata in: {cm_path}")

    print("\n[OK] Valutazione completata con successo!\n")


def train_pipeline(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    exp_name: str,
):
    """Esegue l'addestramento da zero di MiniAlexNet con train/val split stratificato."""
    print("\n=======================================================")
    print("[GTSRB] ADDESTRAMENTO MINIALEXNET")
    print("=======================================================\n")

    train_csv_path = os.path.join(data_dir, "Train.csv")
    if not os.path.exists(train_csv_path):
        print(f"[AVVISO] File '{train_csv_path}' non trovato.")
        print("   Avvio download ed estrazione automatica del dataset GTSRB...")
        os.makedirs(data_dir, exist_ok=True)
        download_and_extract_dataset(data_dir)

    print(f"--> Lettura dataset da: {train_csv_path}")
    df_train_full = pd.read_csv(train_csv_path)

    # Split 80% train - 20% validation stratificato per classe
    print("--> Creazione dello split Train/Validation (80/20 stratificato, seed=42)...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=42,
        stratify=df_train_full["ClassId"],
    )

    print(f"    Campioni Training:   {len(df_train)}")
    print(f"    Campioni Validation: {len(df_val)}")

    transform = get_transforms(size=(48, 48))
    ds_train = CustomGTSRB(ann_df=df_train, img_dir=data_dir, transform=transform)
    ds_val = CustomGTSRB(ann_df=df_val, img_dir=data_dir, transform=transform)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MiniAlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = os.path.join("results", "checkpoints")
    log_dir = os.path.join("results", "logs")

    print(f"--> Inizio training ({epochs} epoche, lr={lr}, batch_size={batch_size})...")
    trained_model = train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        exp_name=exp_name,
        logdir=log_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # Valutazione finale automatica al termine del training
    best_checkpoint = os.path.join(checkpoint_dir, f"{exp_name}_best.pth")
    print("\n--> Avvio valutazione automatica sul Test Set con il miglior checkpoint salvato...")
    evaluate_pipeline(data_dir, best_checkpoint, batch_size, device)


def run_demo():
    """Avvia la web application dimostrativa in Streamlit."""
    print("\n[DEMO] Avvio della Web App Streamlit (demo.py)...")
    cmd = [sys.executable, "-m", "streamlit", "run", "demo.py"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Web App arrestata.")


def main():
    parser = argparse.ArgumentParser(
        description="CLI per la classificazione di segnali stradali tedeschi (GTSRB) con PyTorch."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "train", "demo"],
        default="eval",
        help="Modalità operativa: 'eval' (valuta sul test set), 'train' (addestra la CNN), 'demo' (avvia web app).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Percorso alla cartella contenente i dati GTSRB (default: 'data').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("results", "checkpoints", "custom_cnn_best.pth"),
        help="Percorso al file pesi (.pth) per la valutazione (default: 'results/checkpoints/custom_cnn_best.pth').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero di epoche per l'addestramento (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Dimensione del batch (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate per l'ottimizzatore Adam (default: 0.001).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="custom_cnn",
        help="Nome dell'esperimento per checkpoint e log TensorBoard (default: 'custom_cnn').",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Dispositivo hardware da utilizzare (default: auto-detect CUDA/CPU).",
    )

    args = parser.parse_args()

    # Selezione del dispositivo
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "eval":
        evaluate_pipeline(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=device,
        )
    elif args.mode == "train":
        train_pipeline(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            exp_name=args.exp_name,
        )
    elif args.mode == "demo":
        run_demo()


if __name__ == "__main__":
    main()
