"""src.metrics.

Modulo dedicato al calcolo, alla visualizzazione e all'analisi comparativa
delle metriche di valutazione per il dataset GTSRB (classificazione multiclasse a 43 classi).

Funzioni incluse:
    - imbalance_stats: Calcola Imbalance Ratio, Entropia Normalizzata e Degree of Imbalance.
    - calculate_metrics: Calcola Accuracy, Macro/Weighted F1, Macro Precision/Recall e Top-k Accuracy.
    - plot_metrics: Mostra la tabella riassuntiva e genera il grafico a barre comparativo con Seaborn.
"""

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
import seaborn as sns


def imbalance_stats(freq_df: pd.Series, num_classes: int = 43):
  """Calcola le metriche di sbilanciamento del dataset:

  - Imbalance Ratio (IR)
  - Entropia Normalizzata (H_norm)
  - Degree of Imbalance (D = 1 - H_norm)
  """
  ir = freq_df.values.max() / freq_df.values.min()
  total_samples = freq_df.values.sum()

  # Entropia di Shannon
  dsum = 0.0
  for count in freq_df:
    p_i = count / total_samples
    dsum += p_i * np.log2(p_i)

  h = -1.0 * dsum
  h_max = np.log2(num_classes)
  h_norm = h / h_max
  d = 1.0 - h_norm

  df_summary = pd.DataFrame({
      "Metrica": [
          "Imbalance Ratio (IR)",
          "Entropia Normalizzata (H)",
          "Degree of Imbalance (D)",
      ],
      "Valore Calcolato": [f"{ir:.2f}", f"{h_norm:.4f}", f"{d:.4f}"],
      "Riferimento (Bilanciato)": ["1.00", "1.00", "0.00"],
  })
  display(df_summary)
  return df_summary


def calculate_metrics(
    y_true, y_pred, y_probs=None, model_name: str = "Modello", num_classes=43
):
  """Calcola in modo standardizzato le metriche per la classificazione multiclasse:

  - Accuracy
  - Macro Precision (PPV)
  - Macro Recall (TPR)
  - Macro F1 (media non pesata, sensibile allo sbilanciamento)
  - Weighted F1 (media pesata per il supporto di classe)
  - Top-3 e Top-5 Accuracy (se y_probs è fornito)
  """
  classes = np.arange(num_classes)

  metrics = {
      "Modello": model_name,
      "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
      "Macro Precision / PPV (%)": (
          precision_score(y_true, y_pred, average="macro", zero_division=0)
          * 100
      ),
      "Macro Recall / TPR (%)": (
          recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
      ),
      "Macro F1 (%)": (
          f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
      ),
      "Weighted F1 (%)": (
          f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
      ),
      "Top-3 Acc (%)": (
          top_k_accuracy_score(y_true, y_probs, k=3, labels=classes) * 100
          if y_probs is not None
          else None
      ),
      "Top-5 Acc (%)": (
          top_k_accuracy_score(y_true, y_probs, k=5, labels=classes) * 100
          if y_probs is not None
          else None
      ),
  }
  return metrics


def plot_metrics(*metrics_args, save_plot_path=None):
  """Raccoglie i risultati di N modelli, mostra la tabella comparativa

  e genera il grafico a barre con le percentuali su ciascuna barra.

  Ritorna:
      pd.DataFrame con i risultati ordinati e arrotondati.
  """
  # Gestione input flessibile: plot_metrics(m1, m2) oppure plot_metrics([m1, m2])
  if len(metrics_args) == 1 and isinstance(metrics_args[0], (list, tuple)):
    metrics_list = list(metrics_args[0])
  else:
    metrics_list = list(metrics_args)

  if not metrics_list:
    raise ValueError("Fornire almeno un dizionario di metriche da confrontare.")

  df_results = pd.DataFrame(metrics_list)
  numeric_cols = df_results.select_dtypes(include=[np.number]).columns
  df_results[numeric_cols] = df_results[numeric_cols].round(2)

  display(df_results)

  df_plot = pd.melt(
      df_results,
      id_vars=["Modello"],
      var_name="Metrica",
      value_name="Punteggio (%)",
  )

  n_models = len(df_results)
  plt.figure(figsize=(max(8, n_models * 3.2), 5.5))
  sns.set_theme(style="whitegrid")

  ax = sns.barplot(
      data=df_plot,
      x="Modello",
      y="Punteggio (%)",
      hue="Metrica",
      palette="viridis",
  )
  plt.title(
      "Confronto Prestazioni Modelli sul Test Set (GTSRB)",
      fontsize=13,
      fontweight="bold",
      pad=15,
  )
  plt.xlabel("Architettura / Modello", fontsize=11, labelpad=10)
  plt.ylabel("Punteggio (%)", fontsize=11)
  plt.ylim(0, 105)
  plt.legend(title="Metrica", loc="lower right", frameon=True)

  for p in ax.patches:
    height = p.get_height()
    if height is not None and not np.isnan(height) and height > 0:
      ax.annotate(
          f"{height:.1f}%",
          (p.get_x() + p.get_width() / 2.0, height),
          ha="center",
          va="bottom",
          fontsize=9,
          xytext=(0, 3),
          textcoords="offset points",
      )

  plt.tight_layout()

  if save_plot_path:
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    plt.savefig(save_plot_path, dpi=300)
    print(f"--> Grafico salvato in: '{save_plot_path}'")

  plt.show()
  return df_results
