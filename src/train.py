"""src.train.

Modulo dedicato all'addestramento, all'estrazione di feature e all'inferenza
dei modelli PyTorch per il dataset GTSRB.

Componenti inclusi:
    - AverageValueMeter: Accumulatore ponderato per il tracciamento di loss e accuratezza.
    - train_cnn: Loop di training e validation con supporto TensorBoard e checkpointing.
"""

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class AverageValueMeter:
    """Accumulatore per calcolare la media esatta (ponderata sul numero di campioni)
    di metriche e loss durante l'epoca.
    
    Pattern consolidato visto nei LAB 4 e 5.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.num = 0

    def add(self, value: float, num: int):
        self.sum += value * num
        self.num += num

    def value(self) -> float:
        try:
            return self.sum / self.num
        except ZeroDivisionError:
            return 0.0

def train_cnn(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    exp_name: str = "custom_cnn",
    logdir: str = "../results/logs",
    checkpoint_dir: str = "../results/checkpoints",
    device: torch.device = None,
) -> torch.nn.Module:
    """Esegue il ciclo di addestramento e validazione con logging su TensorBoard
    e salvataggio del miglior checkpoint in base alla validation accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--> Dispositivo di allenamento: {device}")
    model = model.to(device)

    # Creazione directory e inizializzazione TensorBoard
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(logdir, exp_name))

    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    loaders = {"train": train_loader, "val": val_loader}
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n=== Epoca {epoch + 1}/{epochs} ===")

        for mode in ["train", "val"]:
            loss_meter.reset()
            acc_meter.reset()

            if mode == "train":
                model.train()
            else:
                model.eval()

            pbar = tqdm(loaders[mode], desc=f"[{mode.upper()}]", leave=False)

            with torch.set_grad_enabled(mode == "train"):
                for batch in pbar:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    batch_size = images.size(0)

                    # 1. Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # 2. Backward pass & step di ottimizzazione
                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # 3. Accuratezza del batch
                    preds = outputs.argmax(dim=1)
                    acc = (preds == labels).float().mean().item()

                    # 4. Aggiornamento meter ponderati
                    loss_meter.add(loss.item(), batch_size)
                    acc_meter.add(acc, batch_size)

                    pbar.set_postfix({
                        "loss": f"{loss_meter.value():.4f}",
                        "acc": f"{acc_meter.value():.4f}",
                    })

            epoch_loss = loss_meter.value()
            epoch_acc = acc_meter.value()

            print(f"[{mode.upper()}] Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc * 100:.2f}%")

            # 5. Logging scalari su TensorBoard
            writer.add_scalar(f"loss/{mode}", epoch_loss, epoch + 1)
            writer.add_scalar(f"accuracy/{mode}", epoch_acc, epoch + 1)

            # 6. Checkpointing: salviamo il modello con la miglior validation accuracy
            if mode == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                checkpoint_path = os.path.join(checkpoint_dir, f"{exp_name}_best.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  --> Salvato nuovo miglior checkpoint in: {checkpoint_path} (Val Acc: {best_val_acc * 100:.2f}%)")

    writer.close()
    print(f"\n--> Training completato! Miglior Val Accuracy: {best_val_acc * 100:.2f}%")
    return model
