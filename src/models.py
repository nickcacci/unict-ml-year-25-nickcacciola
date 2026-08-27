"""Modulo per la definizione e l'inizializzazione delle architetture di Deep Learning
utilizzate nel progetto GTSRB (German Traffic Sign Recognition Benchmark).

Modelli inclusi:
    - MiniAlexNet: Rete neurale convoluzionale (CNN) custom a 5 blocchi conv
                   e testa densa a 3 layer FC, ottimizzata per input 48x48 e 43 classi.
    - get_resnet18_feature_extractor: Spina dorsale convoluzionale di ResNet-18 pre-addestrata
                                      su ImageNet per l'estrazione di feature embeddings (512-D).
"""

import torch
from torch import nn
from torchvision import models
H=48
W=48
K=43

class MiniAlexNet (nn.Module):
    def __init__(self, input_channels=3, out_channels=K):
        super ().__init__()
        self.feature_extractor = nn.Sequential (

            #Conv1
            nn.Conv2d(input_channels, out_channels=16, kernel_size=5, padding= 2), #Input 3 x H x W Output: 16 x H x W
            nn.MaxPool2d(2), #Input 16 x H x W Output 16 x H/2 x W/2
            nn.ReLU(),

            #Conv2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding= 2), #Input 16 x H/2 x W/2 Output: 32 x H/2 x W/2
            nn.MaxPool2d(2), #Input 32 x H/2 x W/2 Output 32 x H/4 x W/4
            nn.ReLU(),

            #Conv3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # Input 32 x H/4 x W/4 Output 64 x H/4 X W/4
            nn.ReLU(),

            #Conv4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), #Input 64 x H/4 X W/4 Output 128 x H/4 x W/4
            nn.ReLU(),
            
            #Conv5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), #Input 128 x H/4 x W/4 Output 256 x H/4 x W/4
            nn.MaxPool2d(2),#Input 256 x H/4 x W/4 Output 256 x H/8 x H/8
            nn.ReLU()
        )
        self.classifier = nn.Sequential (
            #FC6
            nn.Linear(256*(H//8)**2, 2048),
            nn.ReLU(),

            #FC7
            nn.Linear(2048,1024),
            nn.ReLU(),

            #FC8
            nn.Linear(1024, K),
            # Nell'esempio del Prof vi è una RELU alla fine, ma la softmax si aspetta i logits puri
            # quindi l'ho commentata
            #nn.ReLU(),

        )

    def forward(self, x):
        x= self.feature_extractor(x)
        # I layer lineari (nn.Linear) accettano solo matrici 2D del tipo [BatchSize, in_features]
        # x adesso è un tensore 4D [BatchSize, 266, 6, 6]
        # quindi x.shape[0] = BatchSize
        # La funzione .view() in PyTorch serve a cambiare la forma (shape) di un tensore senza spostare o copiare i dati in memoria.
        # Per capirla, immagina la memoria del computer: Tutti i numeri di un tensore sono in realtà memorizzati in RAM come un'unica lunga fila ordinata di numeri (1D).
        #  La funzione .view() non tocca quella fila: cambia solo il "modo" (le "lenti") con cui guardi quella sequenza (ad esempio come una griglia 2D, un cubo 3D o un ipercubo 4D).
        # prende come parametri le nuove dimensioni
        # -1: è un jolly (calcolo automatico). Però si può mettere solo una volta per invocazione.
        # Dice a PyTorch: "Moltiplica tutte le restanti dimensioni tra loro e calcola tu la lunghezza della riga".
        # x= self.classifier (x.view(x.shape[0],-1))
        #view() è rigido: funziona solo se il tensore è contiguo in memoria. Se prima hai fatto un'operazione come .permute() o .transpose(), view() fallisce con un RuntimeError.
        #flatten() è sicuro: se il tensore non è contiguo, se ne occupa lui in automatico (fa una copia contigua internamente) senza mai bloccare l'esecuzione.
        x= self.classifier (x.flatten(1))
        return x
        
def get_resnet18_feature_extractor(device=None):
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  weights = models.ResNet18_Weights.DEFAULT
  resnet18 = models.resnet18(weights=weights)
  # Rimuove l'ultimo layer fully connected (fc)
  feature_extractor = nn.Sequential(*list(resnet18.children())[:-1]).to(device)
  feature_extractor.eval()
  return feature_extractor

def cnn_inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    checkpoint_path: str = None,
    device: torch.device = None,
):
    """Valuta il modello sul test set estraendo predizioni e probabilità Softmax.

    Returns:
        tuple: (y_true, y_preds, y_probs)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None:
        print(f"--> Caricamento pesi del miglior modello da: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))

    model.to(device)
    model.eval()

    probs_list = []
    all_preds = []
    all_labels = []

    print(f"--> Valutazione in corso sul Test Set ({device})...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[TEST]", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"]

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())

            preds = logits.argmax(dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = torch.cat(probs_list, dim=0).numpy()

    return all_labels, all_preds, all_probs

def extract_resnet_features(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
):
    """Estrae i vettori di feature da un DataLoader usando una spina dorsale convoluzionale.

    Returns:
        tuple: (X, y) come matrici NumPy pronte per scikit-learn.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    X_list = []
    y_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Estrazione feature", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"]

            feats = model(images)
            feats = torch.flatten(feats, 1)

            X_list.append(feats.cpu().numpy())
            y_list.append(labels.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y
