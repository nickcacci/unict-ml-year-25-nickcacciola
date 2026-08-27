import zipfile
from torch.utils.data import Dataset
from torchvision.io import decode_image
import requests
import os
import pandas as pd
import random
import matplotlib.pyplot as plt


def extract_dataset(data_dir="data"):
    
    possible_zips = [
        os.path.join(data_dir, "archive.zip"),
        os.path.join(data_dir, "gtsrb-german-traffic-sign.zip")
    ]

    # Cerca se uno dei file zip esiste nella cartella data/
    zip_path = next((p for p in possible_zips if os.path.exists(p)), None)

    if zip_path:
        print(f"Estrazione di '{zip_path}' in corso...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Estrazione completata con successo!")
        
        # Rimozione del file zip dopo l'estrazione riuscita
        os.remove(zip_path)
        print(f"File zip '{zip_path}' rimosso con successo per liberare spazio su disco!")
    elif os.path.exists(os.path.join(data_dir, "Train.csv")):
        print("I dati risultano già estratti nella cartella data/!")
    else:
        print(f"Nessun file .zip o dati trovati in {os.path.abspath(data_dir)}.")

def download_and_extract_dataset(data_dir="data"):
    response = requests.get('https://www.kaggle.com/api/v1/datasets/download/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign')
    with open(os.path.join(data_dir,"gtsrb-german-traffic-sign.zip"), 'wb') as f:
        f.write(response.content)
    extract_dataset(data_dir)

def show_random_sample (ds: Dataset):
    """
    Funzione di test per l'implementazione custom della classe Dataset.
    """
    random_idx = random.randint(0, len(ds) - 1)
    sample = ds[random_idx]

    img_tensor = sample["image"]  # Tensor [C, H, W]
    label = sample["label"]
    """ bbox = sample["bbox"]  # [Roi.X1, Roi.Y1, Roi.X2, Roi.Y2] """

    # 3. Carica l'immagine di riferimento corrispondente dalla cartella Meta/
    meta_img_path = os.path.join(ds.dir, "Meta", f"{label}.png")
    meta_img_tensor = decode_image(meta_img_path)

    # 4. Prepariamo i tensori per Matplotlib ([C, H, W] -> [H, W, C])
    img_display = img_tensor.permute(1, 2, 0).numpy()
    meta_display = meta_img_tensor.permute(1, 2, 0).numpy()

    # 5. Stampa le informazioni nel terminale
    print(f"=== Campione Casuale Estratto ===")
    print(f"Indice:       {random_idx}")
    print(f"Classe ID:    {label}")
    print(f"Risoluzione:  {img_tensor.shape[2]}x{img_tensor.shape[1]} pixel")
    """ print(
        f"Bounding Box: X1={bbox[0]:.0f}, Y1={bbox[1]:.0f}, X2={bbox[2]:.0f}, Y2={bbox[3]:.0f}"
    ) """

    # 6. Visualizzazione Side-by-Side con 2 Subplot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # --- Subplot 1: Immagine reale del Dataset con Bounding Box ---
    ax[0].imshow(img_display)
    """x1, y1, x2, y2 = bbox.numpy()
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
        label="ROI Bounding Box",
    )
    ax[0].add_patch(rect) """
    ax[0].set_title(
        f"Campione Dataset #{random_idx}\n({img_tensor.shape[2]}x{img_tensor.shape[1]} px)"
    )
    ax[0].legend(loc="upper right")
    ax[0].axis("on")

    # --- Subplot 2: Cartello di Riferimento (Meta) ---
    ax[1].imshow(meta_display)
    ax[1].set_title(f"Segnale Ufficiale (Meta)\nClasse: {label}")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

class CustomGTSRB(Dataset):
    def __init__(self, ann_df: pd.DataFrame, img_dir, transform=None, target_transform=None):
        self.ann_df= ann_df
        self.dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
    
    @classmethod
    def from_csv (cls, dir, annotations_file, transform=None, target_transform=None):
        ann_df=pd.read_csv(os.path.join(dir,annotations_file))
        return cls(ann_df,img_dir=dir, transform=transform, target_transform=target_transform)


    def __len__(self):
        return len(self.ann_df)
    
    def __getitem__(self, idx):
        """
            The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
        """  
        row=self.ann_df.iloc[idx]
        img_path = os.path.join(self.dir, row["Path"]) 
        label = int (row["ClassId"])
        image = decode_image(img_path)


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)


        """ bbox = torch.tensor(
            [row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]],
            dtype=torch.float32,
        )
        return {"image": image, "label": label, "bbox": bbox} """

        return {"image": image, "label": label}