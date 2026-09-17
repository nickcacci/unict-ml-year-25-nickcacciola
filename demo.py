"""demo.py.

Applicazione Web interattiva realizzata con Streamlit per la classificazione
dei segnali stradali tedeschi (GTSRB) tramite la rete convoluzionale MiniAlexNet.

Funzionalità:
    - Caricamento di immagini da file locale (JPG, PNG).
    - Selezione di campioni casuali dal Test Set per test rapido e verifica della Ground Truth.
    - Acquisizione in tempo reale da webcam.
    - Confronto visivo tra l'immagine analizzata e il cartello canonico da catalogo ufficiale.
    - Visualizzazione della classifica delle Top-K previsioni con relative percentuali di confidenza.
"""

import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms


from src.models import MiniAlexNet
from src.utils import GTSRB_CLASSES



st.set_page_config(
    page_title="GTSRB Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pred-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



TRANSFORM = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource(show_spinner="Caricamento checkpoint di MiniAlexNet in corso...")
def get_cached_model(checkpoint_path: str, device_name: str):
    """Carica e memorizza in cache il modello MiniAlexNet."""
    device = torch.device(device_name)
    model = MiniAlexNet()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        st.error(f"Checkpoint non trovato al percorso: `{checkpoint_path}`")

    model.to(device)
    model.eval()
    return model, device


def predict_image(image: Image.Image, model: torch.nn.Module, device: torch.device):
    """Esegue l'inferenza su una PIL Image e restituisce predizioni e probabilità."""
    # Conversione in RGB nel caso sia RGBA o Grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    input_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1]
    return top_indices, probs



with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-light.png", width=64)
    st.title("Pannello di Controllo")

    st.markdown("### ⚙️ Modello & Dispositivo")
    cuda_available = torch.cuda.is_available()
    default_device = "cuda" if cuda_available else "cpu"

    device_option = st.selectbox(
        "Dispositivo hardware:",
        options=["cuda", "cpu"] if cuda_available else ["cpu"],
        index=0,
        help="CUDA sfrutta la GPU dedicata; CPU è compatibile con qualsiasi macchina.",
    )

    checkpoint_default = os.path.join("results", "checkpoints", "custom_cnn_best.pth")
    if not os.path.exists(checkpoint_default):
        checkpoint_default = os.path.join("notebooks", "checkpoints", "custom_cnn_best.pth")

    checkpoint_path = st.text_input(
        "Percorso Checkpoint (.pth):",
        value=checkpoint_default,
        help="Percorso relativo o assoluto ai pesi del modello addestrato.",
    )

    top_k = st.slider("Numero di predizioni Top-K da mostrare:", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("### 🎓 Informazioni Progetto")
    st.markdown(
        """
        - **Corso**: Machine Learning (A.A. 2025/2026)
        - **Ateneo**: Università degli Studi di Catania (DMI)
        - **Autore**: Nicolò Cacciola
        - **Architettura**: MiniAlexNet (CNN 5 blocchi conv, ~21M parametri)
        - **Dataset**: GTSRB (43 classi)
        """
    )



st.markdown('<div class="main-title">🚦 GTSRB Traffic Sign Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Dimostrazione interattiva del sistema di riconoscimento di segnali stradali basato su Deep Learning.</div>',
    unsafe_allow_html=True,
)

# Caricamento modello
model, device = get_cached_model(checkpoint_path, device_option)

# Modalità di input
tab1, tab2, tab3 = st.tabs([
    "📁 Carica Immagine",
    "🎲 Esempio dal Test Set",
    "📸 Webcam in Tempo Reale",
])

input_image = None
ground_truth_class = None
source_info = ""

with tab1:
    uploaded_file = st.file_uploader(
        "Scegli un'immagine da classificare (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        source_info = f"File caricato: **{uploaded_file.name}**"

with tab2:
    st.write("Estrai un'immagine reale dal Test Set di GTSRB per testare il modello e confrontare la Ground Truth.")
    test_csv_path = os.path.join("data", "Test.csv")

    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            sample_button = st.button("🎲 Estrai Campione Casuale", use_container_width=True)

        if sample_button or "sample_idx" in st.session_state:
            if sample_button:
                st.session_state.sample_idx = random.randint(0, len(df_test) - 1)

            idx = st.session_state.sample_idx
            sample_row = df_test.iloc[idx]
            img_rel_path = os.path.join("data", sample_row["Path"])

            if os.path.exists(img_rel_path):
                input_image = Image.open(img_rel_path)
                ground_truth_class = int(sample_row["ClassId"])
                source_info = f"Campione Test #{idx} (`{sample_row['Path']}`)"
            else:
                st.warning(f"File immagine non trovato: `{img_rel_path}`")
    else:
        st.info("File `data/Test.csv` non rilevato. Assicurati che il dataset sia estratto nella cartella `data/`.")

with tab3:
    camera_file = st.camera_input("Inquadra un cartello stradale con la webcam:")
    if camera_file is not None:
        input_image = Image.open(camera_file)
        source_info = "Scatto da Webcam"



if input_image is not None:
    st.markdown("---")
    col_input, col_pred, col_details = st.columns([1.2, 1.5, 1.8], gap="large")

    # Colonna 1: Immagine di input
    with col_input:
        st.markdown("### 🖼️ Immagine di Input")
        st.image(input_image, caption=source_info, use_container_width=True)
        w, h = input_image.size
        st.caption(f"Risoluzione originale: **{w} x {h} px**")

    # Esecuzione inferenza
    top_indices, probs = predict_image(input_image, model, device)
    best_class = int(top_indices[0])
    best_conf = float(probs[best_class])

    # Colonna 2: Predizione Principale (Top-1) e Cartello Ufficiale
    with col_pred:
        st.markdown("### 🎯 Predizione del Modello")
        st.markdown(
            f"""
            <div class="pred-box">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">Classe Prevista #{best_class}</div>
                <div style="font-size: 1.35rem; font-weight: 700; margin: 4px 0 10px 0;">{GTSRB_CLASSES.get(best_class, f'Classe {best_class}')}</div>
                <div style="font-size: 1.1rem;">Confidenza: <b>{best_conf * 100:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Se disponibile la Ground Truth (da Test Set), verifica correttezza
        if ground_truth_class is not None:
            if best_class == ground_truth_class:
                st.success(f"✅ **Classificazione Corretta!** La Ground Truth reale è la Classe {ground_truth_class}.")
            else:
                st.error(
                    f"❌ **Classificazione Errata!** La classe reale era: **#{ground_truth_class} - {GTSRB_CLASSES.get(ground_truth_class, '')}**."
                )

        # Visualizzazione del cartello canonico da catalogo ufficiale Meta/
        canonical_img_path = os.path.join("data", "Meta", f"{best_class}.png")
        if os.path.exists(canonical_img_path):
            st.markdown("##### 📖 Cartello Canonico Ufficiale (Meta):")
            st.image(canonical_img_path, width=120)

    # Colonna 3: Classifica Top-K e Grafico a Barre
    with col_details:
        st.markdown(f"### 📊 Classifica Top-{top_k}")

        chart_data = []
        for i in range(top_k):
            cls_id = int(top_indices[i])
            conf = float(probs[cls_id])
            label = f"#{cls_id} {GTSRB_CLASSES.get(cls_id, f'Classe {cls_id}')}"
            chart_data.append({
                "Classe": label,
                "Probabilità (%)": round(conf * 100, 2),
            })

            # Visualizzazione a barre singole
            col_bar_label, col_bar_val = st.columns([3, 1])
            with col_bar_label:
                st.write(f"**{i+1}.** {label}")
            with col_bar_val:
                st.write(f"**{conf * 100:.2f}%**")
            st.progress(conf)

        # Tabella riassuntiva
        with st.expander("🔍 Mostra dettagli probabilità"):
            df_chart = pd.DataFrame(chart_data)
            st.dataframe(df_chart, use_container_width=True)
else:
    st.info("👆 Seleziona una delle schede in alto per caricare un'immagine o provare un campione casuale dal Test Set.")
