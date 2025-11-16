import os
import numpy as np
import pandas as pd
import cv2
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image

# ===============================
# ALAP BEÁLLÍTÁSOK
# ===============================

st.set_page_config(page_title="Lakosság számláló", page_icon="🏠", layout="wide")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

FILE_ID = "1UctmGsjmzKBu74jLou7WaYZ9LoIe-DRt"
MODEL_PATH = "model.h5"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    gdown.download(URL, MODEL_PATH, quiet=False)


# ===============================
# HELYI FÜGGVÉNYEK
# ===============================

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def enhance_with_clahe(image: Image.Image) -> Image.Image:
    img_array = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def normalize(img_array: np.ndarray) -> np.ndarray:
    mean = np.array([0.339, 0.324, 0.285], dtype=np.float32)
    std = np.array([0.139, 0.125, 0.122], dtype=np.float32)
    img_float = img_array.astype(np.float32) / 255.0
    return (img_float - mean) / std

def segment_buildings(mask: np.ndarray, min_size: int = 50):
    binary_mask = (mask > 0.4).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    buildings = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        buildings.append({'area': area, 'bbox': (x, y, w, h), 'centroid': centroids[i]})
    return buildings

def estimate_type(area_m2: float) -> str:
    if area_m2 < 250:
        return 'kis_lakohaz'
    elif area_m2 < 1000:
        return 'kozepes_lakohaz'
    elif area_m2 < 2500:
        return 'nagy_lakohaz'
    else:
        return 'tarsashaz'

def estimate_population(btype: str, area: float) -> float:
    base = {'kis_lakohaz': 2.9, 'kozepes_lakohaz': 3.2, 'nagy_lakohaz': 4.1, 'tarsashaz': 45}.get(btype, 0)
    if btype in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        return round(base * max(1, area / 100), 1)
    elif btype == 'tarsashaz':
        return round(base * max(8, area / 80) / 10, 1)
    return base


# ===============================
# MODELL BETÖLTÉS PATCH-CSEL
# ===============================

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from tensorflow.keras.layers import DepthwiseConv2D

        class PatchedDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, groups=None, **kwargs):
                super().__init__(*args, **kwargs)

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coef': dice_coef,
                'DepthwiseConv2D': PatchedDepthwiseConv2D
            },
            compile=False
        )
        return model

    except Exception as e:
        st.error(f"Modell betöltési hiba: {e}")
        return None


# ===============================
# ROBUSZTUS ANALÍZIS FUNKCIÓ
# ===============================

def analyze(model, image: Image.Image, px_to_m: float = 0.5, debug: bool = False):
    image = enhance_with_clahe(image)
    orig = np.array(image.convert("RGB"))
    h, w = orig.shape[:2]

    # Input shape
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    # Heurisztika
    if len(input_shape) == 4:
        if input_shape[-1] in (1, 3):
            channels_last = True
            target_h, target_w = input_shape[1], input_shape[2]
        else:
            channels_last = False
            target_h, target_w = input_shape[2], input_shape[3]
    else:
        channels_last = True
        target_h, target_w = 256, 256

    resized = cv2.resize(orig, (target_w, target_h))
    norm = normalize(resized)

    if channels_last:
        input_img = norm[None, ...]
    else:
        input_img = np.transpose(norm, (2, 0, 1))[None, ...]

    if debug:
        st.write("DEBUG model.input_shape:", model.input_shape)
        st.write("DEBUG input_img.shape:", input_img.shape)

    pred = model.predict(input_img, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred0 = pred[0]
    else:
        pred0 = pred

    if debug:
        st.write("DEBUG type(pred):", type(pred))
        st.write("DEBUG pred0.shape:", getattr(pred0, "shape", None))

    # Maszk kiválasztás
    if pred0.ndim == 4:
        mask_small = pred0[0, :, :, 0]
    elif pred0.ndim == 3:
        mask_small = pred0[:, :, 0]
    else:
        raise ValueError(f"Nem támogatott predikciós forma: {pred0.shape}")

    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    buildings = segment_buildings(mask)

    results = []
    total_pop = 0.0
    for i, b in enumerate(buildings):
        area_m2 = b['area'] * (px_to_m ** 2)
        btype = estimate_type(area_m2)
        pop = estimate_population(btype, area_m2)
        total_pop += pop
        results.append({'id': i + 1, 'type': btype, 'area_m2': round(area_m2, 1), 'population': pop, 'bbox': b['bbox']})

    return orig, mask, results, total_pop


# ===============================
# STREAMLIT FELÜLET
# ===============================

def main():
    st.title("🏠 Épület Analizátor")
    st.sidebar.header("Beállítások")

    px_to_m = st.sidebar.slider("Pixel → méter", 0.1, 2.0, 0.5, 0.1)
    debug = st.sidebar.checkbox("Debug infó megjelenítése", value=False)

    uploaded = st.file_uploader("Kép feltöltése", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Feltöltött kép")
