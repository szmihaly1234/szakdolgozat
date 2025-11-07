import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2
from PIL import Image
import requests
import os
import time
import pandas as pd
import gdown

file_id = "1UctmGsjmzKBu74jLou7WaYZ9LoIe-DRt"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "model.h5", quiet=False)


st.set_page_config(page_title="Lakosság számláló", page_icon="🏠", layout="wide")

# ===============================
# MODELL BETÖLTÉS
# ===============================

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        file_id = "1UctmGsjmzKBu74jLou7WaYZ9LoIe-DRt"
        destination = "model.h5"
        url = f"https://drive.google.com/uc?id={file_id}"

        if not os.path.exists(destination):
            gdown.download(url, destination, quiet=False)

        def dice_coef(y_true, y_pred):
            smooth = 1.0
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def dice_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        model = tf.keras.models.load_model(
            destination,
            custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef},
            compile=False
        )
        return model

    except Exception as e:
        st.error(f"Modell betöltési hiba: {e}")
        return None


# ===============================
# KÉPFELDOLGOZÁS
# ===============================

def enhance_with_clahe(image):
    img_array = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def normalize(img_array):
    mean = np.array([0.339, 0.324, 0.285])
    std = np.array([0.139, 0.125, 0.122])
    img_float = img_array.astype(np.float32) / 255.0
    return (img_float - mean) / std

# ===============================
# SZEGMENTÁLÁS ÉS ANALÍZIS
# ===============================

def segment_buildings(mask, min_size=50):
    binary_mask = (mask > 0.4).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
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

def estimate_type(area_m2):
    if area_m2 < 250: return 'kis_lakohaz'
    elif area_m2 < 1000: return 'kozepes_lakohaz'
    elif area_m2 < 2500: return 'nagy_lakohaz'
    else: return 'tarsashaz'

def estimate_population(btype, area):
    base = {'kis_lakohaz': 2.9, 'kozepes_lakohaz': 3.2, 'nagy_lakohaz': 4.1, 'tarsashaz': 45}.get(btype, 0)
    if btype in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        return round(base * max(1, area / 100), 1)
    elif btype == 'tarsashaz':
        return round(base * max(8, area / 80) / 10, 1)
    return base

def analyze(model, image, px_to_m=0.5):
    image = enhance_with_clahe(image)
    orig = np.array(image.convert("RGB"))
    h, w = orig.shape[:2]
    resized = cv2.resize(orig, (256, 256))
    input_img = normalize(resized)[None, ...]

    pred, _ = model.predict(input_img, verbose=0)
    mask = cv2.resize(pred[0,:,:,0], (w, h), interpolation=cv2.INTER_NEAREST)
    buildings = segment_buildings(mask)

    results = []
    total_pop = 0
    for i, b in enumerate(buildings):
        area_m2 = b['area'] * (px_to_m ** 2)
        btype = estimate_type(area_m2)
        pop = estimate_population(btype, area_m2)
        total_pop += pop
        results.append({'id': i+1, 'type': btype, 'area_m2': round(area_m2,1), 'population': pop, 'bbox': b['bbox']})

    return orig, mask, results, total_pop

# ===============================
# STREAMLIT FELÜLET
# ===============================

def main():
    st.title("🏠 Épület Analizátor")
    st.sidebar.header("Beállítások")
    px_to_m = st.sidebar.slider("Pixel → méter", 0.1, 2.0, 0.5, 0.1)
    uploaded = st.file_uploader("Kép feltöltése", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Feltöltött kép", use_column_width=True)

        if st.button("Elemzés indítása"):
            model = load_model()
            if model is None:
                st.error("❌ Modell betöltése sikertelen.")
                return

            with st.spinner("Elemzés folyamatban..."):
                orig, mask, buildings, total_pop = analyze(model, image, px_to_m)

            st.subheader("📊 Eredmények")
            st.metric("Épületek száma", len(buildings))
            st.metric("Lakosság becslés", f"{total_pop:.0f} fő")

            st.subheader("🖼️ Szegmentáció")
            overlay = orig.copy()
            overlay[mask > 0.5] = [255, 0, 0]
            result = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
            st.image(result, caption="Szegmentált kép", use_column_width=True)

            st.subheader("📦 Épületek")
            vis = orig.copy()
            for b in buildings:
                x, y, w, h = b['bbox']
                cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
                label = f"{b['type']} ({b['population']} fő)"
                cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            st.image(vis, caption="Detektált épületek", use_column_width=True)

            st.subheader("💾 Export")
            df = pd.DataFrame(buildings)
            st.download_button("CSV letöltése", df.to_csv(index=False), "epulet_adatok.csv", "text/csv")

if __name__ == "__main__":
    main()

