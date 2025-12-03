import os
import numpy as np
import pandas as pd
import cv2
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image
import time
import math

# ===============================
# 1. KONFIGURÁCIÓ
# ===============================

MODEL_FILE_ID = "19Mw_N1ilU58ipoQ6-BdSbPVtHAlSsn2u"
MODEL_PATH = "model.h5"

WEIGHTS_FILE_ID = "1yMIvlRR6mqKLQ46k9Gh-cvGi83mPJnIB" 
WEIGHTS_PATH = "paris_tuned_weights.weights.h5"

# ===============================
# 2. ALAP BEÁLLÍTÁSOK
# ===============================

st.set_page_config(page_title="Lakosság számláló (CHECKSUM FIX)", page_icon="🏗️", layout="wide")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SPACENET_MEAN = np.array([0.339, 0.324, 0.285], dtype=np.float32)
SPACENET_STD  = np.array([0.139, 0.125, 0.122], dtype=np.float32)

# ===============================
# 3. SEGÉDFÜGGVÉNYEK
# ===============================

def ensure_file_from_drive(file_id, output_path):
    if os.path.exists(output_path):
        if os.path.getsize(output_path) < 10000:
            os.remove(output_path)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

if not os.path.exists(MODEL_PATH):
    ensure_file_from_drive(MODEL_FILE_ID, MODEL_PATH)

def meters_per_pixel_web_mercator(latitude_deg, zoom):
    R = 6378137.0
    lat_rad = math.radians(latitude_deg)
    return math.cos(lat_rad) * (2 * math.pi * R) / (256 * (2 ** zoom))

def compute_px_to_m_mode(mode, manual, lat, zoom, known_m, meas_px):
    if mode == "Auto (Google Maps)":
        return meters_per_pixel_web_mercator(lat, zoom) if (lat and zoom) else manual
    elif mode == "Kalibráció (ismert tárgy)":
        return known_m / meas_px if (known_m and meas_px) else manual
    return manual

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def enhance_with_clahe(image):
    img_array = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))

def spacenet_preprocessing(image_or_array):
    if isinstance(image_or_array, Image.Image):
        img = np.array(image_or_array)
    else:
        img = image_or_array
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_float = img.astype(np.float32) / 255.0
    return (img_float - SPACENET_MEAN) / SPACENET_STD

def pad_to_square_by_longer_side(img):
    h, w = img.shape[:2]
    if h == w: return img.copy()
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)

def segment_buildings_from_binary(binary_mask, min_size=50):
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    buildings = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size: continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        buildings.append({'area': area, 'bbox': (x, y, w, h), 'centroid': centroids[i]})
    return buildings

# ===============================
# 4. OSZTÁLYOZÓ LOGIKA
# ===============================

BUILDING_TYPE_POPULATION = {
    'kis_lakohaz': 2.9, 'kozepes_lakohaz': 3.2, 'nagy_lakohaz': 4.1,
    'tarsashaz': 45, 'kereskedelmi': 0, 'ipari': 0
}

def estimate_building_type(area_m2):
    if area_m2 < 100: return 'kis_lakohaz'
    elif area_m2 < 300: return 'kozepes_lakohaz'
    elif area_m2 < 1000: return 'nagy_lakohaz'
    else: return 'tarsashaz'

def estimate_population(building_type, area):
    if building_type not in BUILDING_TYPE_POPULATION: return 0
    base_pop = BUILDING_TYPE_POPULATION[building_type]
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        return base_pop * max(1, area / 100)
    elif building_type == 'tarsashaz':
        return base_pop * (max(8, area / 80) / 10)
    return base_pop

# ===============================
# 5. MODELL BETÖLTÉS (JAVÍTOTT CHECKSUM)
# ===============================

@st.cache_resource(show_spinner=False)
def load_model_pro(weights_path=None):
    # Fontos: Session törlése, hogy biztosan tiszta lappal induljunk
    K.clear_session()
    
    try:
        from tensorflow.keras.layers import DepthwiseConv2D

        class FixedDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(**kwargs)

        # 1. Alapmodell
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'dice_loss': dice_loss, 
                'dice_coef': dice_coef, 
                'DepthwiseConv2D': FixedDepthwiseConv2D 
            },
            compile=False
        )
        
        info_msg = "Globális modell"
        
        # 2. Súlyok betöltése
        if weights_path:
            if os.path.exists(weights_path):
                fsize = os.path.getsize(weights_path)
                try:
                    model.load_weights(weights_path)
                    info_msg = f"Párizsi modell"
                except Exception as load_err:
                    return None, f"HIBA: {load_err}", 0
            else:
                return None, f"HIBA: Fájl nem található", 0
        
        # 3. CHECKSUM (Most már az UTOLSÓ 5 réteget nézzük!)
        weights_sum = 0.0
        # A reversed() miatt a kimeneti (Output) rétegekkel kezdjük
        count = 0
        for layer in reversed(model.layers):
            if layer.weights:
                weights = layer.get_weights()
                weights_sum += np.sum([np.sum(np.abs(w)) for w in weights])
                count += 1
                if count >= 5: break # Csak az utolsó 5 réteg elég a különbséghez
                
        return model, info_msg, weights_sum

    except Exception as e:
        return None, f"Kritikus hiba: {str(e)}", 0

# ===============================
# 6. INPUT ELŐKÉSZÍTÉS
# ===============================

def make_input_tensor(model, square_rgb):
    input_shape = model.input_shape
    if isinstance(input_shape, list): input_shape = input_shape[0]
    
    if len(input_shape) == 4:
        if input_shape[-1] in (1, 3):
            channels_last, target_h, target_w = True, input_shape[1], input_shape[2]
        else:
            channels_last, target_h, target_w = False, input_shape[2], input_shape[3]
    else:
        channels_last, target_h, target_w = True, square_rgb.shape[0], square_rgb.shape[1]

    if target_h is None: target_h = 320
    if target_w is None: target_w = 320

    resized = cv2.resize(square_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    pre = spacenet_preprocessing(resized)

    if channels_last: input_tensor = pre[None, ...]
    else: input_tensor = np.transpose(pre, (2, 0, 1))[None, ...]

    return input_tensor, channels_last, (target_h, target_w)

# ===============================
# 7. ANALÍZIS
# ===============================

def analyze(model, image, px_to_m, threshold):
    image = enhance_with_clahe(image)
    orig = np.array(image.convert("RGB"))
    h, w = orig.shape[:2]

    square = pad_to_square_by_longer_side(orig)
    input_tensor, channels_last, _ = make_input_tensor(model, square)

    pred = model.predict(input_tensor, verbose=0)
    pred0 = pred[0] if isinstance(pred, (list, tuple)) else pred

    if pred0.ndim == 4: p = pred0[0]; mask_square = p[..., 0] if channels_last else p[0, ...]
    elif pred0.ndim == 3: mask_square = pred0[..., 0] if channels_last else pred0[0, ...]
    else: raise ValueError(f"Shape hiba: {pred0.shape}")

    mask_square_resized = cv2.resize(mask_square, (square.shape[1], square.shape[0]), interpolation=cv2.INTER_NEAREST)
    size = max(h, w)
    start_y, start_x = (size - h) // 2, (size - w) // 2
    mask_orig_region = mask_square_resized[start_y:start_y + h, start_x:start_x + w]

    binary_mask = (mask_orig_region > threshold).astype(np.uint8)
    buildings = segment_buildings_from_binary(binary_mask, min_size=50)

    results = []
    total_pop = 0
    for i, b in enumerate(buildings):
        area_m2 = b['area'] * (px_to_m ** 2)
        btype = estimate_building_type(area_m2)
        pop = estimate_population(btype, area_m2)
        total_pop += pop
        results.append({'id': i+1, 'type': btype, 'area_m2': round(area_m2,1), 'population': pop, 'bbox': b['bbox']})

    return orig, mask_orig_region, binary_mask, results, total_pop

# ===============================
# 8. MAIN UI
# ===============================

def clear_model_cache():
    st.cache_resource.clear()

def main():
    st.title("Lakosságszámláló - Végleges 🚀")

    # --- SIDEBAR: MODELL ÉS SÚLY INFO ---
    st.sidebar.title("⚙️ Beállítások")
    
    st.sidebar.subheader("1. Modell Verzió")
    
    model_option = st.sidebar.radio(
        "Tudásbázis:",
        ("Globális (Eredeti)", "Európa/Párizs (Finomhangolt)"),
        on_change=clear_model_cache
    )
    
    active_weights_path = None
    if model_option == "Európa/Párizs (Finomhangolt)":
        with st.spinner("Súlyok ellenőrzése..."):
            ensure_file_from_drive(WEIGHTS_FILE_ID, WEIGHTS_PATH)
        active_weights_path = WEIGHTS_PATH

    # --- MODELL BETÖLTÉS ---
    with st.spinner("Modell betöltése memóriába..."):
        model, status_msg, check_sum = load_model_pro(active_weights_path)
    
    if model is None:
        st.error(status_msg)
        st.stop()

    st.sidebar.divider()
    st.sidebar.info(f"Aktív: {status_msg}")
    
    # Checksum megjelenítése (Most már változnia KELL!)
    st.sidebar.metric(
        label="Checksum (Végződés)", 
        value=f"{check_sum:.1f}",
        delta="Más" if active_weights_path else "Eredeti"
    )

    # --- MÉRETARÁNY ---
    st.sidebar.subheader("2. Méretarány")
    mode = st.sidebar.selectbox("Mód", ["Kézi (slider)", "Auto (Google Maps)", "Kalibráció"])
    manual_px_to_m = st.sidebar.slider("Pixel -> Méter", 0.05, 5.0, 0.5, 0.05)
    
    lat, zoom, known_m, meas_px = None, None, None, None
    if mode == "Auto (Google Maps)":
        lat = st.sidebar.number_input("Szélesség", value=47.4979)
        zoom = st.sidebar.number_input("Zoom", value=18)
    if mode == "Kalibráció":
        known_m = st.sidebar.number_input("Távolság (m)", value=100.0)
        meas_px = st.sidebar.number_input("Pixel (px)", value=200.0)

    px_to_m = compute_px_to_m_mode(mode, manual_px_to_m, lat, zoom, known_m, meas_px)
    st.sidebar.caption(f"Aktív MPP: {px_to_m:.4f}")
    
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

    # --- ELEMZÉS ---
    st.write("---")
    uploaded = st.file_uploader("Kép feltöltése", type=["jpg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Feltöltött kép", width=600)
        
        if st.button("Elemzés indítása", type="primary"):
            with st.spinner("Neurális hálózat futtatása..."):
                try:
                    orig, _, mask_binary, buildings, total_pop = analyze(model, image, px_to_m, threshold)
                    
                    # EREDMÉNYEK
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Épületek", len(buildings))
                    c2.metric("Becsült Lakosság", int(total_pop))
                    c3.metric("Modell", "Párizs" if active_weights_path else "Globális")
                    
                    # SZÍNES MASZK
                    overlay = orig.copy()
                    color = [0, 255, 0] if active_weights_path else [255, 0, 0] 
                    overlay[mask_binary.astype(bool)] = color
                    res = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
                    
                    st.image(res, caption=f"Szegmentáció Eredménye", use_column_width=True)
                    
                    # EXPORT
                    df = pd.DataFrame(buildings)
                    st.download_button("Adatok letöltése (CSV)", df.to_csv(), "adatok.csv")

                except Exception as e:
                    st.error(f"Hiba: {e}")

if __name__ == "__main__":
    main()
