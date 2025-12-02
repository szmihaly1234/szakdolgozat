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
# 1. KONFIGURÁCIÓ ÉS FÁJLOK
# ===============================

# EREDETI MODELL (A szerkezet innen jön)
MODEL_FILE_ID = "19Mw_N1ilU58ipoQ6-BdSbPVtHAlSsn2u"
MODEL_PATH = "model.h5"

# ÚJ SÚLYOK (Párizs/Európa) - A te feltöltött fájlod
WEIGHTS_FILE_ID = "1yMIvlRR6mqKLQ46k9Gh-cvGi83mPJnIB" 
WEIGHTS_PATH = "paris_tuned_weights.weights.h5"

def ensure_file_from_drive(file_id, output_path):
    """Ellenőrzi, hogy megvan-e a fájl, ha nincs, letölti Drive-ról."""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        # quiet=False, hogy lássuk a logot a konzolon, ha baj van
        gdown.download(url, output_path, quiet=False)
    return output_path

# Induláskor az alapmodellt mindenképp letöltjük (ez a "csontváz")
if not os.path.exists(MODEL_PATH):
    print("Alapmodell letöltése...")
    ensure_file_from_drive(MODEL_FILE_ID, MODEL_PATH)

# ===============================
# 2. ALAP BEÁLLÍTÁSOK
# ===============================

st.set_page_config(page_title="Lakosság számláló (Debug)", page_icon="🏠", layout="wide")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# SpaceNet statisztikák a normalizáláshoz
SPACENET_MEAN = np.array([0.339, 0.324, 0.285], dtype=np.float32)
SPACENET_STD  = np.array([0.139, 0.125, 0.122], dtype=np.float32)

# ===============================
# 3. SEGÉDFÜGGVÉNYEK (Matek, Képfeldolgozás)
# ===============================

def meters_per_pixel_web_mercator(latitude_deg: float, zoom: int) -> float:
    R = 6378137.0  # WGS84 Earth radius (m)
    lat_rad = math.radians(latitude_deg)
    return math.cos(lat_rad) * (2 * math.pi * R) / (256 * (2 ** zoom))

def compute_px_to_m_mode(mode: str,
                         manual_px_to_m: float,
                         latitude_deg: float | None,
                         zoom: int | None,
                         known_real_m: float | None,
                         measured_pixels: float | None) -> float:
    if mode == "Auto (Google Maps)":
        if latitude_deg is None or zoom is None:
            return manual_px_to_m
        return meters_per_pixel_web_mercator(latitude_deg, zoom)
    elif mode == "Kalibráció (ismert tárgy)":
        if known_real_m is None or measured_pixels is None or measured_pixels <= 0:
            return manual_px_to_m
        return known_real_m / measured_pixels
    else:
        return manual_px_to_m

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

def spacenet_preprocessing(image_or_array) -> np.ndarray:
    if isinstance(image_or_array, Image.Image):
        img = np.array(image_or_array)
    else:
        img = image_or_array

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_float = img.astype(np.float32) / 255.0
    return (img_float - SPACENET_MEAN) / SPACENET_STD

def pad_to_square_by_longer_side(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img.copy()
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101)
    return padded

def segment_buildings_from_binary(binary_mask: np.ndarray, min_size: int = 50):
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)

    buildings = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        buildings.append({'area': area, 'bbox': (x, y, w, h), 'centroid': centroids[i]})
    return buildings

# ===============================
# 4. OSZTÁLYOZÓ LOGIKA
# ===============================

BUILDING_TYPE_POPULATION = {
    'kis_lakohaz': 2.9,
    'kozepes_lakohaz': 3.2,
    'nagy_lakohaz': 4.1,
    'tarsashaz': 45,
    'kereskedelmi': 0,
    'ipari': 0
}

def estimate_building_type(area_m2: float) -> str:
    if area_m2 < 100:
        return 'kis_lakohaz'
    elif area_m2 < 300:
        return 'kozepes_lakohaz'
    elif area_m2 < 1000:
        return 'nagy_lakohaz'
    else:
        return 'tarsashaz'

def estimate_population(building_type: str, area: float) -> float:
    if building_type not in BUILDING_TYPE_POPULATION:
        return 0
    base_pop = BUILDING_TYPE_POPULATION[building_type]
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        apartments = max(1, area / 100)
        population = base_pop * apartments
    elif building_type == 'tarsashaz':
        apartments = max(8, area / 80)
        population = base_pop * (apartments / 10)
    else:
        population = base_pop
    return round(population, 1)

# ===============================
# 5. MODELL BETÖLTÉS (DEBUG VERZIÓ)
# ===============================

@st.cache_resource(show_spinner=False)
def load_model_with_weights(weights_path=None):
    """
    Betölti a modellt. Visszaadja a modellt ÉS egy státuszüzenetet is.
    """
    try:
        from tensorflow.keras.layers import DepthwiseConv2D

        # Patch a hibás réteghez (Keras verzió inkompatibilitás javítása)
        class PatchedDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, groups=None, **kwargs):
                super().__init__(*args, **kwargs)

        # 1. Alapmodell betöltése (szerkezet)
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coef': dice_coef,
                'DepthwiseConv2D': PatchedDepthwiseConv2D
            },
            compile=False
        )
        
        status_msg = "Alapmodell (Globális) betöltve."
        
        # 2. Ha kértünk súlyokat, megpróbáljuk betölteni
        if weights_path:
            if os.path.exists(weights_path):
                # Ellenőrizzük, nem sérült-e a fájl (túl kicsi fájl gyanús)
                fsize = os.path.getsize(weights_path)
                if fsize < 10000: 
                    return None, f"HIBA: A súlyfájl túl kicsi ({fsize} bájt). Töröld le és próbáld újra!"

                print(f"Súlyok felülírása innen: {weights_path}")
                model.load_weights(weights_path)
                status_msg = f"✅ SIKER: Párizsi súlyok aktívak! (Fájl méret: {fsize/1024/1024:.2f} MB)"
            else:
                return None, f"HIBA: A súlyfájl nem található a lemezen: {weights_path}"
        
        return model, status_msg

    except Exception as e:
        st.error(f"Modell betöltési hiba: {e}")
        return None, str(e)

# ===============================
# 6. INPUT ELŐKÉSZÍTÉS
# ===============================

def make_input_tensor(model, square_rgb: np.ndarray) -> tuple[np.ndarray, bool, tuple[int,int]]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 4:
        if input_shape[-1] in (1, 3):
            channels_last = True
            target_h = input_shape[1] if input_shape[1] else square_rgb.shape[0]
            target_w = input_shape[2] if input_shape[2] else square_rgb.shape[1]
        else:
            channels_last = False
            target_h = input_shape[2] if input_shape[2] else square_rgb.shape[0]
            target_w = input_shape[3] if input_shape[3] else square_rgb.shape[1]
    else:
        channels_last = True
        target_h, target_w = square_rgb.shape[:2]

    resized = cv2.resize(square_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    pre = spacenet_preprocessing(resized)

    if channels_last:
        input_tensor = pre[None, ...]
    else:
        input_tensor = np.transpose(pre, (2, 0, 1))[None, ...]

    return input_tensor, channels_last, (target_h, target_w)

# ===============================
# 7. ANALÍZIS FÜGGVÉNY
# ===============================

def analyze(model, image: Image.Image, px_to_m: float = 0.5, threshold: float = 0.5):
    image = enhance_with_clahe(image)
    orig = np.array(image.convert("RGB"))
    h, w = orig.shape[:2]

    square = pad_to_square_by_longer_side(orig)
    input_tensor, channels_last, _ = make_input_tensor(model, square)

    pred = model.predict(input_tensor, verbose=0)
    pred0 = pred[0] if isinstance(pred, (list, tuple)) else pred

    if pred0.ndim == 4:
        p = pred0[0]
        mask_square = p[..., 0] if channels_last else p[0, ...]
    elif pred0.ndim == 3:
        mask_square = pred0[..., 0] if channels_last else pred0[0, ...]
    else:
        raise ValueError(f"Váratlan predikciós alak: {pred0.shape}")

    mask_square_resized = cv2.resize(mask_square, (square.shape[1], square.shape[0]), interpolation=cv2.INTER_NEAREST)

    size = max(h, w)
    start_y = (size - h) // 2
    start_x = (size - w) // 2
    mask_orig_region = mask_square_resized[start_y:start_y + h, start_x:start_x + w]

    binary_mask = (mask_orig_region > threshold).astype(np.uint8)
    buildings = segment_buildings_from_binary(binary_mask, min_size=50)

    results = []
    total_pop = 0.0
    for i, b in enumerate(buildings):
        area_m2 = b['area'] * (px_to_m ** 2)
        btype = estimate_building_type(area_m2)
        pop = estimate_population(btype, area_m2)
        total_pop += pop
        results.append({
            'id': i + 1,
            'type': btype,
            'area_m2': round(area_m2, 1),
            'population': pop,
            'bbox': b['bbox']
        })

    return orig, mask_orig_region, binary_mask, results, total_pop

# ===============================
# 8. STREAMLIT FELÜLET (MAIN)
# ===============================

def main():
    st.title("Épületek szegmentálása - Transfer Learning Demo 🛠️")
    
    # --- A. MODELL VÁLASZTÓ ---
    st.sidebar.header("1. Modell Verzió")
    model_option = st.sidebar.radio(
        "Melyik tudást használjam?",
        ("Globális (Eredeti)", "Európa/Párizs (Finomhangolt)")
    )
    
    # Logika: Csak akkor töltjük le és állítjuk be a súlyokat, ha a felhasználó kérte
    active_weights_path = None
    
    if model_option == "Európa/Párizs (Finomhangolt)":
        with st.spinner("Európai súlyok letöltése a Drive-ról..."):
            # Itt hívjuk meg a letöltést
            ensure_file_from_drive(WEIGHTS_FILE_ID, WEIGHTS_PATH)
        # Ha letöltöttük, beállítjuk az útvonalat
        active_weights_path = WEIGHTS_PATH
    
    # --- B. MODELL BETÖLTÉS ÉS VISSZAJELZÉS ---
    model, status_msg = load_model_with_weights(active_weights_path)
    
    if model is None:
        st.error("Kritikus hiba: A modell nem töltődött be!")
        st.stop()

    # Állapotjelző csík
    if "SIKER" in status_msg:
        st.success(status_msg)
    else:
        st.info(status_msg)

    # --- C. MÉRET NORMALIZÁLÁS (MPP) ---
    st.sidebar.header("2. Méretarány (MPP)")
    mode = st.sidebar.selectbox("Mód", ["Kézi (slider)", "Auto (Google Maps)", "Kalibráció (ismert tárgy)"])

    manual_px_to_m = st.sidebar.slider("Pixel → méter (kézi)", 0.05, 5.0, 0.5, 0.05)

    if mode == "Auto (Google Maps)":
        latitude_deg = st.sidebar.number_input("Szélesség (°)", value=47.4979, help="Budapest példa: ~47.5")
        zoom = st.sidebar.number_input("Zoom (integer)", min_value=0, max_value=22, value=18, step=1)
    else:
        latitude_deg, zoom = None, None

    if mode == "Kalibráció (ismert tárgy)":
        known_real_m = st.sidebar.number_input("Ismert távolság / tárgy méret (m)", min_value=0.0, value=100.0)
        measured_pixels = st.sidebar.number_input("Képen mért pixel távolság (px)", min_value=0.0, value=200.0)
    else:
        known_real_m, measured_pixels = None, None

    px_to_m = compute_px_to_m_mode(mode, manual_px_to_m, latitude_deg, zoom, known_real_m, measured_pixels)
    st.sidebar.caption(f"Aktív MPP: ~{px_to_m:.4f} m/px")

    threshold = st.sidebar.slider("Maszk threshold (érzékenység)", 0.0, 1.0, 0.5, 0.05)

    # --- D. KÉPFELTÖLTÉS ÉS ELEMZÉS ---
    st.write("---")
    uploaded = st.file_uploader("Kép feltöltése elemzéshez", type=["jpg", "jpeg", "png"])
    
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption=f"Feltöltött kép ({image.size[0]}x{image.size[1]})", use_column_width=True)

        if st.button("Elemzés indítása"):
            with st.spinner("Elemzés folyamatban..."):
                t0 = time.time()
                try:
                    orig, mask_continuous, mask_binary, buildings, total_pop = analyze(model, image, px_to_m, threshold)
                except Exception as e:
                    st.error(f"Hiba az elemzés során: {e}")
                    st.exception(e)
                    st.stop()
                infer_time = time.time() - t0

            # EREDMÉNYEK
            col1, col2, col3 = st.columns(3)
            col1.metric("Épületek száma", len(buildings))
            col2.metric("Lakosság becslés", f"{total_pop:.0f} fő")
            col3.metric("Futási idő", f"{infer_time:.2f} s")

            st.subheader("🖼️ Szegmentáció Eredménye")
            
            # Vizuális visszajelzés: Zöld maszk = Párizs, Piros maszk = Eredeti
            overlay_color = [0, 255, 0] if active_weights_path else [0, 0, 255] # Zöld vs Piros (BGR-ben a cv2 miatt ez Piros vs Kék lehet, de a lényeg, hogy más)
            
            # Matplotlib/RGB konverzió miatt:
            # Ha cv2-t használunk (ami BGR), és RGB képet adunk neki:
            # [0, 255, 0] -> Zöld
            # [255, 0, 0] -> Piros
            
            vis_color = [0, 255, 0] if active_weights_path else [255, 0, 0]
            
            overlay = orig.copy()
            overlay[mask_binary.astype(bool)] = vis_color
            
            result_img = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
            st.image(result_img, caption=f"Szegmentált kép ({'Finomhangolt' if active_weights_path else 'Eredeti'} modell)", use_column_width=True)

            st.subheader("📦 Detektált Épületek")
            vis = orig.copy()
            for b in buildings:
                x, y, w_box, h_box = b['bbox']
                cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                # Felirat
                label = f"{b['type']}"
                cv2.putText(vis, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            st.image(vis, caption="Bounding Boxok", use_column_width=True)

            st.subheader("💾 Adatok Exportálása")
            df = pd.DataFrame(buildings)
            st.download_button("CSV letöltése", df.to_csv(index=False), "epulet_adatok.csv", "text/csv")

if __name__ == "__main__":
    main()
