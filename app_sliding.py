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

st.set_page_config(page_title="Lakosság számláló (Smart Scaling)", page_icon="🏗️", layout="wide")
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

def spacenet_preprocessing(img_array):
    if img_array.ndim == 2: img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.ndim == 3 and img_array.shape[2] == 4: img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else: img = img_array
    
    img_float = img.astype(np.float32) / 255.0
    return (img_float - SPACENET_MEAN) / SPACENET_STD

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
# 5. MODELL BETÖLTÉS
# ===============================

@st.cache_resource(show_spinner=False)
def load_model_pro(weights_path=None):
    K.clear_session()
    try:
        from tensorflow.keras.layers import DepthwiseConv2D
        class FixedDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(**kwargs)

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'DepthwiseConv2D': FixedDepthwiseConv2D},
            compile=False
        )
        
        info_msg = "Globális modell"
        if weights_path:
            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                info_msg = f"Párizsi modell"
            else:
                return None, f"HIBA: Fájl nem található", 0
        
        weights_sum = 0.0
        count = 0
        for layer in reversed(model.layers):
            if layer.weights:
                weights = layer.get_weights()
                weights_sum += np.sum([np.sum(np.abs(w)) for w in weights])
                count += 1
                if count >= 5: break
                
        return model, info_msg, weights_sum

    except Exception as e:
        return None, f"Kritikus hiba: {str(e)}", 0

# ===============================
# 6. SLIDING WINDOW LOGIKA (JAVÍTOTT)
# ===============================

def predict_sliding_window(model, image, tile_size=320, overlap=0.25):
    h, w, c = image.shape
    stride = int(tile_size * (1 - overlap))
    
    pad_h = (tile_size - (h % stride)) % stride + (tile_size - stride)
    pad_w = (tile_size - (w % stride)) % stride + (tile_size - stride)
    
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw, _ = padded_image.shape
    
    full_mask = np.zeros((ph, pw), dtype=np.float32)
    count_mask = np.zeros((ph, pw), dtype=np.float32)
    
    total_steps = ((ph - tile_size) // stride + 1) * ((pw - tile_size) // stride + 1)
    progress_bar = st.progress(0)
    step_count = 0
    
    # Batch processing helyett egyenként, de memóriakímélő
    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            tile = padded_image[y:y+tile_size, x:x+tile_size]
            pre = spacenet_preprocessing(tile)
            input_tensor = np.expand_dims(pre, axis=0)
            
            pred = model.predict(input_tensor, verbose=0)
            mask = pred[0, :, :, 0]
            
            full_mask[y:y+tile_size, x:x+tile_size] += mask
            count_mask[y:y+tile_size, x:x+tile_size] += 1
            
            step_count += 1
            if step_count % 10 == 0:
                 progress_bar.progress(min(step_count / total_steps, 1.0))
    
    progress_bar.empty()
    
    avg_mask = np.divide(full_mask, count_mask, out=np.zeros_like(full_mask), where=count_mask!=0)
    final_mask = avg_mask[:h, :w]
    
    return final_mask

# ===============================
# 7. ANALÍZIS + SKÁLÁZÁS
# ===============================

def analyze(model, image, px_to_m, threshold, scale_factor=1.0):
    """
    Scale Factor: Ha 0.5, akkor a képet felére kicsinyítjük elemzés előtt.
    Ez segít, ha az épületek túl nagyok (túl részletes a kép) a modellnek.
    """
    
    # 1. Eredeti betöltése
    image_enhanced = enhance_with_clahe(image)
    orig_np = np.array(image_enhanced.convert("RGB"))
    original_h, original_w = orig_np.shape[:2]

    # 2. Átméretezés (Downscaling) elemzéshez
    # Ha scale_factor < 1.0, akkor kicsinyítünk, hogy a házak "beleférjenek" a modell ablakába
    if scale_factor != 1.0:
        analyze_w = int(original_w * scale_factor)
        analyze_h = int(original_h * scale_factor)
        img_for_analysis = cv2.resize(orig_np, (analyze_w, analyze_h), interpolation=cv2.INTER_AREA)
    else:
        img_for_analysis = orig_np

    # 3. Sliding Window a (esetleg kicsinyített) képen
    raw_mask_small = predict_sliding_window(model, img_for_analysis, tile_size=320, overlap=0.25)

    # 4. Maszk visszanövelése az eredeti méretre
    if scale_factor != 1.0:
        raw_mask = cv2.resize(raw_mask_small, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    else:
        raw_mask = raw_mask_small

    # 5. Bináris maszk és épületkeresés
    binary_mask = (raw_mask > threshold).astype(np.uint8)
    buildings = segment_buildings_from_binary(binary_mask, min_size=50) # min_size pixelben, eredeti képen

    results = []
    total_pop = 0
    
    # MPP korrekció nem kell, mert visszaméreteztük a maszkot az eredeti képre!
    # A px_to_m most az eredeti kép pixelére vonatkozik.
    
    for i, b in enumerate(buildings):
        area_m2 = b['area'] * (px_to_m ** 2)
        btype = estimate_building_type(area_m2)
        pop = estimate_population(btype, area_m2)
        total_pop += pop
        results.append({'id': i+1, 'type': btype, 'area_m2': round(area_m2,1), 'population': pop, 'bbox': b['bbox']})

    return orig_np, raw_mask, binary_mask, results, total_pop

# ===============================
# 8. MAIN UI
# ===============================

def clear_model_cache():
    st.cache_resource.clear()

def main():
    st.title("Lakosságszámláló - Smart Scaling 🧠")

    st.sidebar.title("⚙️ Beállítások")
    
    st.sidebar.subheader("1. Modell")
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

    with st.spinner("Modell betöltése..."):
        model, status_msg, check_sum = load_model_pro(active_weights_path)
    
    if model is None:
        st.error(status_msg)
        st.stop()
    
    st.sidebar.metric("Checksum", f"{check_sum:.1f}")

    # --- MÉRETARÁNY ---
    st.sidebar.subheader("2. Skálázás (FONTOS!)")
    
    st.sidebar.info("Ha túl közeliek/nagyok az épületek a képen, a modell nem ismeri fel őket. Ilyenkor csökkentsd a 'Kép Skálázást'!")
    
    # ÚJ SLIDER:
    scale_factor = st.sidebar.slider("Elemzési Zoom (Kicsinyítés)", 0.1, 1.5, 1.0, 0.1, help="1.0 = Eredeti méret. 0.5 = 50%-ra kicsinyítve fut a modell (jobb nagy házakhoz).")

    st.sidebar.divider()
    
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
    st.sidebar.caption(f"Aktív MPP (Eredeti képen): {px_to_m:.4f}")
    
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

    # --- ELEMZÉS ---
    st.write("---")
    uploaded = st.file_uploader("Kép feltöltése", type=["jpg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Feltöltött kép", width=600)
        
        if st.button("Elemzés indítása", type="primary"):
            progress_text = st.empty()
            progress_text.text("Elemzés folyamatban...")
            
            try:
                # ITT adjuk át a skálázási faktort
                orig, _, mask_binary, buildings, total_pop = analyze(model, image, px_to_m, threshold, scale_factor)
                
                progress_text.text("Kész!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Épületek", len(buildings))
                c2.metric("Becsült Lakosság", int(total_pop))
                c3.metric("Zoom Faktor", f"{scale_factor}x")
                
                overlay = orig.copy()
                color = [0, 255, 0] if active_weights_path else [255, 0, 0] 
                overlay[mask_binary.astype(bool)] = color
                res = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
                
                st.image(res, caption="Eredmény", use_column_width=True)
                
                df = pd.DataFrame(buildings)
                if not df.empty:
                    st.download_button("Adatok letöltése (CSV)", df.to_csv(), "adatok.csv")
                else:
                    st.warning("Nem találtam épületet. Próbáld állítani a Skálázást!")

            except Exception as e:
                st.error(f"Hiba: {e}")
                import traceback
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
