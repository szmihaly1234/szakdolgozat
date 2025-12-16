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
import requests
from io import BytesIO
from geopy.geocoders import Nominatim

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

st.set_page_config(page_title="Lakosság AI (Ingyenes Verzió)", page_icon="🛰️", layout="wide")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SPACENET_MEAN = np.array([0.339, 0.324, 0.285], dtype=np.float32)
SPACENET_STD  = np.array([0.139, 0.125, 0.122], dtype=np.float32)

# ===============================
# 3. SEGÉDFÜGGVÉNYEK (FILE & GEO)
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
    if mode == "Auto (GPS)":
        return meters_per_pixel_web_mercator(lat, zoom) if (lat and zoom) else manual
    elif mode == "Kalibráció (ismert tárgy)":
        return known_m / meas_px if (known_m and meas_px) else manual
    return manual

# --- ESRI LETÖLTŐ ---

def deg2num(lat_deg, lon_deg, zoom):
    """Koordináták átváltása csempe (tile) indexekre."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_esri_satellite(location_name, zoom=19):
    """
    Ingyenes Esri World Imagery letöltése.
    """
    # 1. Geocoding (Hely megkeresése) - JAVÍTOTT RÉSZ
    try:
        # Egyedi user_agent, hogy ne tiltsanak le, és TIMEOUT beállítása
        geolocator = Nominatim(user_agent="lakossag_app_free_v2")
        
        # ITT A JAVÍTÁS: timeout=10 (10 másodpercet vár, nem 1-et)
        location = geolocator.geocode(location_name, timeout=10)
        
    except Exception as e:
        return None, None, f"Geocoding hiba: {str(e)}"
    
    if not location:
        return None, None, "Nem találom ezt a települést/címet."
    
    lat, lon = location.latitude, location.longitude
    
    # 2. Csempék letöltése
    xtile, ytile = deg2num(lat, lon, zoom)
    
    base_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"
    
    # 3x3 rács
    full_image = Image.new('RGB', (256*3, 256*3))
    headers = {'User-Agent': 'Mozilla/5.0'} 
    
    try:
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                url = f"{base_url}/{zoom}/{ytile + y_offset}/{xtile + x_offset}"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    tile_img = Image.open(BytesIO(response.content))
                    paste_x = (x_offset + 1) * 256
                    paste_y = (y_offset + 1) * 256
                    full_image.paste(tile_img, (paste_x, paste_y))
                else:
                    print(f"Hiba a csempénél: {url}")
                    
        return full_image, (lat, lon, zoom), None

    except Exception as e:
        return None, None, f"Hálózati hiba: {str(e)}"

# ===============================
# 4. KÉPFELDOLGOZÁS ÉS MODEL
# ===============================

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
    if img_array.ndim == 2: 
        img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.ndim == 3 and img_array.shape[2] == 4: 
        img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else: 
        img = img_array
    
    img_float = img.astype(np.float32) / 255.0
    return (img_float - SPACENET_MEAN) / SPACENET_STD

def segment_buildings_with_road_filter(binary_mask, min_size=50, max_aspect_ratio=5.0):
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size: continue
            
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        shortest = min(w, h)
        longest = max(w, h)
        if shortest == 0: continue 
        aspect_ratio = longest / shortest
        
        if aspect_ratio > max_aspect_ratio:
            continue
            
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(cnt)
        buildings.append({
            'area': area,
            'bbox': (x_bbox, y_bbox, w_bbox, h_bbox),
            'ratio': aspect_ratio
        })
        
    return buildings

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
                
        return model, info_msg, 0

    except Exception as e:
        return None, f"Kritikus hiba: {str(e)}", 0

# ===============================
# 6. SLIDING WINDOW LOGIKA
# ===============================

def predict_sliding_window(model, full_image, tile_size=320, overlap=0.5):
    h, w, c = full_image.shape
    stride = int(tile_size * (1 - overlap))
    if stride < 1: stride = 1
    
    pad_h = (tile_size - (h % stride)) % stride + (tile_size - stride)
    pad_w = (tile_size - (w % stride)) % stride + (tile_size - stride)
    
    padded_image = cv2.copyMakeBorder(full_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw, _ = padded_image.shape
    
    full_mask = np.zeros((ph, pw), dtype=np.float32)
    count_mask = np.zeros((ph, pw), dtype=np.float32)
    
    total_steps = ((ph - tile_size) // stride + 1) * ((pw - tile_size) // stride + 1)
    progress_bar = st.progress(0)
    step_count = 0
    
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
    return avg_mask[:h, :w]

# ===============================
# 7. ANALÍZIS & BECSLÉS
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

def analyze(model, image, px_to_m, threshold, overlap, road_sensitivity):
    image_enhanced = enhance_with_clahe(image)
    orig_np = np.array(image_enhanced.convert("RGB"))
    
    raw_mask = predict_sliding_window(model, orig_np, tile_size=320, overlap=overlap)
    binary_mask = (raw_mask > threshold).astype(np.uint8)

    buildings = segment_buildings_with_road_filter(binary_mask, min_size=50, max_aspect_ratio=road_sensitivity)

    results = []
    total_pop = 0
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
    st.title("Lakosság AI (Ingyenes 🆓)")

    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Beállítások")
    
    st.sidebar.subheader("1. AI Modell")
    model_option = st.sidebar.radio("Verzió:", ("Globális", "Párizs/EU"), on_change=clear_model_cache)
    active_weights_path = WEIGHTS_PATH if model_option == "Párizs/EU" else None
    
    if active_weights_path:
        ensure_file_from_drive(WEIGHTS_FILE_ID, WEIGHTS_PATH)

    with st.spinner("Modell betöltése..."):
        model, status_msg, _ = load_model_pro(active_weights_path)
    
    if not model:
        st.error(status_msg)
        st.stop()
    st.sidebar.success(f"Aktív: {status_msg}")

    st.sidebar.subheader("2. Méretarány")
    mode = st.sidebar.selectbox("Mód", ["Kézi (slider)", "Auto (GPS)", "Kalibráció"])
    manual_px_to_m = st.sidebar.slider("Pixel -> Méter", 0.05, 2.0, 0.3, 0.05)
    
    st.sidebar.subheader("3. Finomhangolás")
    quality_mode = st.sidebar.select_slider("Minőség (Overlap)", options=["Gyors", "Normál", "Magas", "Ultra"], value="Magas")
    overlap_map = {"Gyors": 0.1, "Normál": 0.25, "Magas": 0.5, "Ultra": 0.75}
    
    threshold = st.sidebar.slider("Küszöb (Threshold)", 0.2, 0.9, 0.5)
    
    st.sidebar.subheader("🚫 Út szűrés (Road Filter)")
    road_ratio = st.sidebar.slider("Max Hossz/Szél arány", 2.0, 10.0, 5.0)

    # --- FÜLEK ---
    tab1, tab2 = st.tabs(["📁 Fájl feltöltés", "🌍 Ingyenes Műholdkép (Esri)"])

    # --- 1. TAB: FÁJL ---
    with tab1:
        uploaded = st.file_uploader("Kép feltöltése", type=["jpg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Feltöltött kép", width=600)
            
            lat, zoom, known_m, meas_px = None, None, None, None
            if mode == "Auto (GPS)":
                st.info("Feltöltött képnél add meg a koordinátákat a méréshez:")
                c1, c2 = st.columns(2)
                lat = c1.number_input("Szélesség (Lat)", value=47.4979)
                zoom = c2.number_input("Zoom Level (kb)", value=19)
            
            px_to_m = compute_px_to_m_mode(mode, manual_px_to_m, lat, zoom, known_m, meas_px)

            if st.button("Elemzés indítása (Fájl)", type="primary"):
                run_analysis(model, image, px_to_m, threshold, overlap_map[quality_mode], road_ratio)

    # --- 2. TAB: HELYKERESÉS (INGYENES) ---
    with tab2:
        st.subheader("Keress bármelyik településre (Ingyenes!)")
        st.caption("Esri World Imagery műholdképeket használunk. Nem kell API kulcs.")
        
        search_query = st.text_input("Település / Cím", placeholder="pl. Pécs, Széchenyi tér")
        
        if st.button("Keresés és Letöltés"):
            if not search_query:
                st.warning("Írj be valamit!")
            else:
                with st.spinner(f"Kapcsolódás az Esri műholdakhoz ({search_query})..."):
                    img, geo_data, err = download_esri_satellite(search_query, zoom=19)
                
                if err:
                    st.error(err)
                
                if img:
                    st.session_state['last_search_img'] = img
                    st.session_state['last_geo'] = geo_data
                    st.success(f"Siker! Megtaláltam.")
        
        if 'last_search_img' in st.session_state:
            st.image(st.session_state['last_search_img'], caption="Letöltött műholdkép (Esri)", width=600)
            
            if st.button("Elemzés futtatása ezen a képen", type="primary", key="esri_run"):
                lat, lon, zoom = st.session_state['last_geo']
                auto_px_to_m = meters_per_pixel_web_mercator(lat, zoom)
                
                run_analysis(model, st.session_state['last_search_img'], auto_px_to_m, threshold, overlap_map[quality_mode], road_ratio)

def run_analysis(model, image, px_to_m, threshold, overlap, road_ratio):
    progress_text = st.empty()
    progress_text.info("Neurális hálózat futtatása...")
    start = time.time()
    try:
        orig, _, mask_binary, buildings, total_pop = analyze(
            model, image, px_to_m, threshold, overlap, road_ratio
        )
        elapsed = time.time() - start
        progress_text.success(f"Kész! ({elapsed:.1f}s)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Talált Épületek", len(buildings))
        c2.metric("Becsült Lakosság", int(total_pop))
        c3.metric("Pixel méret", f"{px_to_m:.3f} m")

        vis_mask = np.zeros(mask_binary.shape, dtype=np.uint8)
        for b in buildings:
            x, y, w, h = b['bbox']
            cv2.rectangle(vis_mask, (x, y), (x+w, y+h), 255, -1)
            
        overlay = orig.copy()
        overlay[vis_mask == 255] = [0, 255, 0] 
        diff = cv2.bitwise_xor(mask_binary, vis_mask)
        overlay[diff == 1] = [255, 0, 0] 
        
        res = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
        st.image(res, caption="Eredmény", use_column_width=True)
        
        df = pd.DataFrame(buildings)
        if not df.empty:
            st.download_button("CSV Letöltése", df.to_csv(), "adatok.csv")
        else:
            st.warning("Nem találtam épületet.")
            
    except Exception as e:
        st.error(f"Hiba történt: {e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
