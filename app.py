import os
import time
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

# Letöltés, ha hiányzik
if not os.path.exists(MODEL_PATH):
    gdown.download(URL, MODEL_PATH, quiet=False)

# ===============================
# SpaceNet statok
# ===============================

SPACENET_MEAN = np.array([0.339, 0.324, 0.285], dtype=np.float32)
SPACENET_STD  = np.array([0.139, 0.125, 0.122], dtype=np.float32)

# ===============================
# SEGÉDFÜGGVÉNYEK
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

def spacenet_preprocessing(image_or_array) -> np.ndarray:
    """SpaceNet-szerű előfeldolgozás: RGB csatornák egységesítése + mean/std normalizálás."""
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

def crop_center_square(img_array: np.ndarray, size: int = 320) -> np.ndarray:
    """
    Torzítás helyett: előbb skálázunk úgy, hogy a kisebbik oldal >= size, majd középről kivágunk size×size-et.
    """
    h, w = img_array.shape[:2]
    if min(h, w) >= size:
        # Skálázás felfelé nem szükséges, de kivágunk középről
        new_h, new_w = h, w
        resized = img_array
    else:
        scale = size / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    cropped = resized[start_y:start_y + size, start_x:start_x + size]
    return cropped

def segment_buildings_from_binary(binary_mask: np.ndarray, min_size: int = 50):
    """
    Épületek szegmentálása már thresholdolt bináris maszkból (0/1).
    """
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
# OSZTÁLYOZÓ + LAKOSSÁGSZÁM LOGIKA
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
    if area_m2 < 250:
        return 'kis_lakohaz'
    elif area_m2 < 1000:
        return 'kozepes_lakohaz'
    elif area_m2 < 2500:
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
# INPUT ADAPTÁLÁS A MODELLHEZ
# ===============================

def make_input_tensor(model, cropped_rgb: np.ndarray) -> tuple[np.ndarray, bool, tuple[int, int]]:
    """
    SpaceNet előfeldolgozás + méret és csatornasorrend igazítása a modell input_shape alapján.
    Visszaad: (input_tensor, channels_last, (target_h, target_w))
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    # Döntés channels_last vs channels_first és célméret
    if len(input_shape) == 4:
        if input_shape[-1] in (1, 3):  # channels_last: (None, H, W, C)
            channels_last = True
            target_h = input_shape[1] if input_shape[1] else cropped_rgb.shape[0]
            target_w = input_shape[2] if input_shape[2] else cropped_rgb.shape[1]
        else:  # channels_first: (None, C, H, W)
            channels_last = False
            target_h = input_shape[2] if input_shape[2] else cropped_rgb.shape[0]
            target_w = input_shape[3] if input_shape[3] else cropped_rgb.shape[1]
    else:
        channels_last = True
        target_h, target_w = cropped_rgb.shape[:2]

    resized = cv2.resize(cropped_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    pre = spacenet_preprocessing(resized)

    if channels_last:
        input_tensor = pre[None, ...]                      # (1, H, W, C)
    else:
        input_tensor = np.transpose(pre, (2, 0, 1))[None, ...]  # (1, C, H, W)

    return input_tensor, channels_last, (target_h, target_w)

# ===============================
# ANALÍZIS FUNKCIÓ
# ===============================

def analyze(model, image: Image.Image, px_to_m: float = 0.5, threshold: float = 0.5):
    # Kép előkészítés: CLAHE + középső 320×320 kivágás torzítás nélkül
    image = enhance_with_clahe(image)
    orig = np.array(image.convert("RGB"))
    h, w = orig.shape[:2]

    cropped = crop_center_square(orig, size=320)

    # Modell input előkészítése
    input_tensor, channels_last, _ = make_input_tensor(model, cropped)

    # Inference
    pred = model.predict(input_tensor, verbose=0)
    pred0 = pred[0] if isinstance(pred, (list, tuple)) else pred

    # Maszk kinyerése
    if pred0.ndim == 4:
        p = pred0[0]
        mask_cropped = p[..., 0] if channels_last else p[0, ...]
    elif pred0.ndim == 3:
        mask_cropped = pred0[..., 0] if channels_last else pred0[0, ...]
    else:
        raise ValueError(f"Váratlan predikciós alak: {pred0.shape}")

    # Maszk visszavetítése az eredeti képméretre (középső 320×320 területről)
    # Előbb a kivágott területhez illesztjük a maszkot (ha az input_resize eltért 320-tól),
    mask_320 = cv2.resize(mask_cropped, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Az eredeti kép közepére visszahelyezzük a 320×320 maszkot
    mask_full = np.zeros((h, w), dtype=np.float32)
    start_x = (w - 320) // 2
    start_y = (h - 320) // 2
    mask_full[start_y:start_y + 320, start_x:start_x + 320] = mask_320

    # Threshold alkalmazása csúszka értékkel
    binary_mask = (mask_full > threshold).astype(np.uint8)

    # Épületek kinyerése
    buildings = segment_buildings_from_binary(binary_mask, min_size=50)

    # Osztályozás + lakosság becslés
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

    return orig, mask_full, binary_mask, results, total_pop

# ===============================
# STREAMLIT FELÜLET
# ===============================

def main():
    st.title("🏠 Épület Analizátor lakosságszám becsléssel (SpaceNet előfeldolgozás, 320×320 crop, threshold csúszka)")
    st.sidebar.header("Beállítások")

    px_to_m = st.sidebar.slider("Pixel → méter", 0.1, 2.0, 0.5, 0.1)
    threshold = st.sidebar.slider("Maszk threshold", 0.0, 1.0, 0.5, 0.05)

    model = load_model()
    if model is None:
        st.error("❌ Modell betöltése sikertelen. Ellenőrizd a model.h5 fájlt és a kompatibilitást.")
        return

    st.caption("Modell információk")
    st.write("• Input shape:", model.input_shape)

    uploaded = st.file_uploader("Kép feltöltése", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Tölts fel egy képet a kezdéshez.")
        return

    image = Image.open(uploaded)
    st.image(image, caption=f"Feltöltött kép — {image.size[0]}×{image.size[1]}", use_column_width=True)

    if st.button("Elemzés indítása"):
        with st.spinner("Elemzés folyamatban..."):
            t0 = time.time()
            try:
                orig, mask_continuous, mask_binary, buildings, total_pop = analyze(model, image, px_to_m, threshold)
            except Exception as e:
                st.error(f"Hiba az elemzés során: {e}")
                st.exception(e)
                return
            infer_time = time.time() - t0

        st.subheader("📊 Eredmények")
        st.metric("Épületek száma", len(buildings))
        st.metric("Lakosság becslés", f"{total_pop:.0f} fő")
        st.metric("Futási idő", f"{infer_time:.2f} s")

        st.subheader("🖼️ Szegmentáció (threshold alkalmazva)")
        overlay = orig.copy()
        overlay[mask_binary.astype(bool)] = [255, 0, 0]
        result_img = cv2.addWeighted(orig, 0.6, overlay, 0.4, 0)
        st.image(result_img, caption=f"Szegmentált kép (piros = épület, threshold={threshold:.2f})", use_column_width=True)

        st.subheader("📦 Épületek")
        vis = orig.copy()
        for b in buildings:
            x, y, w_box, h_box = b['bbox']
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            label = f"{b['type']} ({b['population']} fő)"
            cv2.putText(vis, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        st.image(vis, caption="Detektált épületek", use_column_width=True)

        st.subheader("💾 Export")
        df = pd.DataFrame(buildings)
        st.download_button("CSV letöltése", df.to_csv(index=False), "epulet_adatok.csv", "text/csv")

# MAIN FÜGGVÉNY MEGHÍVÁSA
if __name__ == "__main__":
    main()
