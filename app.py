import os
import numpy as np
import pandas as pd
import cv2
import gdown
import streamlit as st
import torch
import requests
import math
from PIL import Image
from io import BytesIO
from geopy.geocoders import ArcGIS
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# 1. MODELL KONFIGURÁCIÓ ÉS BETÖLTÉS
# ==========================================
PT_MODEL_FILE_ID = "1Pn5gSZSQ9D3CEGHKsnmXRGo3dt7dhMnT" 
PT_MODEL_PATH = "best_resnet34_unet.pth"

@st.cache_resource(show_spinner="ResNet34 AI Modell betöltése...")
def load_pytorch_model():
    if not os.path.exists(PT_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={PT_MODEL_FILE_ID}"
        gdown.download(url, PT_MODEL_PATH, quiet=False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SMP modell inicializálása (ResNet34 kódolóval)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=1,
        activation=None 
    )
    
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=device))
    
    # KÖTELEZŐ JAVÍTÁS: .eval() mód a csúszóablakos illesztési hibák elkerülése végett!
    model.to(device).eval() 
    return model, device

# ==========================================
# 2. CSÚSZÓABLAKOS ÉS BECSLÉSI LOGIKA
# ==========================================

def sliding_window_inference(model, device, image_np, window_size=512, stride=256):
    """
    Tetszőleges méretű képet elemez matematikai paddinggel, 
    hogy a szélek és a sarkok is 100%-osan le legyenek fedve.
    """
    H_orig, W_orig, _ = image_np.shape
    
    # 1. Matematikailag tökéletes padding (kibővítés) kiszámítása
    pad_h = 0
    if H_orig < window_size:
        pad_h = window_size - H_orig
    elif (H_orig - window_size) % stride != 0:
        pad_h = stride - ((H_orig - window_size) % stride)

    pad_w = 0
    if W_orig < window_size:
        pad_w = window_size - W_orig
    elif (W_orig - window_size) % stride != 0:
        pad_w = stride - ((W_orig - window_size) % stride)

    # 2. Kép kibővítése tükrözéssel (BORDER_REFLECT) a folytonos átmenetért
    if pad_h > 0 or pad_w > 0:
        image_padded = cv2.copyMakeBorder(image_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        image_padded = image_np.copy()

    H_pad, W_pad, _ = image_padded.shape

    full_prob_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    count_map = np.zeros((H_pad, W_pad), dtype=np.float32)

    patch_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # 3. Csúszóablak futtatása (Garantáltan végigmegy a kibővített képen)
    for y in range(0, H_pad - window_size + 1, stride):
        for x in range(0, W_pad - window_size + 1, stride):
            crop = image_padded[y:y+window_size, x:x+window_size]
            input_t = patch_transform(image=crop)["image"].float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob = torch.sigmoid(model(input_t)).cpu().numpy()[0, 0]
                full_prob_map[y:y+window_size, x:x+window_size] += prob
                count_map[y:y+window_size, x:x+window_size] += 1.0

    # Átlagolás a pontos átfedések (stride) miatt
    full_prob_map /= np.maximum(count_map, 1.0)
    
    # 4. Visszavágás az eredeti, feltöltött kép méretére!
    return full_prob_map[:H_orig, :W_orig]


def analyze_and_clean_mask(mask, lat, zoom, original_h, original_w, real_width_m=None):
    """
    A bináris maszkból kinyeri az épületeket és megbecsüli a lakosságot.
    """
    # Pixel -> Méter konverzió meghatározása
    if lat is None or zoom is None:
        # Saját kép: Felhasználói input alapján (Szélesség méterben / Szélesség pixelben)
        if real_width_m and original_w > 0:
            m_per_px = real_width_m / original_w
        else:
            m_per_px = 0.3 # Alapértelmezett becslés
    else:
        # ArcGIS: Matematikai képlet a zoom szint és szélességi kör alapján
        m_per_px = (math.cos(math.radians(lat)) * 40075016.686 / (256 * 2**zoom)) * (256/512)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    buildings = []
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 50: continue # Túl kicsi zajok eldobása
        
        cv2.drawContours(clean_mask, [cnt], -1, 1, thickness=cv2.FILLED)
        area_m2 = area_px * (m_per_px**2)
        
        # Népességbecslési heurisztika alapterület alapján
        if area_m2 < 100:
            b_type, pop = 'Kis lakóház', 2.9 * max(1, area_m2/100)
        elif area_m2 < 300:
            b_type, pop = 'Közepes lakóház', 3.2 * max(1, area_m2/100)
        elif area_m2 < 1000:
            b_type, pop = 'Nagy lakóház', 4.1 * max(1, area_m2/100)
        else:
            b_type, pop = 'Társasház / Intézmény', 45 * (max(8, area_m2/80)/10)
            
        buildings.append({
            'Típus': b_type, 
            'Terület (m²)': round(area_m2, 1), 
            'Becsült lakosság': round(pop, 1)
        })
    return clean_mask, buildings

# ==========================================
# 3. STREAMLIT FELHASZNÁLÓI FELÜLET
# ==========================================
st.set_page_config(page_title="Lakosság AI (ResNet34)", layout="wide", page_icon="🛰️")
st.title("🛰️ Lakosság AI - Műholdas Népességbecslés")

st.sidebar.header("⚙️ Beállítások")
source_option = st.sidebar.radio("Adatforrás kiválasztása:", ("Műholdas Kereső", "Saját kép feltöltése"))
threshold = st.sidebar.slider("AI Érzékenység (Threshold)", 0.100, 0.995, 0.400, 0.050)

img_to_process = None
current_lat, current_zoom = None, None
real_width_m = None

if source_option == "Műholdas Kereső":
    zoom_level = st.sidebar.select_slider("Műholdkép Zoom szint", options=[18, 19, 20], value=19)
    query = st.text_input("Helyszín keresése (Cím vagy Település):", "Szeged, Pihenő utca 69")
    if st.button("Helyszín lekérése és elemzése"):
        geolocator = ArcGIS()
        loc = geolocator.geocode(query)
        if loc:
            current_lat, current_zoom = loc.latitude, zoom_level
            n = 2.0 ** zoom_level
            xtile = int((loc.longitude + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.asinh(math.tan(math.radians(current_lat))) / math.pi) / 2.0 * n)
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{ytile}/{xtile}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200:
                img_to_process = Image.open(BytesIO(resp.content)).convert("RGB").resize((512, 512))
            else:
                st.error("Hiba a műholdkép letöltésekor az ArcGIS szerverről.")
        else:
            st.error("A megadott helyszín nem található.")

else:
    uploaded_file = st.file_uploader("Válassz egy műholdképet (JPG/PNG):", type=['png', 'jpg', 'jpeg'])
    real_width_m = st.sidebar.number_input(
        "📏 Kép valós szélessége (méterben):", 
        min_value=10, max_value=20000, value=300, step=10,
        help="Add meg, hogy a kép bal szélétől a jobb széléig hány méter a távolság a valóságban."
    )
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file).convert("RGB")
        st.success(f"Kép feltöltve: {img_to_process.size[0]}x{img_to_process.size[1]} pixel.")

# ==========================================
# 4. ELEMZÉS ÉS MEGJELENÍTÉS
# ==========================================

if img_to_process:
    with st.spinner("AI elemzés folyamatban (Sliding Window)..."):
        img_np = np.array(img_to_process)
        h, w, _ = img_np.shape
        model, device = load_pytorch_model()
        
        # Inferencia: Saját képnél vagy nagy felbontásnál csúszóablakot használunk
        if source_option == "Saját kép feltöltése" or h > 512 or w > 512:
            prob_map = sliding_window_inference(model, device, img_np)
        else:
            # Fix méretű ArcGIS csempe elemzése
            inference_transforms = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            aug = inference_transforms(image=img_np)
            input_t = aug["image"].float().unsqueeze(0).to(device)
            with torch.no_grad():
                prob_map = torch.sigmoid(model(input_t)).cpu().numpy()[0, 0]

        # DEBUG SÁV
        st.warning(f"🔍 DEBUG: Max valószínűség: {prob_map.max():.4f} | Min: {prob_map.min():.4f}")
        
        # Küszöbérték és utómunka
        mask = (prob_map > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # Népességbecslés hívása (Paraméterátadással a méretezéshez)
        final_mask, buildings_data = analyze_and_clean_mask(mask, current_lat, current_zoom, h, w, real_width_m)
        
        # Eredmény kép készítése (Zöld maszk rávetítése)
        overlay = img_np.copy()
        overlay[final_mask == 1] = [0, 255, 0]
        res_img = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
        
        # UI Megjelenítés elrendezése
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.image(res_img, caption="AI által felismert épületek", use_container_width=True)
        with c2:
            total_pop = sum(b['Becsült lakosság'] for b in buildings_data)
            st.metric("👥 Becsült összlakosság", int(total_pop))
            st.metric("🏠 Felismert épületek", len(buildings_data))
            if buildings_data:
                st.dataframe(pd.DataFrame(buildings_data), hide_index=True)
