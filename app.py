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
# 1. MODELL BETÖLTÉSE (ResNet34 U-Net)
# ==========================================
PT_MODEL_FILE_ID = "1Pn5gSZSQ9D3CEGHKsnmXRGo3dt7dhMnT" 
PT_MODEL_PATH = "best_resnet34_unet.pth"

@st.cache_resource(show_spinner="AI Modell letöltése és betöltése (ResNet34)...")
def load_pytorch_model():
    if not os.path.exists(PT_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={PT_MODEL_FILE_ID}"
        gdown.download(url, PT_MODEL_PATH, quiet=False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SMP modell inicializálása
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None 
    )
    
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=device))
    model.to(device).eval() 
    return model, device

# ==========================================
# 2. NÉPESSÉGBECSLÉSI LOGIKA
# ==========================================
def analyze_and_clean_mask(mask, lat, zoom):
    # Ha feltöltött képünk van (nincs lat/zoom), alapértelmezett méretarányt használunk
    if lat is None or zoom is None:
        m_per_px = 0.5 
    else:
        m_per_px = (math.cos(math.radians(lat)) * 40075016.686 / (256 * 2**zoom)) * (256/512)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    buildings = []
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 50: continue 
        
        cv2.drawContours(clean_mask, [cnt], -1, 1, thickness=cv2.FILLED)
        area_m2 = area_px * (m_per_px**2)
        
        # Becslési heurisztika
        if area_m2 < 100:
            b_type, pop = 'Kis lakóház', 2.9 * max(1, area_m2/100)
        elif area_m2 < 300:
            b_type, pop = 'Közepes lakóház', 3.2 * max(1, area_m2/100)
        elif area_m2 < 1000:
            b_type, pop = 'Nagy lakóház', 4.1 * max(1, area_m2/100)
        else:
            b_type, pop = 'Társasház / Intézmény', 45 * (max(8, area_m2/80)/10)
            
        buildings.append({'Típus': b_type, 'Terület (m²)': round(area_m2, 1), 'Becsült lakosság': round(pop, 1)})
    return clean_mask, buildings

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Lakosság AI (ResNet34)", layout="wide", page_icon="🛰️")
st.title("🛰️ Lakosság AI - Műholdas Népességbecslés")

st.sidebar.header("⚙️ Beállítások")
source_option = st.sidebar.radio("Adatforrás kiválasztása:", ("Műholdas Kereső", "Saját kép feltöltése"))
threshold = st.sidebar.slider("Érzékenység (Threshold)", 0.100, 0.995, 0.400, 0.050)

img_to_process = None
current_lat, current_zoom = None, None

if source_option == "Műholdas Kereső":
    zoom_level = st.sidebar.select_slider("Műholdkép Zoom szint", options=[18, 19, 20], value=19)
    query = st.text_input("Helyszín keresése:", "Budapest, Hősök tere")
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
                st.error("Hiba a műholdkép letöltésekor.")
        else:
            st.error("A helyszín nem található.")

else:
    uploaded_file = st.file_uploader("Válassz egy műholdképet (JPG/PNG):", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file).convert("RGB").resize((512, 512))
        st.success("Kép sikeresen feltöltve!")

# ==========================================
# 4. KÖZÖS ELEMZÉSI LOGIKA (JAVÍTVA)
# ==========================================
# JAVÍTÁS 1: Hozzáadva az ImageNet normalizáció
inference_transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(),
])

if img_to_process:
    with st.spinner("AI elemzés futtatása..."):
        img_np = np.array(img_to_process) 

        # Előfeldolgozás
        aug = inference_transforms(image=img_np)
        model, device = load_pytorch_model()
        
        # Tenzor előkészítése
        input_t = aug["image"].float().unsqueeze(0).to(device)
        
        # JAVÍTÁS 2: Kivettük a manuális osztást (input_t = input_t / 255.0)

        with torch.no_grad():
            output = model(input_t)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0] 
            
            # DEBUG INFORMÁCIÓ 
            st.warning(f"🔍 DEBUG: Max valószínűség: {prob.max():.4f} | Min: {prob.min():.4f}")
            
            # Küszöbérték alkalmazása
            mask = (prob > threshold).astype(np.uint8)
            
            # Morfológiai szűrés (zajok eltüntetése)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            
            # Népességbecslés
            final_mask, buildings_data = analyze_and_clean_mask(mask, current_lat, current_zoom)
            
            # Megjelenítés
            overlay = img_np.copy()
            overlay[final_mask == 1] = [0, 255, 0]
            res_img = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
            
            # UI Frissítés
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.image(res_img, caption="Elemzett eredmény", use_container_width=True)
            with c2:
                total_pop = sum(b['Becsült lakosság'] for b in buildings_data)
                st.metric("👥 Becsült összlakosság", int(total_pop))
                st.metric("🏠 Felismert épületek", len(buildings_data))
                if buildings_data:
                    st.dataframe(pd.DataFrame(buildings_data), hide_index=True)
