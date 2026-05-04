import os
import numpy as np
import pandas as pd
import cv2
import gdown
import streamlit as st
import torch
import torch.nn as nn
import requests
import math
from PIL import Image
from io import BytesIO
from geopy.geocoders import ArcGIS
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. ARCHITEKTÚRA DEFINÍCIÓ (U-Net)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool(d1)
        d2 = self.down2(p1); p2 = self.pool(d2)
        d3 = self.down3(p2); p3 = self.pool(d3)
        d4 = self.down4(p3); p4 = self.pool(d4)
        bn = self.bottleneck(p4)
        u4 = self.up4(bn); c4 = self.conv4(torch.cat([d4, u4], dim=1))
        u3 = self.up3(c4); c3 = self.conv3(torch.cat([d3, u3], dim=1))
        u2 = self.up2(c3); c2 = self.conv2(torch.cat([d2, u2], dim=1))
        u1 = self.up1(c2); c1 = self.conv1(torch.cat([d1, u1], dim=1))
        return self.out(c1)

# ==========================================
# 2. MODELL BETÖLTÉSE
# ==========================================
PT_MODEL_FILE_ID = "1GtvejvLLhNAUHe1oMz9I7BlNXLBDAxCd" 
PT_MODEL_PATH = "unet_building_segmentation_paris_2.pth"

@st.cache_resource(show_spinner="AI Modell letöltése...")
def load_pytorch_model():
    if not os.path.exists(PT_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={PT_MODEL_FILE_ID}"
        gdown.download(url, PT_MODEL_PATH, quiet=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model, device

# ==========================================
# 3. NÉPESSÉGBECSLÉSI LOGIKA
# ==========================================
def analyze_and_clean_mask(mask, lat, zoom):
    # Ha feltöltött képünk van (nincs lat/zoom), alapértelmezett méretarányt használunk
    if lat is None or zoom is None:
        m_per_px = 0.5 # Átlagos felbontás feltöltött képnél
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
        
        if area_m2 < 100:
            b_type, pop = 'Kis lakóház', 2.9 * max(1, area_m2/100)
        elif area_m2 < 300:
            b_type, pop = 'Közepes lakóház', 3.2 * max(1, area_m2/100)
        elif area_m2 < 1000:
            b_type, pop = 'Nagy lakóház', 4.1 * max(1, area_m2/100)
        else:
            b_type, pop = 'Társasház', 45 * (max(8, area_m2/80)/10)
            
        buildings.append({'Típus': b_type, 'Terület (m²)': round(area_m2, 1), 'Becsült lakosság': round(pop, 1)})
    return clean_mask, buildings

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Lakosság AI (PyTorch)", layout="wide", page_icon="🛰️")
st.title("🛰️ Lakosság AI - Műholdas Népességbecslés")

st.sidebar.header("⚙️ Beállítások")
source_option = st.sidebar.radio("Adatforrás kiválasztása:", ("Műholdas Kereső", "Saját kép feltöltése"))
threshold = st.sidebar.slider("Érzékenység (Threshold)", 0.100, 0.995, 0.500, 0.005) # Alapérték módosítva 0.5-re

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
# 5. KÖZÖS ELEMZÉSI LOGIKA
# ==========================================
inference_transforms = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
])

if img_to_process:
    with st.spinner("AI elemzés futtatása..."):
        img_np = np.array(img_to_process) # RGB formátum

        # 1. Előfeldolgozás Albumentations segítségével
        aug = inference_transforms(image=img_np)
        
        # 2. Float konverzió és Batch dimenzió hozzáadása
        model, device = load_pytorch_model()
        input_t = aug["image"].float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_t)
            # A sigmoid már garantáltan 0.0 és 1.0 közötti abszolút valószínűséget ad
            prob = torch.sigmoid(output).cpu().numpy()[0, 0] 
            
            # 3. Küszöbérték (Threshold) közvetlen alkalmazása az oldalsávról
            mask = (prob > threshold).astype(np.uint8)
            
            # 4. Morfológiai szűrés (apró zajok és lyukak eltüntetése)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            
            # 5. Népességbecslés és poligonok tisztítása
            final_mask, buildings_data = analyze_and_clean_mask(mask, current_lat, current_zoom)
            
            # 6. Megjelenítés - Zöld réteg ráhúzása a képre
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
