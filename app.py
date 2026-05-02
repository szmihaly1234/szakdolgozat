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
PT_MODEL_FILE_ID = "1gZgDnZiX1nTfBLQiqESLFcQzZO5HHrVy" 
PT_MODEL_PATH = "unet_building_segmentation.pth"

@st.cache_resource(show_spinner="AI Modell letöltése és betöltése a memóriába...")
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
# 3. NÉPESSÉGBECSLÉSI ÉS MASZK TISZTÍTÓ LOGIKA
# ==========================================
def analyze_and_clean_mask(mask, lat, zoom):
    m_per_px = (math.cos(math.radians(lat)) * 40075016.686 / (256 * 2**zoom)) * (256/512)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_mask = np.zeros_like(mask)
    buildings = []
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        
        # Szigorú zajszűrés: ami 50 pixel alatti, az kuka
        if area_px < 50:  
            continue 
            
        # Ha átment a teszten, rárajzoljuk a tiszta maszkra
        cv2.drawContours(clean_mask, [cnt], -1, 1, thickness=cv2.FILLED)
        
        area_m2 = area_px * (m_per_px**2)
        
        # Kategorizálás és lakosság becslés
        if area_m2 < 100:
            b_type, pop = 'Kis lakóház', 2.9 * max(1, area_m2/100)
        elif area_m2 < 300:
            b_type, pop = 'Közepes lakóház', 3.2 * max(1, area_m2/100)
        elif area_m2 < 1000:
            b_type, pop = 'Nagy lakóház', 4.1 * max(1, area_m2/100)
        else:
            b_type, pop = 'Társasház', 45 * (max(8, area_m2/80)/10)
            
        buildings.append({
            'Típus': b_type, 
            'Terület (m²)': round(area_m2, 1), 
            'Becsült lakosság': round(pop, 1)
        })
        
    return clean_mask, buildings

# ==========================================
# 4. STREAMLIT FELHASZNÁLÓI FELÜLET (UI)
# ==========================================
st.set_page_config(page_title="Lakosság AI (PyTorch)", layout="wide", page_icon="🛰️")
st.title("🛰️ Lakosság AI - Műholdas Népességbecslés")

# Oldalsáv
st.sidebar.header("⚙️ Beállítások")
# Finomított csúszka, apróbb lépésekkel a pengeéles kalibráláshoz
threshold = st.sidebar.slider("Érzékenység (Threshold)", 0.500, 0.995, 0.950, 0.005, format="%.3f", help="Nagyon finom hangolás. Keresd meg a 'pengeélt'!")
zoom_level = st.sidebar.select_slider("Műholdkép Zoom szint", options=[18, 19, 20], value=19)

# Kereső
query = st.text_input("Helyszín keresése (település, utca):", "Budapest, Hősök tere")

if st.button("Elemzés Futtatása", type="primary"):
    with st.spinner("Műholdkép elemzése folyamatban..."):
        geolocator = ArcGIS()
        loc = geolocator.geocode(query)
        
        if loc:
            lat, lon = loc.latitude, loc.longitude
            n = 2.0 ** zoom_level
            xtile = int((lon + 180.0) / 360.0 * n)
            ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
            
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{ytile}/{xtile}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
                img_np = np.array(img)

                # OpenCV BGR formátum konverzió (Albumentations kompatibilitás)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_input = img_bgr.astype(np.float32)

                model, device = load_pytorch_model()
                input_t = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device=device)

                with torch.no_grad():
                    output = model(input_t)
                    
                    # 1. Alap valószínűségek generálása
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                    
                    prob_min = prob.min()
                    prob_max = prob.max()
                    
                    # 2. NORMALIZÁLÁS (0-1 közé széthúzzuk az értékeket)
                    prob_shifted = prob - prob_min
                    range_val = prob_max - prob_min
                    
                    if range_val > 0:
                        prob_normalized = prob_shifted / range_val
                    else:
                        prob_normalized = prob_shifted
                        
                    # Gamma korrekció LÁGYÍTVA (hatvány 2.0-ra csökkentve)
                    prob_gamma = np.power(prob_normalized, 2.0) 
                    
                    # --- AUTOMATIKUS (OTSU) KÜSZÖB BECSLÉS ---
                    prob_8bit = (prob_gamma * 255).astype(np.uint8)
                    otsu_ret, _ = cv2.threshold(prob_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    otsu_suggested = otsu_ret / 255.0
                    
                    st.info(f"💡 **AI Javaslat:** Az algoritmus szerint az ideális vágási pont: **{otsu_suggested:.3f}**. Próbáld oda húzni a csúszkát!")
                    
                    # 3. Közvetlen küszöbölés a beállított csúszka alapján
                    mask = (prob_gamma > threshold).astype(np.uint8)
                    
                    # Heatmap generálás a vizualizációhoz
                    heatmap_colored = cv2.applyColorMap(prob_8bit, cv2.COLORMAP_JET)

                # ==========================================
                # POST-PROCESSING (Zajszűrés és foltozás)
                # ==========================================
                # a) Opening: Letörli a kicsi, magányos zöld pöttyöket
                kernel_open = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                
                # b) Closing: Összetapasztja a házak körüli lyukakat
                kernel_close = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

                # c) Csak a tényleges, elég nagy épületeket tartjuk meg a maszkban
                final_mask, buildings_data = analyze_and_clean_mask(mask, lat, zoom_level)
                
                total_pop = sum(b['Becsült lakosság'] for b in buildings_data)

                # Zöld maszk rávetítése az eredeti képre
                overlay = img_np.copy()
                overlay[final_mask == 1] = [0, 255, 0]
                res_img = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)

                # ==========================================
                # MEGJELENÍTÉS STREAMLITEN
                # ==========================================
                st.subheader("🔍 A Modell 'Látása' (Heatmap)")
                st.write("A piros/sárga részeket veszi épületnek, a kéket háttérnek.")
                
                col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
                with col_h2:
                    st.image(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.markdown("---")
                
                c1, c2 = st.columns([1.5, 1])
                
                with c1:
                    st.image(res_img, caption=f"Szegmentált eredmény ({query})", use_container_width=True)
                
                with c2:
                    st.metric("👥 Becsült összlakosság", int(total_pop))
                    st.metric("🏠 Felismert épületek", len(buildings_data))
                    
                    if buildings_data:
                        df = pd.DataFrame(buildings_data)
                        st.dataframe(df, hide_index=True)
            else:
                st.error("Nem sikerült letölteni a műholdképet. Kérlek próbálj meg egy másik helyszínt!")
        else:
            st.error("A megadott helyszín nem található.")
