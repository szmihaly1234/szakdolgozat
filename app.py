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

# ===============================
# 1. KONFIGURÁCIÓ
# ===============================
PT_MODEL_FILE_ID = "IDE_ÍRD_BE_A_DRIVE_ID_T" 
PT_MODEL_PATH = "unet_model.pth"

# Ha van saját UNet osztályod, ide másold be a struktúrát!
# Példa egy egyszerű struktúrára (helyettesítsd a sajátoddal):
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Ide jön a te 512x512-es architektúrád
        pass
    def forward(self, x):
        # ...
        return x

# ===============================
# 2. SEGÉDFÜGGVÉNYEK
# ===============================

@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(PT_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={PT_MODEL_FILE_ID}"
        gdown.download(url, PT_MODEL_PATH, quiet=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet() # Inicializáld az osztályodat
    # Súlyok betöltése (map_location fontos a CPU/GPU váltás miatt)
    state_dict = torch.load(PT_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def meters_per_pixel(lat, zoom):
    return math.cos(math.radians(lat)) * (2 * math.pi * 6378137.0) / (256 * (2 ** zoom))

def get_satellite_img(query, zoom=19):
    geolocator = ArcGIS()
    loc = geolocator.geocode(query)
    if not loc: return None, None
    
    # Egyszerűsített tile letöltés (1db központi tile a példa kedvéért)
    # A korábbi 3x3-as rácsodat is visszateheted ide
    lat, lon = loc.latitude, loc.longitude
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    return Image.open(BytesIO(res.content)), (lat, zoom)

# ===============================
# 3. UI ÉS LOGIKA
# ===============================

st.set_page_config(page_title="Building AI - PyTorch", layout="wide")
st.title("🛰️ Épületszegmentáló (PyTorch Only)")

model, device = load_pytorch_model()

search_query = st.text_input("Helyszín keresése:", "Budapest, Parlament")

if st.button("Keresés és Elemzés"):
    img, geo_data = get_satellite_img(search_query)
    
    if img:
        # 1. Előkészítés 512x512-re
        img_resized = img.resize((512, 512))
        img_np = np.array(img_resized)
        
        # 2. PyTorch Preprocessing (RGB, NCHW, Normalizált)
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # 3. Predikció
        with torch.no_grad():
            output = model(input_tensor)
            # Ha a modelled logits-ot ad vissza:
            prob = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
            mask = (prob > 0.5).astype(np.uint8)
            
        # 4. Megjelenítés
        col1, col2 = st.columns(2)
        col1.image(img_resized, caption="Eredeti műholdkép (512x512)")
        
        # Maszk rávetítése (Overlay)
        overlay = img_np.copy()
        overlay[mask == 1] = [255, 0, 0] # Piros szín az épületeknek
        combined = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
        col2.image(combined, caption="Szegmentált eredmény")
        
        # Terület számítás
        lat, zoom = geo_data
        m_per_px = meters_per_pixel(lat, zoom)
        # Mivel 512-re skáláztuk az eredetileg 256-os tile-t, korrigálni kell
        pixel_size_corrected = m_per_px * (256 / 512)
        
        building_pixels = np.sum(mask)
        total_area = building_pixels * (pixel_size_corrected ** 2)
        st.metric("Becsült beépített terület", f"{total_area:.2f} m²")
    else:
        st.error("Nem található ilyen helyszín.")
