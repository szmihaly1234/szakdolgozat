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

# --- ARCHITEKTÚRA --- (Változatlan a legutóbbi működő verzióhoz képest)
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
    def forward(self, x): return self.conv(x)

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

# --- KONFIGURÁCIÓ ---
PT_MODEL_FILE_ID = "1gZgDnZiX1nTfBLQiqESLFcQzZO5HHrVy" 
PT_MODEL_PATH = "unet_building_segmentation.pth"

@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(PT_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={PT_MODEL_FILE_ID}"
        gdown.download(url, PT_MODEL_PATH, quiet=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(); state_dict = torch.load(PT_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict); model.to(device).eval()
    return model, device

# --- BECSLÉSI LOGIKA ---
def analyze_buildings(mask, lat, zoom):
    m_per_px = (math.cos(math.radians(lat)) * 40075016.686 / (256 * 2**zoom)) * (256/512)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings = []
    for cnt in contours:
        area_m2 = cv2.contourArea(cnt) * (m_per_px**2)
        if area_m2 < 30: continue 
        if area_m2 < 100: b_type, pop = 'Kis lakóház', 2.9 * max(1, area_m2/100)
        elif area_m2 < 300: b_type, pop = 'Közepes lakóház', 3.2 * max(1, area_m2/100)
        elif area_m2 < 1000: b_type, pop = 'Nagy lakóház', 4.1 * max(1, area_m2/100)
        else: b_type, pop = 'Társasház', 45 * (max(8, area_m2/80)/10)
        buildings.append({'Típus': b_type, 'Terület (m2)': round(area_m2, 1), 'Becsült lakosság': round(pop, 1)})
    return buildings

# --- UI ---
st.set_page_config(page_title="Lakosság AI (PyTorch)", layout="wide")
st.title("🛰️ Lakosság AI - Műholdas Becslés")
model, device = load_pytorch_model()

# Oldalsáv vezérlők
threshold = st.sidebar.slider("Érzékenység (Threshold)", 0.1, 0.95, 0.7, 0.05)
zoom_level = st.sidebar.select_slider("Zoom szint", options=[18, 19, 20], value=19)

query = st.text_input("Helyszín:", "Budapest, Kálvin tér")

if st.button("Elemzés"):
    loc = ArcGIS().geocode(query)
    if loc:
        n = 2.0 ** zoom_level
        xtile = int((loc.longitude + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(loc.latitude))) / math.pi) / 2.0 * n)
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{ytile}/{xtile}"
        img = Image.open(BytesIO(requests.get(url).content)).convert("RGB").resize((512, 512))
        img_np = np.array(img)

        # --- KRITIKUS JAVÍTÁS: NORMALIZÁLÁS ---
        # Egyszerű [0,1] skálázás, ha a modell nem talált semmit a komplexebb SpaceNet normalizációval
        input_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            output = model(input_t)
            # A képen látható "beégés" ellen emeljük a küszöböt
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask = (prob > threshold).astype(np.uint8)

        data = analyze_buildings(mask, loc.latitude, zoom_level)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            overlay = img_np.copy()
            overlay[mask == 1] = [0, 255, 0]
            st.image(cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0), use_container_width=True)
        with c2:
            st.metric("Lakosság", int(sum(b['Becsült lakosság'] for b in data)))
            st.metric("Épületek", len(data))
            st.dataframe(pd.DataFrame(data))
