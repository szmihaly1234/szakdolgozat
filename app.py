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

# --- ARCHITEKTÚRA --- (A hibaüzeneted alapján pontosítva)
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

# --- MODEL BETÖLTÉS ---
@st.cache_resource
def load_pytorch_model():
    # Használd a saját Drive ID-dat!
    file_id = "1gZgDnZiX1nTfBLQiqESLFcQzZO5HHrVy" 
    path = "unet_building_segmentation.pth"
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model, device

# --- UI ÉS LOGIKA ---
st.title("🛰️ Lakosság AI (Javított Preprocessz)")
model, device = load_pytorch_model()

# Paraméterek
threshold = st.sidebar.slider("Küszöbérték", 0.1, 0.99, 0.8) # Emelt alapérték a beégés ellen
query = st.text_input("Cím:", "Budapest, Kálvin tér")

if st.button("Elemzés"):
    loc = ArcGIS().geocode(query)
    if loc:
        # Kép letöltés (Zoom 19)
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/19/{int((1.0 - math.asinh(math.tan(math.radians(loc.latitude))) / math.pi) / 2.0 * 2.0**19)}/{int((loc.longitude + 180.0) / 360.0 * 2.0**19)}"
        img = Image.open(BytesIO(requests.get(url).content)).convert("RGB").resize((512, 512))
        img_np = np.array(img)

        # --- A JAVÍTÁS HELYE ---
        # Ha a notebookban nem használtál Normalize() transzformációt, 
        # akkor a modell 0-255 közötti floatokat vár!
        input_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
        # NINCS osztás 255-tel!

        with torch.no_grad():
            output = model(input_t)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask = (prob > threshold).astype(np.uint8)

        # Megjelenítés
        overlay = img_np.copy()
        overlay[mask == 1] = [0, 255, 0]
        st.image(cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0))
        
        # Statisztika
        st.write(f"Talált épület-pixelek aránya: {100 * mask.mean():.2f}%")
