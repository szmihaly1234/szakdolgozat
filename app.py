import os
import numpy as np
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
# 1. ARCHITEKTÚRA (UNet)
# ===============================
# Ez a rész kritikus: pontosan olyannak kell lennie, mint a notebookodban!
# ===============================
# JAVÍTOTT ARCHITEKTÚRA (A SÚLYOK ALAPJÁN)
# ===============================
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
        # Down part
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        
        # Up part
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
        d1 = self.down1(x)
        p1 = self.pool(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        
        d4 = self.down4(p3)
        p4 = self.pool(d4)
        
        bn = self.bottleneck(p4)
        
        u4 = self.up4(bn)
        c4 = self.conv4(torch.cat([d4, u4], dim=1))
        
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([d3, u3], dim=1))
        
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([d2, u2], dim=1))
        
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([d1, u1], dim=1))
        
        return self.out(c1)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            # Ha a méret nem stimmelne az upsamplingnál
            if x.shape != skip_connection.shape:
                import torch.nn.functional as F
                x = F.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_conv(x)

# ===============================
# 2. KONFIGURÁCIÓ
# ===============================
PT_MODEL_FILE_ID = "IDE_ÍRD_BE_A_DRIVE_ID_T" # Frissítsd!
PT_MODEL_PATH = "unet_model.pth"

# ===============================
# 3. MODELL BETÖLTÉSE
# ===============================
@st.cache_resource
def load_pytorch_model():
    if not os.path.exists(PT_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={PT_MODEL_FILE_ID}"
        gdown.download(url, PT_MODEL_PATH, quiet=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fontos: Ha a notebookodban smp.Unet-et használtál, 
    # akkor itt azt kell példányosítani az osztály helyett!
    model = UNet(in_channels=3, out_channels=1) 
    
    try:
        # map_location kényszeríti a CPU-t, ha nincs GPU a szerveren
        state_dict = torch.load(PT_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Hiba a betöltésnél: {e}")
        st.stop()

# ===============================
# 4. KÉPFELDOLGOZÁS & KERESÉS
# ===============================
def meters_per_pixel(lat, zoom):
    return math.cos(math.radians(lat)) * (2 * math.pi * 6378137.0) / (256 * (2 ** zoom))

def get_satellite_img(query, zoom=19):
    try:
        geolocator = ArcGIS()
        loc = geolocator.geocode(query)
        if not loc: return None, None
        
        lat, lon = loc.latitude, loc.longitude
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        return Image.open(BytesIO(res.content)), (lat, zoom)
    except:
        return None, None

# ===============================
# 5. UI
# ===============================
st.set_page_config(page_title="Building AI (PyTorch)", layout="wide")
st.title("🛰️ Épületszegmentáló - PyTorch Verzió")

model, device = load_pytorch_model()

search_query = st.text_input("Írj be egy címet a műholdkép elemzéséhez:", "Budapest, Hősök tere")

if st.button("Elemzés indítása"):
    with st.spinner("Műholdkép letöltése és feldolgozása..."):
        img, geo_data = get_satellite_img(search_query)
        
        if img:
            # 512x512 RGB feldolgozás
            img_resized = img.resize((512, 512))
            img_np = np.array(img_resized)
            
            # Előkészítés a modellnek (CHW format + Normalizálás)
            input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
                mask = (prob > 0.5).astype(np.uint8)
            
            col1, col2 = st.columns(2)
            col1.image(img_resized, caption="Műholdkép (ArcGIS)")
            
            # Maszk vizualizáció
            overlay = img_np.copy()
            overlay[mask == 1] = [0, 255, 0] # Zöld az épület
            combined = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
            col2.image(combined, caption="Detektált épületek")
            
            # Területbecslés
            lat, zoom = geo_data
            m_per_px = meters_per_pixel(lat, zoom) * (256/512)
            area = np.sum(mask) * (m_per_px ** 2)
            st.success(f"Becsült beépített terület ezen a tile-on: {area:.1f} m²")
        else:
            st.error("Nem sikerült a helyszín beazonosítása.")
