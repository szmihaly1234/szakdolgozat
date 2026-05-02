import os
import numpy as np
import pandas as pd
import cv2
import gdown
import streamlit as st
import tensorflow as tf
import torch  # ÚJ
import torch.nn as nn # ÚJ
from tensorflow.keras import backend as K
from PIL import Image
import time
import math
import requests
from io import BytesIO
from geopy.geocoders import Nominatim, ArcGIS

# ===============================
# 1. KONFIGURÁCIÓ & ID-K
# ===============================

# TensorFlow ID-k (maradnak)
MODEL_FILE_ID = "19Mw_N1ilU58ipoQ6-BdSbPVtHAlSsn2u"
MODEL_PATH = "model.h5"
WEIGHTS_FILE_ID = "1yMIvlRR6mqKLQ46k9Gh-cvGi83mPJnIB" 
WEIGHTS_PATH = "paris_tuned_weights.weights.h5"

# PyTorch ID-k (IDE ÍRD BE A SAJÁT ID-DAT)
PT_MODEL_FILE_ID = "1gZgDnZiX1nTfBLQiqESLFcQzZO5HHrVy" 
PT_MODEL_PATH = "unet_building_segmentation.pth"

# ===============================
# 2. PYTORCH MODEL DEFINÍCIÓ
# ===============================
# Megjegyzés: Itt az UNet osztályodnak szerepelnie kell, 
# hogy a torch.load tudja, mit példányosít.
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Ide másold be a saját UNet __init__ és forward részedet röviden
        # Vagy ha külső fájlban van, importáld be: from my_models import UNet
        pass 

# ===============================
# 3. MODELL BETÖLTÉS (Dinamikus)
# ===============================

@st.cache_resource(show_spinner=False)
def load_any_model(model_type, weights_option=None):
    """
    Betölti a kiválasztott modellt (TF vagy PyTorch)
    """
    if model_type == "PyTorch (Új)":
        ensure_file_from_drive(PT_MODEL_FILE_ID, PT_MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Példányosítás (cseréld a sajátodra)
        model = UNet() 
        try:
            model.load_state_dict(torch.load(PT_MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            return model, "PyTorch UNet", "pt"
        except Exception as e:
            return None, f"PyTorch hiba: {e}", None
            
    else: # TensorFlow ág
        K.clear_session()
        ensure_file_from_drive(MODEL_FILE_ID, MODEL_PATH)
        active_weights = WEIGHTS_PATH if weights_option == "Párizs/EU" else None
        if active_weights:
            ensure_file_from_drive(WEIGHTS_FILE_ID, WEIGHTS_PATH)
            
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
        if active_weights:
            model.load_weights(active_weights)
            return model, "TF Párizsi modell", "tf"
        return model, "TF Globális modell", "tf"

# ===============================
# 4. ÁTDOLGOZOTT ANALÍZIS LOGIKA
# ===============================

def analyze(model, model_framework, image, px_to_m, threshold, overlap, road_sensitivity):
    image_enhanced = enhance_with_clahe(image)
    orig_np = np.array(image_enhanced.convert("RGB"))
    
    # Sliding window hívása (marad a régi, de a belseje okosabb lesz)
    raw_mask = predict_sliding_window_universal(model, model_framework, orig_np, tile_size=320, overlap=overlap)
    
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

def predict_sliding_window_universal(model, framework, full_image, tile_size=320, overlap=0.5):
    h, w, c = full_image.shape
    stride = int(tile_size * (1 - overlap))
    padded_image = cv2.copyMakeBorder(full_image, 0, (tile_size - (h % stride)) % stride, 0, (tile_size - (w % stride)) % stride, cv2.BORDER_REFLECT_101)
    ph, pw, _ = padded_image.shape
    
    full_mask = np.zeros((ph, pw), dtype=np.float32)
    count_mask = np.ones((ph, pw), dtype=np.float32) # count_mask az átlagoláshoz
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            tile = padded_image[y:y+tile_size, x:x+tile_size]
            
            if framework == "tf":
                pre = spacenet_preprocessing(tile)
                pred = model.predict(np.expand_dims(pre, axis=0), verbose=0)[0, :, :, 0]
            else: # PYTORCH ÁG
                # NHWC -> NCHW konverzió és normalizálás
                tile_pt = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                tile_pt = tile_pt.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tile_pt)
                    pred = torch.sigmoid(logits).cpu().numpy()[0, 0, :, :]
            
            full_mask[y:y+tile_size, x:x+tile_size] += pred
            # Itt lehetne count_mask-ot is növelni, ha pontos átlagolás kell

    return full_mask[:h, :w]

# ===============================
# 5. FŐ UI FRISSÍTÉSE
# ===============================

def main():
    st.title("Lakosság AI (Auto-Scale 🛰️)")

    st.sidebar.title("⚙️ Beállítások")
    st.sidebar.subheader("1. AI Modell Kiválasztása")
    
    # Modell típus választó
    engine = st.sidebar.selectbox("Motor:", ["TensorFlow", "PyTorch (Új)"])
    
    if engine == "TensorFlow":
        model_option = st.sidebar.radio("Verzió:", ("Globális", "Párizs/EU"))
    else:
        model_option = "Új verzió"

    with st.spinner("Modell betöltése Driveról..."):
        model, status_msg, framework = load_any_model(engine, model_option)
    
    if not model: 
        st.error(status_msg)
        st.stop()
        
    st.sidebar.success(f"Aktív: {status_msg}")

    # ... (többi UI elem változatlan) ...

    # Elemzés hívásánál átadjuk a framework típusát:
    # run_analysis(model, framework, image, ...) 

# A run_analysis függvényt is frissíteni kell, hogy továbbadja a 'framework' változót!
