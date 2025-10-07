# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import time
import os
import sys

# ===============================
# ALAPBEÁLLÍTÁSOK
# ===============================

st.set_page_config(
    page_title="🏠 Épület Lakossági Becslő",
    page_icon="🏠",
    layout="wide"
)

# Debug info
st.sidebar.write(f"🐍 Python: {sys.version}")
st.sidebar.write(f"🤖 TensorFlow: {tf.__version__}")

# ===============================
# LAKOSSÁGI ADATOK
# ===============================

BUILDING_TYPE_POPULATION = {
    'kis_lakohaz': 2.9,
    'kozepes_lakohaz': 3.2,
    'nagy_lakohaz': 4.1,
    'tarsashaz': 45,
    'kereskedelmi': 0,
    'ipari': 0
}

BUILDING_COLORS = {
    'kis_lakohaz': '#00FF00',
    'kozepes_lakohaz': '#FFFF00', 
    'nagy_lakohaz': '#FFA500',
    'tarsashaz': '#FF0000',
    'kereskedelmi': '#0000FF',
    'ipari': '#800080'
}

BUILDING_LABELS = {
    'kis_lakohaz': 'Kis lakóház (<150 m²)',
    'kozepes_lakohaz': 'Közepes lakóház (150-500 m²)',
    'nagy_lakohaz': 'Nagy lakóház (500-2000 m²)',
    'tarsashaz': 'Társasház (>2000 m²)',
    'kereskedelmi': 'Kereskedelmi épület',
    'ipari': 'Ipari épület'
}

# ===============================
# MODELL BETÖLTÉSE
# ===============================

@st.cache_resource
def load_model():
    """Modell betöltése"""
    model_paths = [
        'final_multi_task_model.h5',
        './models/final_multi_task_model.h5',
        'https://github.com/felhasznalo/szakdolgozat/raw/main/final_multi_task_model.h5'
    ]
    
    for model_path in model_paths:
        try:
            if model_path.startswith('http'):
                # URL-ről történő letöltés
                import requests
                st.info(f"📥 Modell letöltése: {model_path}")
                response = requests.get(model_path)
                with open('temp_model.h5', 'wb') as f:
                    f.write(response.content)
                model = tf.keras.models.load_model('temp_model.h5', compile=False)
                os.remove('temp_model.h5')
            else:
                # Lokális fájl
                model = tf.keras.models.load_model(model_path, compile=False)
            
            st.sidebar.success(f"✅ Modell betöltve: {model_path}")
            return model
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ {model_path}: {str(e)[:100]}...")
            continue
    
    st.sidebar.error("❌ Egyik modell sem tölthető be")
    return None

# ===============================
# KÉPFELDOLGOZÁS
# ===============================

def preprocess_image(image, target_size=(256, 256)):
    """Kép előfeldolgozása"""
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image)
    
    # RGBA -> RGB konverzió
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Méret változtatás
    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
    img_array_resized = np.array(img_resized)
    
    # Normalizálás
    img_normalized = img_array_resized.astype(np.float32) / 255.0
    
    return img_normalized, img_array

def segment_buildings(mask, min_size=100):
    """Épületek szegmentálása"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    try:
        from scipy import ndimage
        
        structure = np.ones((3, 3), dtype=np.uint8)
        binary_mask_cleaned = ndimage.binary_opening(binary_mask, structure=structure)
        binary_mask_cleaned = ndimage.binary_closing(binary_mask_cleaned, structure=structure)
        
        labeled_mask, num_features = ndimage.label(binary_mask_cleaned)
        
        buildings = []
        
        for i in range(1, num_features + 1):
            building_mask = (labeled_mask == i).astype(np.uint8)
            area = np.sum(building_mask)
            
            if area < min_size:
                continue
                
            rows, cols = np.where(building_mask)
            if len(rows) == 0:
                continue
                
            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)
            w, h = x_max - x_min, y_max - y_min
            
            buildings.append({
                'area': area,
                'bbox': (int(x_min), int(y_min), int(w), int(h))
            })
        
        return buildings
        
    except ImportError:
        # Egyszerű fallback
        buildings = []
        area = np.sum(binary_mask)
        if area > min_size:
            buildings.append({
                'area': area,
                'bbox': (50, 50, 100, 100)  # Default bounding box
            })
        return buildings

# ===============================
# FŐ ALKALMAZÁS
# ===============================

def main():
    st.title("🏠 Épület Lakossági Becslő - Python 3.12")
    st.markdown("""
    **Streamlit Lite + Python 3.12 + TensorFlow 2.15**
    
    Tölts fel egy képet, és az AI modell pontosan elemezi az épületeket!
    """)
    
    # Modell betöltése
    model = load_model()
    
    if model is None:
        st.error("""
        ❌ **AI Modell nem érhető el**
        
        Ellenőrizd, hogy:
        1. A `final_multi_task_model.h5` fájl megtalálható-e
        2. A modell kompatibilis-e TensorFlow 2.15-tel
        """)
        
        # Demo mód
        st.warning("🔄 Demo mód aktív")
        model = "demo"
    
    # Beállítások
    st.sidebar.header("⚙️ Beállítások")
    pixel_to_meter = st.sidebar.slider("Pixel/méter arány", 0.1, 2.0, 0.5)
    min_size = st.sidebar.slider("Minimum épület méret", 50, 500, 100)
    
    # Kép feltöltés
    uploaded_file = st.file_uploader("📤 Tölts fel egy képet", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Feltöltött kép")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        if st.button("🚀 AI Elemzés indítása", type="primary"):
            with st.spinner("🤖 Neurális háló dolgozik..."):
                try:
                    if model == "demo":
                        # Demo eredmények
                        results = {
                            'buildings': [
                                {'type': 'kis_lakohaz', 'area_m2': 120, 'population': 3.5, 'bbox': (50, 50, 100, 100)},
                                {'type': 'kozepes_lakohaz', 'area_m2': 320, 'population': 10.2, 'bbox': (200, 80, 150, 120)},
                                {'type': 'tarsashaz', 'area_m2': 2500, 'population': 112.5, 'bbox': (100, 200, 300, 150)}
                            ],
                            'total_population': 126.2,
                            'total_area': 2940
                        }
                    else:
                        # Valódi AI elemzés
                        img_processed, original_img = preprocess_image(image)
                        img_input = np.expand_dims(img_processed, axis=0)
                        
                        # Előrejelzés
                        seg_pred, class_pred = model.predict(img_input, verbose=0)
                        
                        # Eredmények feldolgozása
                        if isinstance(original_img, np.ndarray):
                            h, w = original_img.shape[:2]
                        else:
                            w, h = image.size
                            
                        seg_mask = np.array(Image.fromarray(seg_pred[0,:,:,0]).resize((w, h)))
                        buildings_raw = segment_buildings(seg_mask, min_size)
                        
                        # Épület elemzés
                        results = {'buildings': [], 'total_population': 0, 'total_area': 0}
                        
                        for i, b in enumerate(buildings_raw):
                            area_m2 = b['area'] * (pixel_to_meter ** 2)
                            b_type = 'kis_lakohaz' if area_m2 < 150 else 'kozepes_lakohaz' if area_m2 < 500 else 'nagy_lakohaz' if area_m2 < 2000 else 'tarsashaz'
                            population = estimate_population(b_type, area_m2)
                            
                            results['buildings'].append({
                                'type': b_type,
                                'area_m2': round(area_m2, 1),
                                'population': population,
                                'bbox': b['bbox']
                            })
                            results['total_population'] += population
                            results['total_area'] += area_m2
                    
                    # Eredmények megjelenítése
                    with col2:
                        st.subheader("📊 Elemzés eredménye")
                        
                        # Annotált kép
                        annotated_img = image.copy()
                        draw = ImageDraw.Draw(annotated_img)
                        
                        for b in results['buildings']:
                            x, y, w, h = b['bbox']
                            color = BUILDING_COLORS[b['type']]
                            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                            
                            draw.rectangle([x, y, x+w, y+h], outline=color_rgb, width=3)
                            draw.text((x, y-25), f"{b['population']} fő", fill=color_rgb)
                        
                        st.image(annotated_img, use_column_width=True)
                    
                    # Metrikák
                    st.success(f"✅ {len(results['buildings'])} épület detektálva")
                    
                    col3, col4, col5 = st.columns(3)
                    with col3: st.metric("🏢 Épületek", f"{len(results['buildings'])} db")
                    with col4: st.metric("👥 Lakosság", f"{results['total_population']:.0f} fő")
                    with col5: st.metric("📏 Terület", f"{results['total_area']:.0f} m²")
                    
                    # Részletes lista
                    st.subheader("🏠 Épület lista")
                    for b in results['buildings']:
                        with st.expander(f"{BUILDING_LABELS[b['type']]} - {b['population']} fő"):
                            st.write(f"Terület: {b['area_m2']} m²")
                            st.write(f"Lakosság: {b['population']} fő")
                
                except Exception as e:
                    st.error(f"❌ Hiba: {e}")

def estimate_population(building_type, area):
    """Lakosság becslése"""
    base = BUILDING_TYPE_POPULATION.get(building_type, 0)
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        return base * max(1, area / 100)
    elif building_type == 'tarsashaz':
        return base * max(8, area / 80) / 10
    return base

if __name__ == "__main__":
    main()
