# ===============================
# STREAMLIT ÉPÜLET ANALIZÁTOR - SPACENET OPTIMALIZÁLT
# ===============================

import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io
import time
import os
import pandas as pd

# Streamlit konfiguráció
st.set_page_config(
    page_title="Épület Analizátor",
    page_icon="🏠",
    layout="wide"
)

# ===============================
# KONSTANSOK ÉS BEÁLLÍTÁSOK
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
    'kis_lakohaz': (0, 255, 0),      # zöld
    'kozepes_lakohaz': (255, 255, 0), # sárga
    'nagy_lakohaz': (255, 165, 0),    # narancs
    'tarsashaz': (255, 0, 0),         # piros
    'kereskedelmi': (0, 0, 255),      # kék
    'ipari': (128, 0, 128)            # lila
}

# SpaceNet adatok jellemzői
SPACENET_STATS = {
    'mean': [0.339, 0.324, 0.285],  # SpaceNet RGB csatorna átlagok
    'std': [0.139, 0.125, 0.122]    # SpaceNet RGB csatorna szórások
}

# ===============================
# MODELL BETÖLTÉS
# ===============================

@st.cache_resource(show_spinner=False)
def load_model():
    """Modell betöltése"""
    try:
        # Custom függvények a modellhez
        def dice_coef(y_true, y_pred):
            smooth = 1.0
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def dice_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        # Modell betöltése
        model = tf.keras.models.load_model(
            'final_multi_task_model.h5',
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coef': dice_coef
            },
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Modell betöltési hiba: {e}")
        return None

# ===============================
# SPACENET KOMPATIBILIS KÉPFELDOLGOZÁS
# ===============================

def spacenet_preprocessing(image):
    """SpaceNet adatokhoz igazított kép előfeldolgozás"""
    # Konvertálás numpy array-ré
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Színcsatornák kezelése
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # 1. Kontraszt és élesség javítása (SpaceNet képek jellemzői)
    pil_img = Image.fromarray(img_array)
    
    # Kontraszt növelése
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    # Élesség növelése
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    img_array = np.array(pil_img)
    
    # 2. Hisztogram egyenlítés - jobb kontraszt
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_array = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return img_array

def normalize_for_spacenet(img_array):
    """SpaceNet statisztikák alapján normalizálás"""
    img_float = img_array.astype(np.float32) / 255.0
    
    # SpaceNet átlag és szórás alapján normalizálás
    mean = np.array(SPACENET_STATS['mean'])
    std = np.array(SPACENET_STATS['std'])
    
    img_normalized = (img_float - mean) / std
    return img_normalized

def adjust_image_quality(image, target_brightness=0.6, target_contrast=0.7):
    """Kép minőségének beállítása SpaceNet színjellemzőkhez"""
    img_array = np.array(image)
    
    # Átlagos fényerő és kontraszt számítás
    current_brightness = np.mean(img_array) / 255.0
    current_contrast = np.std(img_array) / 255.0
    
    # Fényerő korrekció
    brightness_ratio = target_brightness / (current_brightness + 1e-7)
    img_adjusted = np.clip(img_array * brightness_ratio, 0, 255).astype(np.uint8)
    
    # Kontraszt korrekció
    if current_contrast < target_contrast:
        # Kontraszt növelése
        alpha = 1.0 + (target_contrast - current_contrast) * 1.5
        img_adjusted = cv2.convertScaleAbs(img_adjusted, alpha=alpha, beta=0)
    
    return Image.fromarray(img_adjusted)

# ===============================
# KÉP FELDOLGOZÓ FUNKCIÓK
# ===============================

def segment_buildings(mask, min_size=50):
    """Épületek szegmentálása"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Morfológiai műveletek - zaj csökkentése
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    buildings = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            continue
            
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        buildings.append({
            'area': area,
            'bbox': (x, y, w, h)
        })
    
    return buildings

def estimate_building_type(area_m2):
    """Épülettípus becslése"""
    if area_m2 < 150:
        return 'kis_lakohaz'
    elif area_m2 < 500:
        return 'kozepes_lakohaz'
    elif area_m2 < 2000:
        return 'nagy_lakohaz'
    else:
        return 'tarsashaz'

def estimate_population(building_type, area):
    """Lakossági becslés"""
    if building_type not in BUILDING_TYPE_POPULATION:
        return 0
        
    base_pop = BUILDING_TYPE_POPULATION[building_type]
    
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        apartments = max(1, area / 100)
        population = base_pop * apartments
    elif building_type == 'tarsashaz':
        apartments = max(8, area / 80)
        population = base_pop * (apartments / 10)
    else:
        population = base_pop
        
    return round(population, 1)

def create_segmentation_visualization(original_img, seg_mask):
    """Szegmentációs maszk vizualizálása"""
    # Szegmentációs maszk színezése
    seg_colored = np.zeros_like(original_img)
    seg_colored[seg_mask > 0.5] = [255, 0, 0]  # Piros szín az épületeknek
    
    # Átlátszó overlay
    alpha = 0.6
    result = cv2.addWeighted(original_img, 1 - alpha, seg_colored, alpha, 0)
    return result

def create_building_visualization(original_img, building_analysis):
    """Épületek vizualizálása bounding box-okkal"""
    result_img = original_img.copy()
    
    for building in building_analysis:
        x, y, w, h = building['bbox']
        color = BUILDING_COLORS.get(building['type'], (255, 255, 255))
        
        # Bounding box rajzolása
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        
        # Címke hátterének rajzolása
        label = f"{building['type']} ({building['population']} fő)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_img, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Címke szöveg
        cv2.putText(result_img, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_img

def analyze_image(model, image, pixel_to_meter=0.5, enhance_quality=True):
    """Kép elemzése SpaceNet kompatibilis módon"""
    try:
        # Kép előkészítése
        if enhance_quality:
            image = adjust_image_quality(image)
        
        original_img = spacenet_preprocessing(image)
        original_shape = original_img.shape[:2]
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Kép előkészítése...")
        img_resized = cv2.resize(original_img, (256, 256))
        
        # SpaceNet normalizálás
        img_input = normalize_for_spacenet(img_resized)
        img_input = np.expand_dims(img_input, axis=0)
        progress_bar.progress(25)
        
        status_text.text("Modell előrejelzés...")
        start_time = time.time()
        seg_pred, _ = model.predict(img_input, verbose=0)
        inference_time = time.time() - start_time
        progress_bar.progress(50)
        
        status_text.text("Eredmények feldolgozása...")
        seg_mask = cv2.resize(seg_pred[0,:,:,0], (original_shape[1], original_shape[0]))
        
        buildings = segment_buildings(seg_mask)
        progress_bar.progress(75)
        
        # Elemzés
        building_analysis = []
        total_population = 0
        
        for i, building in enumerate(buildings):
            area_pixels = building['area']
            area_m2 = area_pixels * (pixel_to_meter ** 2)
            
            building_type = estimate_building_type(area_m2)
            population = estimate_population(building_type, area_m2)
            total_population += population
            
            building_analysis.append({
                'id': i + 1,
                'type': building_type,
                'area_m2': round(area_m2, 1),
                'population': population,
                'bbox': building['bbox']
            })
        
        progress_bar.progress(100)
        status_text.text("✅ Elemzés kész!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return {
            'success': True,
            'original_image': original_img,
            'segmentation_mask': seg_mask,
            'individual_buildings': building_analysis,
            'total_population': total_population,
            'inference_time': inference_time,
            'building_count': len(building_analysis)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ===============================
# FŐ ALKALMAZÁS
# ===============================

def main():
    st.title("🏠 Épület Analizátor - SpaceNet Optimalizált")
    st.markdown("Automatikus épület detekció SpaceNet modelllel")
    
    # Oldalsáv
    with st.sidebar:
        st.header("⚙️ SpaceNet Beállítások")
        
        pixel_to_meter = st.slider(
            "Pixel-méter átváltás",
            0.1, 2.0, 0.5, 0.1,
            help="Egy pixel hány métert reprezentál (SpaceNet: 0.3-0.5)"
        )
        
        enhance_quality = st.checkbox(
            "Képminőség javítása", 
            value=True,
            help="Automatikus kontraszt és fényerő korrekció SpaceNet-hez"
        )
        
        st.markdown("---")
        st.subheader("🎯 SpaceNet Tippek")
        st.markdown("""
        - **Ideális képek**: műholdfelvételek, légi felvételek
        - **Felbontás**: 0.3-1.0 m/pixel
        - **Színek**: természetes RGB
        - **Fényviszonyok**: nappali, felhőtlen
        """)
        
        st.markdown("---")
        st.subheader("🏗️ Épülettípusok")
        
        for building_type, color in BUILDING_COLORS.items():
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            pop = BUILDING_TYPE_POPULATION[building_type]
            
            if pop > 0:
                st.markdown(
                    f"<span style='color:{color_hex}; font-weight:bold'>■</span> "
                    f"{building_type}: {pop} fő/alap",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<span style='color:{color_hex}; font-weight:bold'>■</span> "
                    f"{building_type}: nem lakó",
                    unsafe_allow_html=True
                )
    
    # Fő tartalom
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Kép feltöltése")
        
        uploaded_file = st.file_uploader(
            "Válassz egy képet...",
            type=['jpg', 'jpeg', 'png'],
            help="SpaceNet kompatibilis képek: műhold, légi felvételek"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Kép információk
            st.image(image, caption=f"Feltöltött kép - {image.size[0]}x{image.size[1]}", use_column_width=True)
            
            if st.button("🎯 Kép elemzése", type="primary", use_container_width=True):
                model = load_model()
                
                if model is None:
                    st.error("A modell nem tölthető be. Ellenőrizd a final_multi_task_model.h5 fájlt.")
                    return
                
                result = analyze_image(model, image, pixel_to_meter, enhance_quality)
                
                if result['success']:
                    with col2:
                        st.subheader("📊 SpaceNet Elemzés Eredmények")
                        
                        # Fő metrikák
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Épületek", result['building_count'])
                        col2.metric("Lakosság", f"{result['total_population']:.0f} fő")
                        col3.metric("Idő", f"{result['inference_time']:.2f}s")
                        
                        # Részletes statisztikák
                        st.subheader("📈 Részletes elemzés")
                        
                        if result['individual_buildings']:
                            type_stats = {}
                            for building in result['individual_buildings']:
                                b_type = building['type']
                                if b_type not in type_stats:
                                    type_stats[b_type] = {'count': 0, 'area': 0, 'pop': 0}
                                type_stats[b_type]['count'] += 1
                                type_stats[b_type]['area'] += building['area_m2']
                                type_stats[b_type]['pop'] += building['population']
                            
                            for b_type, stats in type_stats.items():
                                with st.expander(f"🏠 {b_type} ({stats['count']} db)"):
                                    cols = st.columns(3)
                                    cols[0].metric("Darab", stats['count'])
                                    cols[1].metric("Terület", f"{stats['area']:.0f} m²")
                                    cols[2].metric("Lakosság", f"{stats['pop']:.0f} fő")
                        else:
                            st.warning("Nem találhatók épületek. Próbáld meg a képminőség javítását!")
                        
                        # Vizuális eredmények
                        st.subheader("🖼️ SpaceNet Elemzés")
                        
                        # Eredeti kép
                        st.image(result['original_image'], caption="Előfeldolgozott kép", use_column_width=True)
                        
                        # Szegmentációs maszk
                        seg_visual = create_segmentation_visualization(
                            result['original_image'], 
                            result['segmentation_mask']
                        )
                        st.image(seg_visual, caption="Épület szegmentálás (piros = épületek)", use_column_width=True)
                        
                        # Épületek detektálása
                        building_visual = create_building_visualization(
                            result['original_image'], 
                            result['individual_buildings']
                        )
                        st.image(building_visual, 
                                caption=f"Épületek - {result['building_count']} db, {result['total_population']:.0f} fő", 
                                use_column_width=True)
                        
                        # Export
                        st.subheader("💾 Eredmények mentése")
                        
                        if result['individual_buildings']:
                            df = pd.DataFrame(result['individual_buildings'])
                            csv = df.to_csv(index=False)
                            
                            # CSV letöltés
                            st.download_button(
                                "📥 CSV letöltése",
                                csv,
                                "spacenet_elemzes.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                            # Kép letöltése
                            building_visual_pil = Image.fromarray(cv2.cvtColor(building_visual, cv2.COLOR_BGR2RGB))
                            img_buffer = io.BytesIO()
                            building_visual_pil.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Elemzés kép letöltése",
                                img_buffer.getvalue(),
                                "spacenet_elemzes.png",
                                "image/png",
                                use_container_width=True
                            )
                else:
                    st.error(f"Hiba: {result['error']}")
    
    if uploaded_file is None:
        with col2:
            st.info("👆 Tölts fel egy SpaceNet kompatibilis képet")
            
            st.subheader("ℹ️ SpaceNet Optimalizált Használat")
            st.markdown("""
            ### 🎯 SpaceNet Modell Optimalizáció
            
            **Képfeldolgozás javítások:**
            - ✅ **Automatikus kontraszt korrekció**
            - ✅ **Fényerő beállítás SpaceNet színjellemzőkhez**
            - ✅ **Hisztogram egyenlítés jobb kontrasztért**
            - ✅ **Normalizálás SpaceNet statisztikák alapján**
            
            **Ajánlott beállítások:**
            - **Pixel-méter**: 0.3-0.7 (SpaceNet tipikus érték)
            - **Képminőség javítás**: Bekapcsolva
            - **Képtípus**: Műhold, légi felvételek
            - **Felbontás**: Minimum 256x256 pixel
            
            **SpaceNet jellemzők:**
            - Műhold képek 30-50 cm felbontással
            - RGB színes felvételek
            - Városi és külterületi környezet
            - Különböző fényviszonyok
            """)

if __name__ == "__main__":
    main()
