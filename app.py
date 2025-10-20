# ===============================
# STREAMLIT ÉPÜLET ANALIZÁTOR - SPACENET + AID RANDOM FOREST
# ===============================

import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import applications
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io
import time
import os
import pandas as pd
import joblib

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
    'magánház': 2.9,
    'társasház': 3.2, 
    'nagy társasház': 4.1,
    'egyéb': 0
}

BUILDING_COLORS = {
    'magánház': (0, 255, 0),      # zöld
    'társasház': (255, 255, 0),    # sárga
    'nagy társasház': (255, 165, 0), # narancs
    'egyéb': (128, 128, 128)       # szürke
}

# SpaceNet adatok jellemzői
SPACENET_STATS = {
    'mean': [0.339, 0.324, 0.285],
    'std': [0.139, 0.125, 0.122]
}

# ===============================
# MODELL BETÖLTÉS - SPACENET + AID RANDOM FOREST
# ===============================

@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    """Szegmentációs modell betöltése"""
    try:
        def dice_coef(y_true, y_pred):
            smooth = 1.0
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def dice_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

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
        st.error(f"Szegmentációs modell betöltési hiba: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_aid_classifier():
    """AID Random Forest osztályozó betöltése"""
    try:
        model_data = joblib.load('aid_building_classifier.pkl')
        
        # Feature extractor újraépítése
        feature_extractor = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        feature_extractor.trainable = False
        feature_extractor.set_weights(model_data['feature_extractor_weights'])
        
        classifier_info = {
            'classifier': model_data['classifier'],
            'feature_extractor': feature_extractor,
            'class_names': model_data['class_names']
        }
        
        st.success("✅ AID Random Forest osztályozó betöltve")
        return classifier_info
    except Exception as e:
        st.warning(f"⚠️ AID osztályozó nem betölthető: {e}")
        return None

# ===============================
# AID RANDOM FOREST OSZTÁLYOZÓ FUNKCIÓK
# ===============================

def predict_building_type_aid(image_array, aid_classifier):
    """Épülettípus előrejelzés AID Random Forest-al"""
    if aid_classifier is None:
        return 'egyéb', 0.0, {}
    
    try:
        feature_extractor = aid_classifier['feature_extractor']
        classifier = aid_classifier['classifier']
        class_names = aid_classifier['class_names']
        
        # Kép előfeldolgozás
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        img = cv2.resize(image_array, (224, 224))
        img = applications.resnet50.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        # Feature extraction
        features = feature_extractor.predict(img, verbose=0)
        
        # Előrejelzés
        prediction_idx = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        
        building_type = class_names[prediction_idx]
        confidence = probabilities[prediction_idx]
        
        # Összes valószínűség
        all_probabilities = {
            class_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return building_type, confidence, all_probabilities
        
    except Exception as e:
        st.warning(f"AID előrejelzési hiba: {e}")
        return 'egyéb', 0.0, {}

def estimate_population_ksh(building_type, area_m2):
    """KSH alapú lakossági becslés"""
    if building_type not in BUILDING_TYPE_POPULATION:
        return 0
    
    base_pop = BUILDING_TYPE_POPULATION[building_type]
    
    if building_type == 'magánház':
        return base_pop
    elif building_type == 'társasház':
        apartments = max(2, area_m2 / 80)  # Átlagos lakásméret
        return round(apartments * base_pop, 1)
    elif building_type == 'nagy társasház':
        apartments = max(8, area_m2 / 70)  # Kisebb lakások
        return round(apartments * base_pop, 1)
    else:
        return 0

# ===============================
# SPACENET KOMPATIBILIS KÉPFELDOLGOZÁS
# ===============================

def spacenet_preprocessing(image):
    """SpaceNet adatokhoz igazított kép előfeldolgozás"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Kontraszt és élesség javítása
    pil_img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    img_array = np.array(pil_img)
    
    # Hisztogram egyenlítés
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_array = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return img_array

def normalize_for_spacenet(img_array):
    """SpaceNet statisztikák alapján normalizálás"""
    img_float = img_array.astype(np.float32) / 255.0
    mean = np.array(SPACENET_STATS['mean'])
    std = np.array(SPACENET_STATS['std'])
    img_normalized = (img_float - mean) / std
    return img_normalized

def adjust_image_quality(image, target_brightness=0.6, target_contrast=0.7):
    """Kép minőségének beállítása"""
    img_array = np.array(image)
    current_brightness = np.mean(img_array) / 255.0
    current_contrast = np.std(img_array) / 255.0
    
    brightness_ratio = target_brightness / (current_brightness + 1e-7)
    img_adjusted = np.clip(img_array * brightness_ratio, 0, 255).astype(np.uint8)
    
    if current_contrast < target_contrast:
        alpha = 1.0 + (target_contrast - current_contrast) * 1.5
        img_adjusted = cv2.convertScaleAbs(img_adjusted, alpha=alpha, beta=0)
    
    return Image.fromarray(img_adjusted)

# ===============================
# KÉP FELDOLGOZÓ FUNKCIÓK - AID INTEGRÁLT
# ===============================

def segment_buildings(mask, min_size=50):
    """Épületek szegmentálása"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
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

def extract_building_patch(original_img, bbox):
    """Épület kivágása a képről AID osztályozáshoz"""
    x, y, w, h = bbox
    margin = 5
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(original_img.shape[1], x + w + margin)
    y_end = min(original_img.shape[0], y + h + margin)
    
    building_patch = original_img[y_start:y_end, x_start:x_end]
    return building_patch

def analyze_image_with_aid(seg_model, aid_classifier, image, pixel_to_meter=0.5, enhance_quality=True, min_confidence=0.5):
    """Kép elemzése SpaceNet + AID Random Forest kombinációval"""
    try:
        # Kép előkészítése
        if enhance_quality:
            image = adjust_image_quality(image)
        
        original_img = spacenet_preprocessing(image)
        original_shape = original_img.shape[:2]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Kép előkészítése...")
        img_resized = cv2.resize(original_img, (256, 256))
        img_input = normalize_for_spacenet(img_resized)
        img_input = np.expand_dims(img_input, axis=0)
        progress_bar.progress(25)
        
        status_text.text("Modell előrejelzés...")
        start_time = time.time()
        seg_pred, _ = seg_model.predict(img_input, verbose=0)
        inference_time = time.time() - start_time
        progress_bar.progress(50)
        
        status_text.text("Eredmények feldolgozása...")
        seg_mask = cv2.resize(seg_pred[0,:,:,0], (original_shape[1], original_shape[0]))
        buildings = segment_buildings(seg_mask)
        progress_bar.progress(75)
        
        # Épületek osztályozása AID Random Forest-al
        building_analysis = []
        total_population = 0
        high_confidence_count = 0
        
        status_text.text("AID épülettípus osztályozás...")
        
        for i, building in enumerate(buildings):
            area_pixels = building['area']
            area_m2 = area_pixels * (pixel_to_meter ** 2)
            
            # Épület kivágása
            building_patch = extract_building_patch(original_img, building['bbox'])
            
            if building_patch.size == 0:
                continue
            
            # AID Random Forest osztályozás
            building_type, confidence, all_probs = predict_building_type_aid(building_patch, aid_classifier)
            
            if confidence >= min_confidence:
                high_confidence_count += 1
            
            # Lakossági becslés
            population = estimate_population_ksh(building_type, area_m2)
            total_population += population
            
            building_analysis.append({
                'id': i + 1,
                'type': building_type,
                'area_m2': round(area_m2, 1),
                'population': population,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'bbox': building['bbox']
            })
            
            # Progress frissítés
            progress = 75 + (i / len(buildings)) * 20
            progress_bar.progress(int(progress))
        
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
            'building_count': len(building_analysis),
            'high_confidence_count': high_confidence_count,
            'aid_used': aid_classifier is not None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_segmentation_visualization(original_img, seg_mask):
    """Szegmentációs maszk vizualizálása"""
    seg_colored = np.zeros_like(original_img)
    seg_colored[seg_mask > 0.5] = [255, 0, 0]
    alpha = 0.6
    result = cv2.addWeighted(original_img, 1 - alpha, seg_colored, alpha, 0)
    return result

def create_building_visualization(original_img, building_analysis):
    """Épületek vizualizálása AID típusokkal"""
    result_img = original_img.copy()
    
    for building in building_analysis:
        x, y, w, h = building['bbox']
        color = BUILDING_COLORS.get(building['type'], (255, 255, 255))
        confidence = building['confidence']
        
        # Bounding box rajzolása
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        
        # Címke
        label = f"{building['type']} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_img, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        cv2.putText(result_img, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_img

# ===============================
# FŐ ALKALMAZÁS - AID INTEGRÁLT
# ===============================

def main():
    st.title("🏠 Épület Analizátor - SpaceNet + AID Random Forest")
    st.markdown("Automatikus épület detekció és típusosztályozás ML modellekkel")
    
    # Modellek betöltése
    seg_model = load_segmentation_model()
    aid_classifier = load_aid_classifier()
    
    # Oldalsáv
    with st.sidebar:
        st.header("⚙️ Elemzési Beállítások")
        
        pixel_to_meter = st.slider(
            "Pixel-méter átváltás",
            0.1, 2.0, 0.5, 0.1,
            help="Egy pixel hány métert reprezentál"
        )
        
        min_confidence = st.slider(
            "Minimum biztonsági szint",
            0.1, 1.0, 0.6, 0.05,
            help="AID osztályozó minimális biztonsági küszöbe"
        )
        
        enhance_quality = st.checkbox(
            "Képminőség javítása", 
            value=True
        )
        
        st.markdown("---")
        st.subheader("🎯 AID Random Forest Info")
        
        if aid_classifier:
            st.success("✅ AID osztályozó aktív")
            st.write(f"Osztályok: {', '.join(aid_classifier['class_names'])}")
        else:
            st.warning("⚠️ AID osztályozó nem elérhető")
            st.write("Csak méret alapú becslés lesz használva")
        
        st.markdown("---")
        st.subheader("🏗️ Épülettípusok")
        
        for building_type, color in BUILDING_COLORS.items():
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            pop = BUILDING_TYPE_POPULATION[building_type]
            
            st.markdown(
                f"<span style='color:{color_hex}; font-weight:bold'>■</span> "
                f"{building_type}: {pop} fő/alap",
                unsafe_allow_html=True
            )
    
    # Fő tartalom
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Kép feltöltése")
        
        uploaded_file = st.file_uploader(
            "Válassz egy képet...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Feltöltött kép - {image.size[0]}x{image.size[1]}", use_column_width=True)
            
            if st.button("🎯 Kép elemzése AID-vel", type="primary", use_container_width=True):
                if seg_model is None:
                    st.error("Szegmentációs modell nem elérhető")
                    return
                
                result = analyze_image_with_aid(
                    seg_model, 
                    aid_classifier, 
                    image, 
                    pixel_to_meter, 
                    enhance_quality,
                    min_confidence
                )
                
                if result['success']:
                    display_aid_results(result)
                else:
                    st.error(f"Hiba: {result['error']}")
    
    if uploaded_file is None:
        with col2:
            st.info("👆 Tölts fel egy képet az AID elemzéshez")
            
            st.subheader("🚀 AID Random Forest Előnyei")
            st.markdown("""
            ### 🎯 **Intelligens Épülettípus Felismerés**
            
            **Méret + Megjelenés alapú:**
            - ✅ **Geometria**: Terület, alak, arányok
            - ✅ **Megjelenés**: Szín, textúra, struktúra  
            - ✅ **Környezet**: Épület környezetének elemzése
            
            **AID Dataset előnyei:**
            - 10,000+ címkézett kép
            - 30 különböző jelenet kategória
            - Random Forest ensemble módszer
            - Magas pontosságú osztályozás
            """)

def display_aid_results(result):
    """AID eredmények megjelenítése"""
    with col2:
        st.subheader("📊 AID Elemzés Eredmények")
        
        # Fő metrikák
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Épületek", result['building_count'])
        col2.metric("Lakosság", f"{result['total_population']:.0f} fő")
        col3.metric("AID Osztályozó", "✅" if result['aid_used'] else "❌")
        
        if result['building_count'] > 0:
            avg_confidence = np.mean([b['confidence'] for b in result['individual_buildings']])
            col4.metric("Átl. biztonság", f"{avg_confidence:.2f}")
        
        # Részletes statisztikák
        st.subheader("📈 Épülettípus Statisztikák")
        
        if result['individual_buildings']:
            type_stats = {}
            for building in result['individual_buildings']:
                if building['confidence'] >= st.session_state.get('min_confidence', 0.6):
                    b_type = building['type']
                    if b_type not in type_stats:
                        type_stats[b_type] = {'count': 0, 'area': 0, 'pop': 0}
                    type_stats[b_type]['count'] += 1
                    type_stats[b_type]['area'] += building['area_m2']
                    type_stats[b_type]['pop'] += building['population']
            
            for b_type, stats in type_stats.items():
                color = BUILDING_COLORS.get(b_type, (128, 128, 128))
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                
                with st.expander(f"<span style='color:{color_hex}'>🏠 {b_type}</span> ({stats['count']} db)", unsafe_allow_html=True):
                    cols = st.columns(4)
                    cols[0].metric("Darab", stats['count'])
                    cols[1].metric("Terület", f"{stats['area']:.0f} m²")
                    cols[2].metric("Lakosság", f"{stats['pop']:.0f} fő")
                    cols[3].metric("Átlag terület", f"{stats['area']/stats['count']:.0f} m²")
        
        # Biztonsági statisztikák
        if result['aid_used']:
            st.subheader("🎯 Osztályozási Biztonság")
            
            high_confidence = [b for b in result['individual_buildings'] if b['confidence'] >= st.session_state.get('min_confidence', 0.6)]
            low_confidence = [b for b in result['individual_buildings'] if b['confidence'] < st.session_state.get('min_confidence', 0.6)]
            
            col_conf1, col_conf2 = st.columns(2)
            col_conf1.metric("Magas biztonságú", f"{len(high_confidence)} db")
            col_conf2.metric("Alacsony biztonságú", f"{len(low_confidence)} db")
        
        # Vizuális eredmények
        st.subheader("🖼️ Elemzési Eredmények")
        
        # Szegmentációs maszk
        seg_visual = create_segmentation_visualization(
            result['original_image'], 
            result['segmentation_mask']
        )
        st.image(seg_visual, caption="Épület szegmentálás", use_column_width=True)
        
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
            
            st.download_button(
                "📥 CSV letöltése",
                csv,
                "aid_elemzes.csv",
                "text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
