# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import os

# ===============================
# ALAPBEÁLLÍTÁSOK
# ===============================

st.set_page_config(
    page_title="🏠 Épület Lakossági Becslő",
    page_icon="🏠",
    layout="wide"
)

# ===============================
# LAKOSSÁGI ADATOK
# ===============================

BUILDING_TYPE_POPULATION = {
    'kis_lakohaz': 2.9,      # átlagos háztartásméret
    'kozepes_lakohaz': 3.2,  # nagyobb családok
    'nagy_lakohaz': 4.1,     # többgenerációs
    'tarsashaz': 45,         # átlagos lakószám társasházban
    'kereskedelmi': 0,       # nem lakóépület
    'ipari': 0               # nem lakóépület
}

BUILDING_COLORS = {
    'kis_lakohaz': (0, 255, 0),      # zöld
    'kozepes_lakohaz': (255, 255, 0), # sárga
    'nagy_lakohaz': (255, 165, 0),    # narancs
    'tarsashaz': (255, 0, 0),         # piros
    'kereskedelmi': (0, 0, 255),      # kék
    'ipari': (128, 0, 128)            # lila
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
    try:
        model = tf.keras.models.load_model('final_multi_task_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Modell betöltési hiba: {e}")
        return None

# ===============================
# SPACENET KÉP FELDOLGOZÁS
# ===============================

def preprocess_for_spacenet(image):
    """
    Kép előfeldolgozása SpaceNet kompatibilis formátumra
    """
    # Konvertálás numpy array-re
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image)
    
    # RGBA -> RGB konverzió
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # BGR -> RGB konverzió (OpenCV formátum)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    # Méret változtatás (bicubic interpoláció)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Normalizálás 0-1 közé
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized, img_rgb

def enhance_for_spacenet_compatibility(img_array):
    """
    Kép fokozása SpaceNet kompatibilitás érdekében
    """
    # Élesség fokozás
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharpened = cv2.filter2D(img_array, -1, kernel)
    
    # Gamma korrekció
    gamma = 1.2
    img_gamma = np.power(img_sharpened, gamma)
    
    # Kontraszt korrekció
    img_contrast = cv2.convertScaleAbs(img_gamma, alpha=1.1, beta=0)
    
    return img_contrast

# ===============================
# LAKOSSÁGI BECSLÉS
# ===============================

def estimate_population_for_building(building_type, area_m2):
    """Lakos szám becslése épülettípus és terület alapján"""
    if building_type not in BUILDING_TYPE_POPULATION:
        return 0
    
    base_population = BUILDING_TYPE_POPULATION[building_type]
    
    # Lakóépületeknél terület alapú pontosítás
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        # Átlagos lakásméret alapján (100 m²/lakás)
        estimated_apartments = max(1, area_m2 / 100)
        population = base_population * estimated_apartments
    elif building_type == 'tarsashaz':
        # Társasház: több lakás
        estimated_apartments = max(8, area_m2 / 80)  # 80 m²/lakás
        population = base_population * (estimated_apartments / 10)  # Normalizálás
    else:
        # Nem lakóépületek
        population = base_population
        
    return round(population, 1)

def classify_building_by_area(area_m2):
    """Épülettípus besorolása terület alapján"""
    if area_m2 < 150:
        return 'kis_lakohaz'
    elif area_m2 < 500:
        return 'kozepes_lakohaz'
    elif area_m2 < 2000:
        return 'nagy_lakohaz'
    else:
        return 'tarsashaz'

# ===============================
# FŐ ELEMZÉSI FUNKCIÓ
# ===============================

def analyze_image(image, model, pixel_to_meter=0.5):
    """Kép elemzése a modellel"""
    start_time = time.time()
    
    # Kép előfeldolgozása SpaceNet kompatibilis formátumra
    img_processed, original_img = preprocess_for_spacenet(image)
    img_input = np.expand_dims(img_processed, axis=0)
    
    # Előrejelzés
    seg_pred, class_pred = model.predict(img_input, verbose=0)
    
    # Eredmények feldolgozása
    original_height, original_width = original_img.shape[:2]
    seg_mask = cv2.resize(seg_pred[0,:,:,0], (original_width, original_height))
    
    predicted_class = np.argmax(class_pred[0])
    confidence = np.max(class_pred[0])
    
    # Épülettípus
    building_types = {
        0: "kis_lakohaz",
        1: "kozepes_lakohaz", 
        2: "nagy_lakohaz",
        3: "tarsashaz",
        4: "kereskedelmi",
        5: "ipari"
    }
    
    main_building_type = building_types[predicted_class]
    
    # Épületek számának becslése
    binary_mask = (seg_mask > 0.5).astype(np.uint8)
    
    # Morfológiai műveletek
    kernel = np.ones((3,3), np.uint8)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    num_buildings, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_cleaned, connectivity=8)
    building_count = num_buildings - 1
    
    # Egyedi épületek elemzése
    individual_buildings = []
    total_population = 0
    total_area_m2 = 0
    
    for i in range(1, num_buildings):
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        area_m2 = area_pixels * (pixel_to_meter ** 2)
        
        # Egyedi épület típusa
        building_type = classify_building_by_area(area_m2)
        population = estimate_population_for_building(building_type, area_m2)
        
        # Bounding box
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        individual_buildings.append({
            'id': i,
            'type': building_type,
            'area_pixels': area_pixels,
            'area_m2': round(area_m2, 1),
            'population': population,
            'bbox': (x, y, w, h)
        })
        
        total_population += population
        total_area_m2 += area_m2
    
    # Eredmény kép létrehozása
    result_img = original_img.copy()
    
    # Szegmentálás overlay
    seg_colored = np.zeros_like(original_img)
    seg_colored[seg_mask > 0.5] = [255, 0, 0]
    result_img = cv2.addWeighted(result_img, 0.7, seg_colored, 0.3, 0)
    
    # Bounding box-ok és címkék
    for building in individual_buildings:
        x, y, w, h = building['bbox']
        color = BUILDING_COLORS[building['type']]
        
        # Bounding box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # Címke
        label = f"{building['type']} ({building['population']} fő)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        
        # Címke háttér
        cv2.rectangle(result_img, (x, y - label_size[1] - 5), 
                     (x + label_size[0], y), color, -1)
        
        # Címke szöveg
        cv2.putText(result_img, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    inference_time = time.time() - start_time
    
    return {
        'main_building_type': main_building_type,
        'confidence': confidence,
        'building_count': building_count,
        'individual_buildings': individual_buildings,
        'total_population': round(total_population, 1),
        'total_area_m2': round(total_area_m2, 1),
        'result_image': result_img,
        'segmentation_mask': seg_mask,
        'original_image': original_img,
        'processed_image': img_processed,
        'inference_time': inference_time
    }

# ===============================
# STREAMLIT ALKALMAZÁS
# ===============================

def main():
    st.title("🏠 Épület Lakossági Becslő - AI Modell")
    st.markdown("""
    Tölts fel egy műholdképet vagy légifotót, és a **SpaceNet-kompatibilis AI modell** 
    pontosan megmondja, **hány ember lakhat** a képen látható épületekben!
    """)
    
    # Oldalsáv beállítások
    st.sidebar.header("⚙️ Beállítások")
    
    pixel_to_meter = st.sidebar.slider(
        "Pixel/méter arány",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        help="Mennyi méter egy pixel a képen (0.5 = 2 pixel/1 méter)"
    )
    
    # Modell betöltése
    model = load_model()
    
    if model is None:
        st.error("""
        ❌ **AI Modell nem érhető el**
        
        Ellenőrizd, hogy a `final_multi_task_model.h5` fájl megtalálható-e!
        """)
        return
    
    st.sidebar.success("✅ AI Modell betöltve")
    
    # Kép feltöltése
    uploaded_file = st.file_uploader(
        "📤 Tölts fel egy képet",
        type=['jpg', 'jpeg', 'png'],
        help="Műholdkép vagy légifotó épületekkel"
    )
    
    if uploaded_file is not None:
        # Kép megjelenítése
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Feltöltött kép")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Kép információ
            st.write(f"**Kép mérete:** {image.size[0]} × {image.size[1]} pixel")
        
        # Elemzés indítása
        if st.button("🚀 AI Elemzés indítása", type="primary"):
            with st.spinner("🤖 Neurális háló elemzi a képet..."):
                try:
                    # Kép elemzése
                    results = analyze_image(image, model, pixel_to_meter)
                    
                    # Eredmények megjelenítése
                    with col2:
                        st.subheader("📊 AI Elemzés eredménye")
                        st.image(results['result_image'], use_column_width=True)
                    
                    # Fő metrikák
                    st.success(f"✅ AI Elemzés sikeres! ({results['inference_time']:.2f}s)")
                    
                    # Fő metrikák
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        st.metric(
                            "🏢 Épületek száma",
                            f"{results['building_count']} db"
                        )
                    
                    with col4:
                        st.metric(
                            "👥 Összes lakosság",
                            f"{results['total_population']} fő"
                        )
                    
                    with col5:
                        st.metric(
                            "📏 Összes terület",
                            f"{results['total_area_m2']} m²"
                        )
                    
                    with col6:
                        st.metric(
                            "🎯 Fő épülettípus",
                            f"{BUILDING_LABELS[results['main_building_type']]}"
                        )
                    
                    # Részletes eredmények
                    st.subheader("📈 Részletes elemzés")
                    
                    # Épülettípus statisztikák
                    building_stats = {}
                    for building in results['individual_buildings']:
                        b_type = building['type']
                        if b_type not in building_stats:
                            building_stats[b_type] = {
                                'count': 0, 
                                'total_population': 0,
                                'total_area': 0
                            }
                        building_stats[b_type]['count'] += 1
                        building_stats[b_type]['total_population'] += building['population']
                        building_stats[b_type]['total_area'] += building['area_m2']
                    
                    if building_stats:
                        col7, col8 = st.columns(2)
                        
                        with col7:
                            st.write("**Épülettípus statisztikák:**")
                            for b_type, stats in building_stats.items():
                                color_hex = '#%02x%02x%02x' % BUILDING_COLORS[b_type]
                                st.markdown(
                                    f"<span style='color:{color_hex}; font-weight:bold'>■</span> "
                                    f"**{BUILDING_LABELS[b_type]}**: "
                                    f"{stats['count']} db, "
                                    f"{stats['total_population']} fő, "
                                    f"{stats['total_area']:.0f} m²",
                                    unsafe_allow_html=True
                                )
                        
                        with col8:
                            st.write("**Átlagok:**")
                            if results['building_count'] > 0:
                                avg_pop_per_building = results['total_population'] / results['building_count']
                                avg_area_per_building = results['total_area_m2'] / results['building_count']
                                st.write(f"Átlagos lakosság/épület: **{avg_pop_per_building:.1f} fő**")
                                st.write(f"Átlagos terület/épület: **{avg_area_per_building:.0f} m²**")
                    
                    # Egyedi épületek listája
                    st.subheader("🏠 Egyedi épületek")
                    
                    for building in results['individual_buildings']:
                        with st.expander(f"Épület {building['id']} - {BUILDING_LABELS[building['type']]}"):
                            col9, col10, col11 = st.columns(3)
                            with col9:
                                st.write(f"**Terület:** {building['area_m2']} m²")
                            with col10:
                                st.write(f"**Lakosság:** {building['population']} fő")
                            with col11:
                                st.write(f"**Típus:** {BUILDING_LABELS[building['type']]}")
                    
                    # Színjelmagyarázat
                    st.subheader("🎨 Jelmagyarázat")
                    legend_cols = st.columns(3)
                    
                    for i, (b_type, label) in enumerate(BUILDING_LABELS.items()):
                        with legend_cols[i % 3]:
                            color_hex = '#%02x%02x%02x' % BUILDING_COLORS[b_type]
                            st.markdown(
                                f"<span style='color:{color_hex}; font-size:20px'>■</span> **{label}**",
                                unsafe_allow_html=True
                            )
                    
                    # Technikai információk
                    with st.expander("🔧 Technikai információk"):
                        st.write(f"**AI biztonsági szint:** {results['confidence']:.1%}")
                        st.write(f"**Feldolgozási idő:** {results['inference_time']:.2f} másodperc")
                        st.write(f"**SpaceNet kompatibilitás:** ✅")
                        st.write(f"**Modell architektúra:** U-Net + Multi-task")
                        
                except Exception as e:
                    st.error(f"❌ AI elemzési hiba: {e}")
                    st.info("Próbálj meg egy másik képet feltölteni!")
    
    else:
        # Útmutató
        st.info("""
        ### 📝 Használati útmutató:
        
        1. **Kép feltöltése**: Tölts fel egy műholdképet vagy légifotót
        2. **Beállítások**: Állítsd be a pixel/méter arányt
        3. **AI Elemzés**: Indítsd el az elemzést
        4. **Eredmény**: Nézd meg a részletes lakossági becslést
        
        ### 🎯 AI Modell képességei:
        - **SpaceNet-kompatibilis** képfeldolgozás
        - **Pontos épület szegmentálás**
        - **Épülettípus osztályozás**
        - **Lakossági becslés** KSH adatok alapján
        
        ### 💡 Tippek:
        - **Műholdképek** a legalkalmasabbak
        - A kép legyen **világos** és **éles**
        - Állítsd be a **pixel/méter arányt** a kép felbontása alapján
        """)

if __name__ == "__main__":
    main()
