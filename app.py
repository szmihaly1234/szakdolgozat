# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# ===============================
# ALAPBEÁLLÍTÁSOK
# ===============================

st.set_page_config(
    page_title="🏠 Épület Lakossági Becslő",
    page_icon="🏠",
    layout="wide"
)

# ===============================
# MODELL BETÖLTÉSE
# ===============================

@st.cache_resource
def load_model():
    """Modell betöltése cache-eléssel"""
    try:
        model = tf.keras.models.load_model('final_multi_task_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Modell betöltési hiba: {e}")
        return None

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
    'kis_lakohaz': '#00FF00',      # zöld
    'kozepes_lakohaz': '#FFFF00',  # sárga
    'nagy_lakohaz': '#FFA500',     # narancs
    'tarsashaz': '#FF0000',        # piros
    'kereskedelmi': '#0000FF',     # kék
    'ipari': '#800080'             # lila
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
# FŐ FELDOLGOZÓ FUNKCIÓK
# ===============================

def preprocess_image(image):
    """Kép előfeldolgozása"""
    # Konvertálás numpy array-re
    img_array = np.array(image)
    
    # RGB-re konvertálás ha szükséges
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Méret változtatás
    img_resized = cv2.resize(img_array, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Normalizálás
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized, img_array

def segment_individual_buildings(mask, min_size=100):
    """Egyedi épületek szegmentálása"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Morfológiai műveletek
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    individual_buildings = []
    
    for i in range(1, num_labels):
        building_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_size:
            continue
            
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        individual_buildings.append({
            'mask': building_mask,
            'area': area,
            'bbox': (x, y, w, h),
            'centroid': centroids[i],
            'label': i
        })
    
    return individual_buildings

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

def estimate_population_for_building(building_type, area):
    """Lakos szám becslése"""
    if building_type not in BUILDING_TYPE_POPULATION:
        return 0
        
    base_population = BUILDING_TYPE_POPULATION[building_type]
    
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        estimated_apartments = max(1, area / 100)
        population = base_population * estimated_apartments
    elif building_type == 'tarsashaz':
        estimated_apartments = max(8, area / 80)
        population = base_population * (estimated_apartments / 10)
    else:
        population = base_population
        
    return round(population, 1)

def create_annotated_image(original_img, building_analysis, seg_mask):
    """Megjelölt kép létrehozása"""
    result_img = original_img.copy()
    
    # Szegmentálás overlay
    seg_colored = np.zeros_like(original_img)
    seg_colored[seg_mask > 0.5] = [255, 0, 0]  # piros
    result_img = cv2.addWeighted(result_img, 0.7, seg_colored, 0.3, 0)
    
    # Bounding box-ok és címkék
    for building in building_analysis:
        x, y, w, h = building['bbox']
        
        # Szín kiválasztása
        color_name = BUILDING_COLORS[building['type']]
        color_rgb = tuple(int(color_name[i:i+2], 16) for i in (1, 3, 5))
        
        # Bounding box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color_rgb, 3)
        
        # Címke háttér
        label = f"{building['type']} ({building['population']} fő)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_img, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color_rgb, -1)
        
        # Címke szöveg
        cv2.putText(result_img, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_img

# ===============================
# STREAMLIT ALKALMAZÁS
# ===============================

def main():
    st.title("🏠 Épület Lakossági Becslő")
    st.markdown("""
    Tölts fel egy műholdképet vagy légifotót, és az AI modell megmondja, 
    **hány ember lakhat** a képen látható épületekben!
    """)
    
    # Oldalsáv beállítások
    st.sidebar.header("⚙️ Beállítások")
    
    pixel_to_meter = st.sidebar.slider(
        "Pixel/méter arány",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        help="Mennyi méter egy pixel a képen"
    )
    
    min_building_size = st.sidebar.slider(
        "Minimum épület méret (px)",
        min_value=50,
        max_value=500,
        value=100,
        help="A kisebb objektumok figyelmen kívül maradnak"
    )
    
    # Modell betöltése
    model = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ A modell fájl nem található. 
        Győződj meg róla, hogy a `final_multi_task_model.h5` fájl megtalálható a mappában.
        """)
        return
    
    # Kép feltöltése
    uploaded_file = st.file_uploader(
        "📤 Tölts fel egy képet",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Műholdkép vagy légifotó épületekkel"
    )
    
    if uploaded_file is not None:
        # Kép megjelenítése
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Feltöltött kép")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Elemzés indítása
        if st.button("🚀 Elemzés indítása", type="primary"):
            with st.spinner("🔍 Kép elemzése folyamatban..."):
                try:
                    # Kép előfeldolgozása
                    img_processed, original_img = preprocess_image(image)
                    img_input = np.expand_dims(img_processed, axis=0)
                    
                    # Előrejelzés
                    start_time = time.time()
                    seg_pred, class_pred = model.predict(img_input, verbose=0)
                    inference_time = time.time() - start_time
                    
                    # Szegmentálás eredménye
                    original_height, original_width = original_img.shape[:2]
                    seg_mask = cv2.resize(seg_pred[0,:,:,0], (original_width, original_height))
                    
                    # Egyedi épületek szegmentálása
                    individual_buildings = segment_individual_buildings(seg_mask, min_building_size)
                    
                    # Épületenkénti elemzés
                    building_analysis = []
                    total_population = 0
                    total_area = 0
                    
                    for i, building in enumerate(individual_buildings):
                        area_pixels = building['area']
                        area_m2 = area_pixels * (pixel_to_meter ** 2)
                        
                        building_type = classify_building_by_area(area_m2)
                        population = estimate_population_for_building(building_type, area_m2)
                        
                        total_population += population
                        total_area += area_m2
                        
                        building_analysis.append({
                            'id': i + 1,
                            'type': building_type,
                            'area_m2': round(area_m2, 1),
                            'population': population,
                            'bbox': building['bbox']
                        })
                    
                    # Eredmény kép generálása
                    annotated_image = create_annotated_image(original_img, building_analysis, seg_mask)
                    
                    # Statisztikák számítása
                    building_type_stats = {}
                    for building in building_analysis:
                        b_type = building['type']
                        if b_type not in building_type_stats:
                            building_type_stats[b_type] = {'count': 0, 'total_population': 0}
                        building_type_stats[b_type]['count'] += 1
                        building_type_stats[b_type]['total_population'] += building['population']
                    
                    # Eredmények megjelenítése
                    with col2:
                        st.subheader("📊 Elemzés eredménye")
                        st.image(annotated_image, use_column_width=True)
                    
                    # Fő metrikák
                    st.success(f"✅ Elemzés kész! Feldolgozási idő: {inference_time:.2f} másodperc")
                    
                    # Metrikák
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        st.metric(
                            "🏢 Épületek száma",
                            f"{len(building_analysis)} db"
                        )
                    
                    with col4:
                        st.metric(
                            "👥 Becsült lakosság",
                            f"{total_population:.0f} fő"
                        )
                    
                    with col5:
                        st.metric(
                            "📏 Összes terület",
                            f"{total_area:.0f} m²"
                        )
                    
                    with col6:
                        avg_pop = total_population / len(building_analysis) if building_analysis else 0
                        st.metric(
                            "📐 Átlag/épület",
                            f"{avg_pop:.1f} fő"
                        )
                    
                    # Részletes eredmények
                    st.subheader("📈 Részletes elemzés")
                    
                    # Épülettípusok szerinti bontás
                    if building_type_stats:
                        types_col, pop_col = st.columns(2)
                        
                        with types_col:
                            st.write("**Épülettípusok eloszlása:**")
                            for b_type, stats in building_type_stats.items():
                                color = BUILDING_COLORS[b_type]
                                label = BUILDING_LABELS[b_type]
                                st.markdown(
                                    f"<span style='color:{color}; font-weight:bold'>■</span> "
                                    f"{label}: {stats['count']} db",
                                    unsafe_allow_html=True
                                )
                        
                        with pop_col:
                            st.write("**Lakosság eloszlása:**")
                            for b_type, stats in building_type_stats.items():
                                color = BUILDING_COLORS[b_type]
                                label = BUILDING_LABELS[b_type]
                                st.markdown(
                                    f"<span style='color:{color}; font-weight:bold'>■</span> "
                                    f"{label}: {stats['total_population']:.1f} fő",
                                    unsafe_allow_html=True
                                )
                    
                    # Épület lista
                    st.subheader("🏠 Épület lista")
                    if building_analysis:
                        for building in building_analysis:
                            color = BUILDING_COLORS[building['type']]
                            label = BUILDING_LABELS[building['type']]
                            
                            with st.expander(f"Épület {building['id']} - {label}"):
                                col7, col8, col9 = st.columns(3)
                                with col7:
                                    st.write(f"**Terület:** {building['area_m2']} m²")
                                with col8:
                                    st.write(f"**Lakosság:** {building['population']} fő")
                                with col9:
                                    st.write(f"**Típus:** {label}")
                    
                    # Színjelmagyarázat
                    st.subheader("🎨 Jelmagyarázat")
                    legend_cols = st.columns(3)
                    color_items = list(BUILDING_LABELS.items())
                    
                    for i, (b_type, label) in enumerate(color_items):
                        with legend_cols[i % 3]:
                            color = BUILDING_COLORS[b_type]
                            st.markdown(
                                f"<span style='color:{color}; font-size:20px'>■</span> "
                                f"**{label}**",
                                unsafe_allow_html=True
                            )
                    
                except Exception as e:
                    st.error(f"❌ Hiba történt az elemzés során: {e}")
                    st.info("Próbálj meg egy másik képet feltölteni!")

    else:
        # Útmutató
        st.info("""
        ### 📝 Használati útmutató:
        
        1. **Kép feltöltése**: Tölts fel egy műholdképet vagy légifotót épületekkel
        2. **Beállítások**: Állítsd be a pixel/méter arányt az oldalsávban
        3. **Elemzés**: Kattints az "Elemzés indítása" gombra
        4. **Eredmény**: Nézd meg a becsült lakosságot és az épület elemzést
        
        ### 💡 Tippek:
        - A legjobb eredményekért használj **világos, kontrasztos** képeket
        - **Műholdképek** a legalkalmasabbak
        - A pixel/méter arányt a kép felbontása alapján állítsd be
        """)

if __name__ == "__main__":
    main()
