# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
import requests

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
# KÉPFELDOLGOZÁS OPENCV NÉLKÜL
# ===============================

def preprocess_image(image):
    """Kép előfeldolgozása OpenCV nélkül"""
    # Konvertálás numpy array-re
    img_array = np.array(image)
    
    # Ha RGBA, konvertáljuk RGB-re
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Méret változtatás
    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Normalizálás
    img_normalized = np.array(img_resized).astype(np.float32) / 255.0
    
    return img_normalized, img_array

def segment_individual_buildings_numpy(mask, min_size=100):
    """Egyedi épületek szegmentálása OpenCV nélkül"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Egyszerű komponens címkézés numpy-val
    from scipy import ndimage
    
    # Morfológiai műveletek scipy-val
    structure = np.ones((3, 3), dtype=np.uint8)
    binary_mask_cleaned = ndimage.binary_opening(binary_mask, structure=structure)
    binary_mask_cleaned = ndimage.binary_closing(binary_mask_cleaned, structure=structure)
    
    # Címkézés
    labeled_mask, num_features = ndimage.label(binary_mask_cleaned)
    
    individual_buildings = []
    
    for i in range(1, num_features + 1):
        building_mask = (labeled_mask == i).astype(np.uint8)
        area = np.sum(building_mask)
        
        if area < min_size:
            continue
            
        # Bounding box számítás
        rows, cols = np.where(building_mask)
        if len(rows) == 0 or len(cols) == 0:
            continue
            
        y_min, y_max = np.min(rows), np.max(rows)
        x_min, x_max = np.min(cols), np.max(cols)
        w = x_max - x_min
        h = y_max - y_min
        
        # Centroid
        centroid = (np.mean(cols), np.mean(rows))
        
        individual_buildings.append({
            'mask': building_mask,
            'area': area,
            'bbox': (int(x_min), int(y_min), int(w), int(h)),
            'centroid': centroid,
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

def create_annotated_image_pil(original_img, building_analysis, seg_mask):
    """Megjelölt kép létrehozása PIL-lel"""
    # Készítsünk egy másolatot az eredeti képből
    if isinstance(original_img, np.ndarray):
        result_img = Image.fromarray(original_img.astype(np.uint8))
    else:
        result_img = original_img.copy()
    
    draw = ImageDraw.Draw(result_img)
    
    # Szegmentálás overlay
    overlay = Image.new('RGBA', result_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Rajzoljuk meg a szegmentált területeket
    seg_mask_resized = Image.fromarray((seg_mask * 255).astype(np.uint8)).resize(result_img.size)
    seg_array = np.array(seg_mask_resized)
    
    # Piros overlay a szegmentált területekre
    red_overlay = Image.new('RGBA', result_img.size, (255, 0, 0, 64))
    result_img = Image.alpha_composite(result_img.convert('RGBA'), red_overlay)
    draw = ImageDraw.Draw(result_img)
    
    # Bounding box-ok és címkék
    for building in building_analysis:
        x, y, w, h = building['bbox']
        
        # Szín kiválasztása
        color_hex = BUILDING_COLORS[building['type']]
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        
        # Bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color_rgb, width=3)
        
        # Címke háttér
        label = f"{building['type']} ({building['population']} fő)"
        
        # Egyszerűbb címke - csak a szám
        simple_label = f"{building['population']} fő"
        
        # Címke háttér
        bbox = draw.textbbox((x, y - 20), simple_label)
        draw.rectangle(bbox, fill=color_rgb)
        
        # Címke szöveg
        draw.text((x, y - 20), simple_label, fill=(255, 255, 255))
    
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
    
    # Demo kép opció
    use_demo_image = st.sidebar.checkbox("Demo kép használata", value=False)
    
    # Modell betöltése
    model = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ A modell fájl nem található. 
        A demo módban működik az alkalmazás becsült értékekkel.
        """)
        # Demo mód
        model = "demo"
    
    # Kép feltöltése vagy demo
    if use_demo_image:
        st.info("🏠 Demo mód - Becsült értékek használata")
        # Létrehozunk egy demo képet
        demo_image = Image.new('RGB', (400, 300), color=(100, 150, 200))
        draw = ImageDraw.Draw(demo_image)
        
        # Rajzoljunk néhány "épületet"
        draw.rectangle([50, 50, 150, 150], fill=(200, 200, 200), outline=(0, 0, 0), width=2)
        draw.rectangle([200, 80, 300, 180], fill=(180, 180, 180), outline=(0, 0, 0), width=2)
        draw.rectangle([100, 200, 350, 280], fill=(220, 220, 220), outline=(0, 0, 0), width=2)
        
        image = demo_image
        uploaded_file = "demo"
    else:
        uploaded_file = st.file_uploader(
            "📤 Tölts fel egy képet",
            type=['jpg', 'jpeg', 'png'],
            help="Műholdkép vagy légifotó épületekkel"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None
    
    if image is not None:
        # Kép megjelenítése
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Feltöltött kép")
            st.image(image, use_column_width=True)
        
        # Elemzés indítása
        if st.button("🚀 Elemzés indítása", type="primary"):
            with st.spinner("🔍 Kép elemzése folyamatban..."):
                try:
                    if model == "demo":
                        # Demo eredmények
                        building_analysis = [
                            {'id': 1, 'type': 'kis_lakohaz', 'area_m2': 120.5, 'population': 3.5, 'bbox': (50, 50, 100, 100)},
                            {'id': 2, 'type': 'kozepes_lakohaz', 'area_m2': 320.0, 'population': 10.2, 'bbox': (200, 80, 100, 100)},
                            {'id': 3, 'type': 'tarsashaz', 'area_m2': 2500.0, 'population': 112.5, 'bbox': (100, 200, 250, 80)}
                        ]
                        total_population = sum(b['population'] for b in building_analysis)
                        total_area = sum(b['area_m2'] for b in building_analysis)
                        
                        # Demo kép létrehozása
                        annotated_image = create_annotated_image_pil(
                            np.array(image), building_analysis, np.zeros((300, 400))
                        )
                        inference_time = 0.5
                        
                    else:
                        # Valódi modell használata
                        img_processed, original_img = preprocess_image(image)
                        img_input = np.expand_dims(img_processed, axis=0)
                        
                        # Előrejelzés
                        start_time = time.time()
                        seg_pred, class_pred = model.predict(img_input, verbose=0)
                        inference_time = time.time() - start_time
                        
                        # Szegmentálás eredménye
                        original_height, original_width = original_img.shape[:2]
                        seg_mask = np.array(Image.fromarray(seg_pred[0,:,:,0]).resize((original_width, original_height)))
                        
                        # Egyedi épületek szegmentálása
                        individual_buildings = segment_individual_buildings_numpy(seg_mask, min_building_size)
                        
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
                        annotated_image = create_annotated_image_pil(original_img, building_analysis, seg_mask)
                    
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
                    st.info("Próbálj meg egy másik képet feltölteni, vagy használd a demo módot!")

    else:
        # Útmutató
        st.info("""
        ### 📝 Használati útmutató:
        
        1. **Kép feltöltése**: Tölts fel egy műholdképet vagy légifotót épületekkel
        2. **Beállítások**: Állítsd be a pixel/méter arányt az oldalsávban
        3. **Elemzés**: Kattints az "Elemzés indítása" gombra
        4. **Eredmény**: Nézd meg a becsült lakosságot és az épület elemzést
        
        💡 **Tipp**: Ha nincs modell fájl, kapcsold be a "Demo kép használata" opciót!
        """)

if __name__ == "__main__":
    main()
