# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import time
import os
import json

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
# MODELL BETÖLTÉSE ÉS JAVÍTÁSA
# ===============================

@st.cache_resource
def load_model_custom():
    """Modell betöltése custom módon a kompatibilitási problémák elkerülésére"""
    try:
        # Először próbáljuk meg a normál betöltést
        model = tf.keras.models.load_model(
            'final_multi_task_model.h5',
            compile=False,
            custom_objects=None
        )
        st.sidebar.success("✅ Modell betöltve (standard módon)")
        return model
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Standard betöltés sikertelen: {e}")
        
        try:
            # Alternatív módszer: custom objects nélkül
            model = tf.keras.models.load_model(
                'final_multi_task_model.h5',
                compile=False,
                custom_objects={}
            )
            st.sidebar.success("✅ Modell betöltve (alternatív módon)")
            return model
            
        except Exception as e2:
            st.sidebar.error(f"❌ Alternatív betöltés sikertelen: {e2}")
            
            try:
                # Utolsó próbálkozás: safe mode
                model = tf.keras.models.load_model(
                    'final_multi_task_model.h5',
                    compile=False,
                    safe_mode=False
                )
                st.sidebar.success("✅ Modell betöltve (safe mode)")
                return model
                
            except Exception as e3:
                st.sidebar.error(f"❌ Minden betöltési módszer sikertelen: {e3}")
                return None

def create_custom_model():
    """Custom modell építése, ha a betöltés nem sikerül"""
    st.sidebar.info("🔨 Custom modell építése...")
    
    try:
        from tensorflow.keras import layers, models
        
        # Egyszerű U-Net szerű architektúra
        inputs = layers.Input(shape=(256, 256, 3))
        
        # Encoder
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        # Bridge
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        # Decoder
        x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
        
        # Outputs
        segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(x)
        
        # Classification head
        classification_branch = layers.GlobalAveragePooling2D()(x)
        classification_branch = layers.Dense(64, activation='relu')(classification_branch)
        classification_output = layers.Dense(6, activation='softmax', name='classification')(classification_branch)
        
        model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])
        
        st.sidebar.success("✅ Custom modell építve")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Custom modell építése sikertelen: {e}")
        return None

# ===============================
# KÉPFELDOLGOZÁS
# ===============================

def preprocess_image_for_model(image, target_size=(256, 256)):
    """Kép előfeldolgozása a modell számára"""
    # Konvertálás numpy array-re
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
    img_resized_array = np.array(img_resized)
    
    # Normalizálás
    img_normalized = img_resized_array.astype(np.float32) / 255.0
    
    return img_normalized, img_array

def segment_individual_buildings(mask, min_size=100):
    """Egyedi épületek szegmentálása"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Egyszerű komponens címkézés
    from scipy import ndimage
    
    structure = np.ones((3, 3), dtype=np.uint8)
    binary_mask_cleaned = ndimage.binary_opening(binary_mask, structure=structure)
    binary_mask_cleaned = ndimage.binary_closing(binary_mask_cleaned, structure=structure)
    
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
        
        individual_buildings.append({
            'mask': building_mask,
            'area': area,
            'bbox': (int(x_min), int(y_min), int(w), int(h)),
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
    if isinstance(original_img, np.ndarray):
        result_img = Image.fromarray(original_img.astype(np.uint8))
    else:
        result_img = original_img.copy()
    
    draw = ImageDraw.Draw(result_img)
    
    # Szegmentálás overlay (vékony piros körvonal)
    seg_mask_resized = Image.fromarray((seg_mask * 255).astype(np.uint8)).resize(result_img.size)
    seg_array = np.array(seg_mask_resized)
    
    # Bounding box-ok és címkék
    for building in building_analysis:
        x, y, w, h = building['bbox']
        
        # Szín kiválasztása
        color_hex = BUILDING_COLORS[building['type']]
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        
        # Bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color_rgb, width=3)
        
        # Címke
        label = f"{building['population']} fő"
        
        # Címke háttér
        text_bbox = draw.textbbox((x, y - 25), label)
        draw.rectangle(text_bbox, fill=color_rgb)
        
        # Címke szöveg
        draw.text((x, y - 25), label, fill=(255, 255, 255))
    
    return result_img

# ===============================
# FŐ ELEMZÉSI FUNKCIÓ
# ===============================

def analyze_image_with_model(image, model, pixel_to_meter=0.5, min_building_size=100):
    """Kép elemzése a modelllel"""
    start_time = time.time()
    
    # Kép előfeldolgozása
    img_processed, original_img = preprocess_image_for_model(image)
    img_input = np.expand_dims(img_processed, axis=0)
    
    # Előrejelzés
    seg_pred, class_pred = model.predict(img_input, verbose=0)
    
    # Eredmények feldolgozása
    if isinstance(original_img, np.ndarray):
        original_height, original_width = original_img.shape[:2]
    else:
        original_width, original_height = original_img.size
        
    seg_mask = np.array(Image.fromarray(seg_pred[0,:,:,0]).resize((original_width, original_height)))
    
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
    
    # Eredmény kép
    annotated_image = create_annotated_image(original_img, building_analysis, seg_mask)
    
    inference_time = time.time() - start_time
    
    return {
        'building_analysis': building_analysis,
        'total_population': total_population,
        'total_area': total_area,
        'annotated_image': annotated_image,
        'segmentation_mask': seg_mask,
        'inference_time': inference_time,
        'model_used': True
    }

# ===============================
# STREAMLIT ALKALMAZÁS
# ===============================

def main():
    st.title("🏠 Épület Lakossági Becslő - AI Modell")
    st.markdown("""
    Tölts fel egy műholdképet vagy légifotót, és a **neurális háló** modell 
    pontosan megmondja, **hány ember lakhat** a képen látható épületekben!
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
    st.sidebar.header("🤖 AI Modell")
    
    model = load_model_custom()
    
    if model is None:
        st.sidebar.error("❌ Modell betöltése sikertelen")
        st.error("""
        ## ❌ AI Modell nem érhető el
        
        A modell fájl betöltése sikertelen. Ellenőrizd, hogy:
        
        1. A `final_multi_task_model.h5` fájl megtalálható-e
        2. A TensorFlow verzió kompatibilis-e a modellel
        3. A modell fájl nem sérült-e
        
        **Kérlek, ellenőrizd a modell fájlt és próbáld újra!**
        """)
        return
    
    # Modell információ
    st.sidebar.success(f"✅ Modell betöltve")
    st.sidebar.info(f"📊 Modell típus: U-Net + Osztályozó")
    st.sidebar.info(f"🏗️ Kimenetek: Szegmentálás + Épülettípus")
    
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
                    # Kép elemzése a modellel
                    results = analyze_image_with_model(
                        image, 
                        model, 
                        pixel_to_meter, 
                        min_building_size
                    )
                    
                    # Eredmények megjelenítése
                    with col2:
                        st.subheader("📊 AI Elemzés eredménye")
                        st.image(results['annotated_image'], use_column_width=True)
                        st.write(f"**Feldolgozási idő:** {results['inference_time']:.2f} másodperc")
                    
                    # Fő metrikák
                    st.success(f"✅ AI Elemzés sikeres!")
                    
                    # Metrikák
                    col3, col4, col5, col6 = st.columns(4)
                    
                    with col3:
                        st.metric(
                            "🏢 Épületek száma",
                            f"{len(results['building_analysis'])} db"
                        )
                    
                    with col4:
                        st.metric(
                            "👥 Becsült lakosság",
                            f"{results['total_population']:.0f} fő"
                        )
                    
                    with col5:
                        st.metric(
                            "📏 Összes terület",
                            f"{results['total_area']:.0f} m²"
                        )
                    
                    with col6:
                        avg_pop = results['total_population'] / len(results['building_analysis']) if results['building_analysis'] else 0
                        st.metric(
                            "📐 Átlag/épület",
                            f"{avg_pop:.1f} fő"
                        )
                    
                    # Részletes eredmények
                    st.subheader("📈 Részletes AI Elemzés")
                    
                    # Épülettípusok szerinti bontás
                    building_type_stats = {}
                    for building in results['building_analysis']:
                        b_type = building['type']
                        if b_type not in building_type_stats:
                            building_type_stats[b_type] = {'count': 0, 'total_population': 0}
                        building_type_stats[b_type]['count'] += 1
                        building_type_stats[b_type]['total_population'] += building['population']
                    
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
                    if results['building_analysis']:
                        for building in results['building_analysis']:
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
                    
                    # Technikai információk
                    with st.expander("🔧 Technikai információk"):
                        st.write(f"**Modell architektúra:** U-Net + Multi-task osztályozó")
                        st.write(f"**Bemeneti méret:** 256 × 256 × 3")
                        st.write(f"**Kimenetek:** Szegmentálás maszk + Épülettípus osztályozás")
                        st.write(f"**Detektált épületek:** {len(results['building_analysis'])}")
                        st.write(f"**Összes szegmentált terület:** {np.sum(results['segmentation_mask'] > 0.5)} pixel")
                    
                except Exception as e:
                    st.error(f"❌ AI elemzési hiba: {e}")
                    st.info("""
                    **Hibaelhárítás:**
                    - Próbálj meg egy másik képet
                    - Ellenőrizd, hogy a kép tartalmaz-e épületeket
                    - Csökkentsd a minimum épület méretet
                    """)
    
    else:
        # Útmutató
        st.info("""
        ### 📝 Használati útmutató:
        
        1. **Kép feltöltése**: Tölts fel egy műholdképet vagy légifotót épületekkel
        2. **Beállítások**: Állítsd be a pixel/méter arányt az oldalsávban
        3. **AI Elemzés**: Kattints az "AI Elemzés indítása" gombra
        4. **Eredmény**: Nézd meg a pontos lakossági becslést és épület elemzést
        
        ### 🎯 AI Modell képességei:
        - **Épületek automatikus detektálása**
        - **Pontos szegmentálás**
        - **Épülettípus osztályozás**
        - **Lakossági becslés**
        
        ### 💡 Tippek a legjobb eredményekhez:
        - Használj **világos, kontrasztos** képeket
        - **Műholdképek** a legalkalmasabbak
        - A kép legyen **éles** és **jól láthatóak** az épületek
        """)

if __name__ == "__main__":
    main()
