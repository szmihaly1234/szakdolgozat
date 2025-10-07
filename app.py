# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import os
import sys

st.set_page_config(page_title="🏠 Epulet Lakossagi Becslo", page_icon="🏠", layout="wide")

# LAKOSSÁGI ADATOK
BUILDING_TYPE_POPULATION = {
    'kis_lakohaz': 2.9, 'kozepes_lakohaz': 3.2, 'nagy_lakohaz': 4.1, 
    'tarsashaz': 45, 'kereskedelmi': 0, 'ipari': 0
}

BUILDING_COLORS = {
    'kis_lakohaz': '#00FF00', 'kozepes_lakohaz': '#FFFF00', 'nagy_lakohaz': '#FFA500',
    'tarsashaz': '#FF0000', 'kereskedelmi': '#0000FF', 'ipari': '#800080'
}

BUILDING_LABELS = {
    'kis_lakohaz': 'Kis lakohaz (<150 m2)', 'kozepes_lakohaz': 'Kozepes lakohaz (150-500 m2)', 
    'nagy_lakohaz': 'Nagy lakohaz (500-2000 m2)', 'tarsashaz': 'Tarsashaz (>2000 m2)',
    'kereskedelmi': 'Kereskedelmi epulet', 'ipari': 'Ipari epulet'
}

# MODELL BETÖLTÉS - TENSORFLOW 2.19 KOMPATIBILIS
def load_model_advanced():
    """Haladó modell betöltés TensorFlow 2.19 kompatibilitással"""
    st.sidebar.header("🔧 Modell Beallitasok")
    
    # Verzió információk
    try:
        import tensorflow as tf
        st.sidebar.info(f"🤖 TensorFlow: {tf.__version__}")
    except:
        st.sidebar.warning("❌ TensorFlow nem elerheto")
        return "demo"
    
    # Kompatibilis modell fájlok keresése
    compatible_models = [
        'final_multi_task_model_compatible.h5',  # Új konvertált modell
        'final_multi_task_model.h5',             # Eredeti modell
        'model_compatible.h5',
        './models/final_multi_task_model_compatible.h5'
    ]
    
    for model_path in compatible_models:
        if os.path.exists(model_path):
            try:
                st.sidebar.info(f"🔄 Modell betoltese: {model_path}")
                
                # Különböző betöltési módszerek
                try:
                    # 1. Standard betöltés
                    model = tf.keras.models.load_model(model_path, compile=False)
                    st.sidebar.success("✅ Modell betoltve!")
                    return model
                except Exception as e1:
                    try:
                        # 2. Custom objects nélkül
                        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
                        st.sidebar.success("✅ Modell betoltve (custom_objects nelkul)")
                        return model
                    except Exception as e2:
                        try:
                            # 3. Safe mode
                            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                            st.sidebar.success("✅ Modell betoltve (safe mode)")
                            return model
                        except Exception as e3:
                            st.sidebar.error(f"❌ Osszes betoltesi modszer sikertelen")
                            continue
                
            except Exception as e:
                st.sidebar.error(f"❌ {model_path}: {str(e)[:100]}...")
                continue
    
    st.sidebar.error("⚠️ Nincs kompatibilis modell!")
    return "demo"

# MODELL FELTÖLTÉS
def handle_model_upload():
    """Modell fájl feltöltés"""
    st.sidebar.header("📤 Modell Feltoltes")
    
    uploaded_model = st.sidebar.file_uploader(
        "Tolts fel modell fajlt",
        type=['h5'],
        help="TensorFlow .h5 modell fajl"
    )
    
    if uploaded_model is not None:
        try:
            # Fájl mentése
            model_path = "uploaded_model.h5"
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getvalue())
            
            file_size = len(uploaded_model.getvalue()) / (1024 * 1024)
            st.sidebar.success(f"✅ Modell feltoltve! ({file_size:.1f} MB)")
            
            # Betöltés
            import tensorflow as tf
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                st.sidebar.success("✅ Uj modell betoltve!")
                return model
            except Exception as e:
                st.sidebar.error(f"❌ Modell betoltesi hiba: {str(e)[:100]}")
                return "demo"
                
        except Exception as e:
            st.sidebar.error(f"❌ Feltoltesi hiba: {str(e)}")
    
    return None

# DEMO MODELL - OKOSABB VERZIÓ
def smart_demo_detection(image, pixel_to_meter=0.5):
    """Okos demo detektálás"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        width, height = image.size
    
    import random
    buildings = []
    
    # Kép méretétől függő számú épület
    num_buildings = max(3, min(8, int((width * height) / 30000)))
    
    for i in range(num_buildings):
        w = random.randint(int(width*0.08), int(width*0.25))
        h = random.randint(int(height*0.08), int(height*0.20))
        x = random.randint(20, width - w - 20)
        y = random.randint(20, height - h - 20)
        
        area_m2 = (w * h) * (pixel_to_meter ** 2)
        building_type = classify_building_by_area(area_m2)
        population = estimate_population_for_building(building_type, area_m2)
        
        buildings.append({
            'id': i + 1,
            'type': building_type,
            'area_m2': round(area_m2, 1),
            'population': population,
            'bbox': (x, y, w, h),
            'confidence': round(random.uniform(0.75, 0.95), 2)
        })
    
    return buildings

# SEGÉDFÜGGVÉNYEK
def classify_building_by_area(area_m2):
    if area_m2 < 150: return 'kis_lakohaz'
    elif area_m2 < 500: return 'kozepes_lakohaz'
    elif area_m2 < 2000: return 'nagy_lakohaz'
    else: return 'tarsashaz'

def estimate_population_for_building(building_type, area):
    if building_type not in BUILDING_TYPE_POPULATION: return 0
    base_population = BUILDING_TYPE_POPULATION[building_type]
    
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        return round(base_population * max(1, area / 100), 1)
    elif building_type == 'tarsashaz':
        return round(base_population * max(8, area / 80) / 10, 1)
    else:
        return base_population

def create_annotated_image(original_img, building_analysis):
    if isinstance(original_img, np.ndarray):
        result_img = Image.fromarray(original_img.astype(np.uint8))
    else:
        result_img = original_img.copy()
    
    draw = ImageDraw.Draw(result_img)
    
    for building in building_analysis:
        x, y, w, h = building['bbox']
        color_hex = BUILDING_COLORS[building['type']]
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        
        draw.rectangle([x, y, x + w, y + h], outline=color_rgb, width=3)
        label = f"{building['population']} fo"
        text_bbox = draw.textbbox((x, y - 25), label)
        draw.rectangle(text_bbox, fill=color_rgb)
        draw.text((x, y - 25), label, fill=(255, 255, 255))
    
    return result_img

# FŐ ELEMZÉSI FUNKCIÓ
def analyze_image_complete(image, model, pixel_to_meter=0.5):
    start_time = time.time()
    
    if model == "demo":
        buildings = smart_demo_detection(image, pixel_to_meter)
        main_building_type = max(set(b['type'] for b in buildings), key=lambda x: sum(1 for b in buildings if b['type'] == x))
        confidence = 0.85
        model_used = False
    else:
        try:
            # Valódi AI elemzés
            img_processed, _ = preprocess_image(image)
            img_input = np.expand_dims(img_processed, axis=0)
            
            seg_pred, class_pred = model.predict(img_input, verbose=0)
            
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                width, height = image.size
                
            seg_mask = np.array(Image.fromarray(seg_pred[0,:,:,0]).resize((width, height)))
            predicted_class = np.argmax(class_pred[0])
            confidence = np.max(class_pred[0])
            
            building_types_dict = {0: "kis_lakohaz", 1: "kozepes_lakohaz", 2: "nagy_lakohaz", 
                                 3: "tarsashaz", 4: "kereskedelmi", 5: "ipari"}
            main_building_type = building_types_dict[predicted_class]
            model_used = True
            
            # Épületek szegmentálása
            buildings = []
            binary_mask = (seg_mask > 0.5).astype(np.uint8)
            
            try:
                from scipy import ndimage
                labeled_mask, num_features = ndimage.label(binary_mask)
                
                for i in range(1, num_features + 1):
                    building_mask = (labeled_mask == i).astype(np.uint8)
                    area = np.sum(building_mask)
                    if area < 100: continue
                    
                    rows, cols = np.where(building_mask)
                    y_min, y_max = np.min(rows), np.max(rows)
                    x_min, x_max = np.min(cols), np.max(cols)
                    w, h = x_max - x_min, y_max - y_min
                    
                    area_m2 = area * (pixel_to_meter ** 2)
                    building_type = classify_building_by_area(area_m2)
                    population = estimate_population_for_building(building_type, area_m2)
                    
                    buildings.append({
                        'id': i, 'type': building_type, 'area_m2': round(area_m2, 1),
                        'population': population, 'bbox': (int(x_min), int(y_min), int(w), int(h)),
                        'confidence': confidence
                    })
            except ImportError:
                # Fallback
                area = np.sum(binary_mask)
                if area > 100:
                    area_m2 = area * (pixel_to_meter ** 2)
                    building_type = classify_building_by_area(area_m2)
                    population = estimate_population_for_building(building_type, area_m2)
                    buildings.append({
                        'id': 1, 'type': building_type, 'area_m2': round(area_m2, 1),
                        'population': population, 'bbox': (50, 50, 100, 100), 'confidence': confidence
                    })
                    
        except Exception as e:
            st.error(f"AI elemzesi hiba: {e}")
            return analyze_image_complete(image, "demo", pixel_to_meter)
    
    # Összesítések
    total_population = sum(b['population'] for b in buildings)
    total_area_m2 = sum(b['area_m2'] for b in buildings)
    annotated_image = create_annotated_image(image, buildings)
    
    return {
        'main_building_type': main_building_type, 'confidence': confidence,
        'building_count': len(buildings), 'individual_buildings': buildings,
        'total_population': round(total_population, 1), 'total_area_m2': round(total_area_m2, 1),
        'annotated_image': annotated_image, 'inference_time': time.time() - start_time,
        'model_used': model_used
    }

# STREAMLIT ALKALMAZÁS
def main():
    st.title("🏠 Epulet Lakossagi Becslo")
    st.markdown("TensorFlow 2.19 kompatibilis verzio")
    
    # Modell betöltés
    model = load_model_advanced()
    
    # Modell feltöltés lehetősége
    uploaded_model = handle_model_upload()
    if uploaded_model is not None:
        model = uploaded_model
    
    # Beállítások
    st.sidebar.header("⚙️ Beallitasok")
    pixel_to_meter = st.sidebar.slider("Pixel/meter arany", 0.1, 2.0, 0.5)
    
    # Modell információ
    if model == "demo":
        st.sidebar.warning("🔸 Demo Mod Aktiv")
        st.sidebar.info("""
        **TensorFlow 2.19 modell kompatibilitasi problema.**
        
        **Megoldasok:**
        1. Modell ujramentese TF 2.13-ra
        2. Modell feltoltese kompatibilis verzioval
        3. Hasznald a demo modot
        """)
    else:
        st.sidebar.success("✅ AI Modell Aktiv!")
    
    # Kép feltöltés
    uploaded_file = st.file_uploader("📤 Tolts fel egy kepet", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feltoltott kep")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.write(f"Meret: {image.size[0]} x {image.size[1]} pixel")
        
        if st.button("🚀 Elemzes inditasa", type="primary"):
            with st.spinner("Elemzes folyamatban..."):
                results = analyze_image_complete(image, model, pixel_to_meter)
                
                with col2:
                    st.subheader("Eredmeny")
                    st.image(results['annotated_image'], use_column_width=True)
                
                if results['model_used']:
                    st.success(f"✅ AI Elemzes kesz! ({results['inference_time']:.2f}s)")
                else:
                    st.info(f"🔸 Okos elemzes kesz! ({results['inference_time']:.2f}s)")
                
                # Metrikák
                col3, col4, col5 = st.columns(3)
                with col3: st.metric("Epuletek", f"{results['building_count']} db")
                with col4: st.metric("Lakossag", f"{results['total_population']} fo")
                with col5: st.metric("Terulet", f"{results['total_area_m2']} m2")
                
                # Részletes információk
                with st.expander("📊 Reszletes statisztikak"):
                    building_stats = {}
                    for building in results['individual_buildings']:
                        b_type = building['type']
                        if b_type not in building_stats:
                            building_stats[b_type] = {'count': 0, 'total_population': 0}
                        building_stats[b_type]['count'] += 1
                        building_stats[b_type]['total_population'] += building['population']
                    
                    for b_type, stats in building_stats.items():
                        color = BUILDING_COLORS[b_type]
                        st.markdown(f"<span style='color:{color}'>■</span> **{BUILDING_LABELS[b_type]}**: {stats['count']} db, {stats['total_population']} fo", unsafe_allow_html=True)
    
    else:
        st.info("""
        ### 🎯 TensorFlow 2.19 Kompatibilitasi Utmutato
        
        **1. Modell konvertalasa (ajanlott):**
        ```python
        # Futtasd a sajat gepeden TF 2.19-el
        import tensorflow as tf
        model = tf.keras.models.load_model('final_multi_task_model.h5')
        model.save('final_multi_task_model_compatible.h5', save_format='h5')
        ```
        
        **2. Kompatibilis modell feltoltese:**
        - Hasznald a bal oldali modell feltolto gombot
        - .h5 formatumu fajt tölts fel
        
        **3. requirements.txt:**
        ```txt
        streamlit==1.28.0
        tensorflow==2.13.0
        numpy==1.24.3
        Pillow==10.0.1
        scipy==1.11.4
        ```
        """)

if __name__ == "__main__":
    main()
