# ===============================
# STREAMLIT ÉPÜLET ANALIZÁTOR
# ===============================

import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
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
    'kis_lakohaz': [0, 255, 0],      # zöld
    'kozepes_lakohaz': [255, 255, 0], # sárga
    'nagy_lakohaz': [255, 165, 0],    # narancs
    'tarsashaz': [255, 0, 0],         # piros
    'kereskedelmi': [0, 0, 255],      # kék
    'ipari': [128, 0, 128]            # lila
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
# KÉP FELDOLGOZÓ FUNKCIÓK
# ===============================

def preprocess_image(image):
    """Kép előfeldolgozása"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    return img_array

def segment_buildings(mask, min_size=50):
    """Épületek szegmentálása"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
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

def analyze_image(model, image, pixel_to_meter=0.5):
    """Kép elemzése"""
    try:
        # Kép előkészítése
        original_img = preprocess_image(image)
        original_shape = original_img.shape[:2]
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Kép előkészítése...")
        img_resized = cv2.resize(original_img, (256, 256))
        img_input = img_resized.astype(np.float32) / 255.0
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
        status_text.text("✅ Kész!")
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

def create_visualization(result):
    """Vizuális eredmények"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Eredeti kép
    axes[0,0].imshow(result['original_image'])
    axes[0,0].set_title('Eredeti kép')
    axes[0,0].axis('off')
    
    # Szegmentálás
    axes[0,1].imshow(result['segmentation_mask'], cmap='jet')
    axes[0,1].set_title('Épület szegmentálás')
    axes[0,1].axis('off')
    
    # Épületek detektálása
    result_img = result['original_image'].copy()
    
    for building in result['individual_buildings']:
        x, y, w, h = building['bbox']
        color = BUILDING_COLORS.get(building['type'], [255, 255, 255])
        
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        label = f"{building['type']} ({building['population']} fő)"
        cv2.putText(result_img, label, (x, max(y-10, 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    axes[1,0].imshow(result_img)
    axes[1,0].set_title(f'Épületek - Összesen: {result["total_population"]:.0f} fő')
    axes[1,0].axis('off')
    
    # Statisztika
    building_types = [b['type'] for b in result['individual_buildings']]
    if building_types:
        type_counts = {typ: building_types.count(typ) for typ in set(building_types)}
        colors = [BUILDING_COLORS.get(typ, [0.5, 0.5, 0.5]) for typ in type_counts.keys()]
        colors = [[c[0]/255, c[1]/255, c[2]/255] for c in colors]
        
        axes[1,1].bar(type_counts.keys(), type_counts.values(), color=colors)
        axes[1,1].set_title('Épülettípusok eloszlása')
        axes[1,1].set_ylabel('Darabszám')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'Nincs észlelt épület', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Épülettípusok eloszlása')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# ===============================
# FŐ ALKALMAZÁS
# ===============================

def main():
    st.title("🏠 Épület Analizátor")
    st.markdown("Automatikus épület detekció és lakossági becslés")
    
    # Oldalsáv
    with st.sidebar:
        st.header("⚙️ Beállítások")
        
        pixel_to_meter = st.slider(
            "Pixel-méter átváltás",
            0.1, 2.0, 0.5, 0.1,
            help="Egy pixel hány métert reprezentál"
        )
        
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
            help="Támogatott formátumok: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Feltöltött kép", use_column_width=True)
            
            if st.button("🎯 Kép elemzése", type="primary", use_container_width=True):
                model = load_model()
                
                if model is None:
                    st.error("A modell nem tölthető be. Ellenőrizd a final_multi_task_model.h5 fájlt.")
                    return
                
                result = analyze_image(model, image, pixel_to_meter)
                
                if result['success']:
                    with col2:
                        st.subheader("📊 Eredmények")
                        
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
                            st.warning("Nem találhatók épületek")
                        
                        # Vizuális eredmények
                        st.subheader("🖼️ Vizuális elemzés")
                        viz = create_visualization(result)
                        st.image(viz, use_column_width=True)
                        
                        # Export
                        st.subheader("💾 Eredmények mentése")
                        
                        if result['individual_buildings']:
                            df = pd.DataFrame(result['individual_buildings'])
                            csv = df.to_csv(index=False)
                            
                            col_dl1, col_dl2 = st.columns(2)
                            col_dl1.download_button(
                                "📥 CSV letöltése",
                                csv,
                                "epulet_elemzes.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            col_dl2.download_button(
                                "📥 Kép letöltése",
                                viz.getvalue(),
                                "epulet_elemzes.png",
                                "image/png",
                                use_container_width=True
                            )
                else:
                    st.error(f"Hiba: {result['error']}")
    
    if uploaded_file is None:
        with col2:
            st.info("👆 Tölts fel egy képet az elemzéshez")
            
            st.subheader("ℹ️ Használati útmutató")
            st.markdown("""
            1. **Kép feltöltése** - Használj műholdképet
            2. **Beállítás** - Állítsd be a pixel-méter átváltást
            3. **Elemzés** - Kattints az 'Elemzés' gombra
            4. **Eredmények** - Nézd meg a statisztikákat
            
            **Ajánlott beállítások:**
            - Városi területek: 0.3-0.7 pixel/méter
            - Magas felbontású képek
            - Világos, kontrasztos képek
            """)

if __name__ == "__main__":
    main()
