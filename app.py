# ===============================
# STREAMLIT ÉPÜLET ANALIZÁTOR ALKALMAZÁS
# Python 3.10 + TensorFlow 2.14 kompatibilis
# ===============================

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import os

# Streamlit oldal beállítások
st.set_page_config(
    page_title="Épület Analizátor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# LAKOSSÁGI ADATOK ÉPÜLETTÍPUSONKÉNT
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
    'kis_lakohaz': [0, 255, 0],      # zöld
    'kozepes_lakohaz': [255, 255, 0], # sárga
    'nagy_lakohaz': [255, 165, 0],    # narancs
    'tarsashaz': [255, 0, 0],         # piros
    'kereskedelmi': [0, 0, 255],      # kék
    'ipari': [128, 0, 128]            # lila
}

# ===============================
# MODELL BETÖLTÉS ÉS ANALÍZIS
# ===============================

@st.cache_resource(show_spinner=False)
def load_model():
    """Modell betöltése cache-eléssel"""
    try:
        # Dice coefficient és loss függvények definiálása
        def dice_coef(y_true, y_pred):
            smooth = 1.0
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def dice_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        # Modell betöltése custom objektumokkal
        model = tf.keras.models.load_model(
            'final_multi_task_model.h5',
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coef': dice_coef
            }
        )
        return model
    except Exception as e:
        st.error(f"❌ Modell betöltési hiba: {e}")
        return None

def segment_individual_buildings(mask, min_size=100):
    """Egyedi épületek szegmentálása connected components alapján"""
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    individual_buildings = []
    
    for i in range(1, num_labels):  # 0 a háttér
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

def estimate_building_type_from_area(area_m2):
    """Épülettípus becslése terület alapján"""
    if area_m2 < 150:
        return 'kis_lakohaz'
    elif area_m2 < 500:
        return 'kozepes_lakohaz'
    elif area_m2 < 2000:
        return 'nagy_lakohaz'
    else:
        return 'tarsashaz'

def estimate_population_for_building(building_type, area):
    """Lakos szám becslése épülettípus és terület alapján"""
    if building_type not in BUILDING_TYPE_POPULATION:
        return 0
        
    base_population = BUILDING_TYPE_POPULATION[building_type]
    
    # Lakóépületeknél terület alapú pontosítás
    if building_type in ['kis_lakohaz', 'kozepes_lakohaz', 'nagy_lakohaz']:
        estimated_apartments = max(1, area / 100)
        population = base_population * estimated_apartments
    elif building_type == 'tarsashaz':
        estimated_apartments = max(8, area / 80)
        population = base_population * (estimated_apartments / 10)
    else:
        population = base_population
        
    return round(population, 1)

def analyze_image_with_model(model, image, pixel_to_meter=0.5):
    """Kép elemzése a modelllel"""
    try:
        # Kép előkészítése
        original_img = np.array(image)
        if len(original_img.shape) == 2:  # Ha fekete-fehér
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        elif original_img.shape[2] == 4:  # Ha RGBA
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)
        
        original_shape = original_img.shape[:2]
        
        # Átméretezés a modell számára
        img_resized = cv2.resize(original_img, (256, 256))
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # Előrejelzés
        start_time = time.time()
        seg_pred, class_pred = model.predict(img_input, verbose=0)
        inference_time = time.time() - start_time
        
        # Szegmentálás eredménye
        seg_mask = cv2.resize(seg_pred[0,:,:,0], (original_shape[1], original_shape[0]))
        
        # Egyedi épületek szegmentálása
        individual_buildings = segment_individual_buildings(seg_mask)
        
        # Épületenkénti elemzés
        building_analysis = []
        total_population = 0
        
        for i, building in enumerate(individual_buildings):
            # Épület területe (m²-ben)
            area_pixels = building['area']
            area_m2 = area_pixels * (pixel_to_meter ** 2)
            
            # Épülettípus becslése terület alapján
            building_type = estimate_building_type_from_area(area_m2)
            
            # Lakossági becslés
            population = estimate_population_for_building(building_type, area_m2)
            total_population += population
            
            building_analysis.append({
                'id': i + 1,
                'type': building_type,
                'area_m2': round(area_m2, 1),
                'population': population,
                'bbox': building['bbox']
            })
        
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
    """Vizuális eredmények létrehozása"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Eredeti kép
    axes[0,0].imshow(result['original_image'])
    axes[0,0].set_title('Eredeti kép')
    axes[0,0].axis('off')
    
    # Szegmentálás
    axes[0,1].imshow(result['segmentation_mask'], cmap='jet')
    axes[0,1].set_title('Épület szegmentálás')
    axes[0,1].axis('off')
    
    # Épületenkénti osztályozás
    result_img = result['original_image'].copy()
    
    for building in result['individual_buildings']:
        x, y, w, h = building['bbox']
        color = BUILDING_COLORS.get(building['type'], [255, 255, 255])
        
        # Bounding box rajzolása
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # Címke
        label = f"{building['type']} ({building['population']} fő)"
        cv2.putText(result_img, label, (x, max(y-10, 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    axes[1,0].imshow(result_img)
    axes[1,0].set_title(f'Épületenkénti osztályozás - Összesen: {result["total_population"]:.0f} fő')
    axes[1,0].axis('off')
    
    # Statisztika
    building_types = [b['type'] for b in result['individual_buildings']]
    type_counts = {typ: building_types.count(typ) for typ in set(building_types)}
    
    if type_counts:
        colors = [BUILDING_COLORS.get(typ, [0.5, 0.5, 0.5]) for typ in type_counts.keys()]
        # Normalizálás matplotlib számára
        colors = [[c[0]/255, c[1]/255, c[2]/255] for c in colors]
        
        axes[1,1].bar(type_counts.keys(), type_counts.values(), color=colors)
        axes[1,1].set_title('Épülettípusok eloszlása')
        axes[1,1].set_ylabel('Darabszám')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'Nincs épület', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Épülettípusok eloszlása')
    
    plt.tight_layout()
    
    # Kép konvertálása Streamlit számára
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# ===============================
# STREAMLIT FELÜLET
# ===============================

def main():
    # Fejléc
    st.title("🏠 Épület Analizátor - Lakossági Becslés")
    st.markdown("""
    Ez az alkalmazás automatikusan detektálja az épületeket műholdképeken és becsüli a lakosság számát 
    épülettípusonként a KSH adatok alapján.
    """)
    
    # Oldalsáv
    st.sidebar.title("⚙️ Beállítások")
    
    # Pixel-méter átváltás
    pixel_to_meter = st.sidebar.slider(
        "Pixel-méter átváltás",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Egy pixel hány métert reprezentál a valóságban"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Épülettípus információk")
    
    for building_type, population in BUILDING_TYPE_POPULATION.items():
        color = BUILDING_COLORS.get(building_type, [255, 255, 255])
        color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        
        if population > 0:
            st.sidebar.markdown(
                f"<span style='color:{color_hex}; font-weight:bold'>{building_type}</span>: "
                f"{population} fő/alapérték", 
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f"<span style='color:{color_hex}; font-weight:bold'>{building_type}</span>: "
                f"nem lakóépület", 
                unsafe_allow_html=True
            )
    
    # Fő tartalom
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Kép feltöltése")
        uploaded_file = st.file_uploader(
            "Tölts fel egy műholdképet",
            type=['jpg', 'jpeg', 'png'],
            help="A kép tartalmazhat épületeket, preferáltan városi környezetben"
        )
        
        if uploaded_file is not None:
            # Kép megjelenítése
            image = Image.open(uploaded_file)
            st.image(image, caption="Feltöltött kép", use_column_width=True)
            
            # Elemzés gomb
            if st.button("🎯 Kép elemzése", type="primary", use_container_width=True):
                with st.spinner("Modell betöltése és kép elemzése..."):
                    # Modell betöltése
                    model = load_model()
                    
                    if model is None:
                        st.error("A modell nem tölthető be. Ellenőrizze, hogy a 'final_multi_task_model.h5' fájl elérhető-e.")
                        return
                    
                    # Kép elemzése
                    result = analyze_image_with_model(model, image, pixel_to_meter)
                    
                    if result['success']:
                        # Eredmények megjelenítése
                        with col2:
                            st.subheader("📊 Elemzés eredménye")
                            
                            # Statisztikák
                            st.metric("Összes épület", result['building_count'])
                            st.metric("Becsült lakosság", f"{result['total_population']:.0f} fő")
                            st.metric("Feldolgozási idő", f"{result['inference_time']:.2f} másodperc")
                            
                            # Részletes statisztikák
                            st.subheader("📈 Részletes statisztikák")
                            
                            # Épülettípus statisztikák
                            type_stats = {}
                            for building in result['individual_buildings']:
                                b_type = building['type']
                                if b_type not in type_stats:
                                    type_stats[b_type] = {'count': 0, 'total_area': 0, 'total_population': 0}
                                
                                type_stats[b_type]['count'] += 1
                                type_stats[b_type]['total_area'] += building['area_m2']
                                type_stats[b_type]['total_population'] += building['population']
                            
                            for b_type, stats in type_stats.items():
                                color = BUILDING_COLORS.get(b_type, [255, 255, 255])
                                color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                                
                                st.markdown(f"**<span style='color:{color_hex}'>{b_type}</span>**", unsafe_allow_html=True)
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Darabszám", stats['count'])
                                with col_b:
                                    st.metric("Terület", f"{stats['total_area']:.0f} m²")
                                with col_c:
                                    st.metric("Lakosság", f"{stats['total_population']:.0f} fő")
                            
                            # Vizuális eredmények
                            st.subheader("🖼️ Vizuális elemzés")
                            visualization = create_visualization(result)
                            st.image(visualization, caption="Részletes elemzés eredménye", use_column_width=True)
                            
                            # Eredmények letöltése
                            st.subheader("💾 Eredmények exportálása")
                            
                            # CSV export
                            import pandas as pd
                            df = pd.DataFrame(result['individual_buildings'])
                            csv = df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Eredmények letöltése (CSV)",
                                data=csv,
                                file_name="epulet_elemzes.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Kép export
                            st.download_button(
                                label="📥 Vizuális elemzés letöltése (PNG)",
                                data=visualization,
                                file_name="epulet_elemzes.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    else:
                        st.error(f"Elemzési hiba: {result['error']}")
    
    # Információk, ha nincs feltöltött kép
    if uploaded_file is None:
        with col2:
            st.info("👆 Kérjük, töltsön fel egy képet az elemzéshez")
            
            st.subheader("ℹ️ Használati útmutató")
            st.markdown("""
            1. **Kép feltöltése**: A bal oldali panelen töltsön fel egy műholdképet
            2. **Beállítások**: Állítsa be a pixel-méter átváltási arányt
            3. **Elemzés**: Kattintson a 'Kép elemzése' gombra
            4. **Eredmények**: Nézze meg a jobb oldali panelen az eredményeket
            
            **Ajánlott képtípusok:**
            - Műholdképek városi területekről
            - JPG vagy PNG formátum
            - Minimum felbontás: 256x256 pixel
            - Világos, kontrasztos képek
            """)
            
            st.subheader("🎯 Épülettípusok")
            for building_type, color in BUILDING_COLORS.items():
                color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                st.markdown(
                    f"<span style='color:{color_hex}; font-weight:bold'>■</span> {building_type}",
                    unsafe_allow_html=True
                )

# Alkalmazás indítása
if __name__ == "__main__":
    main()
