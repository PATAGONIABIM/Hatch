
import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Robust", layout="wide")

st.title("HatchCraft: Robust Vectorizer 🛡️")
st.markdown("### Generación de Patrones Sólidos y Conectados")

# Controls
st.sidebar.header("1. Imagen y Escala")
uploaded_file = st.file_uploader("Subir Textura (PNG/JPG)", type=["png", "jpg", "jpeg"])
grid_size = st.sidebar.number_input("Tamaño Base (m)", 1.0, 100.0, 1.0, help="Tamaño de la celda de repetición")
scale_factor = st.sidebar.slider("Escala (% del Original)", 0.2, 5.0, 1.0, 0.1)

st.sidebar.header("2. Reparación de Dibujo")
closing_sz = st.sidebar.slider("Grosor de Unión (Pixels)", 1, 20, 3, help="Aumenta esto para cerrar huecos entre líneas.")
min_area_val = st.sidebar.number_input("Ignorar formas menores a (px²)", 0, 500, 50)

st.sidebar.header("3. Simplificación")
epsilon_val = st.sidebar.slider("Tolerancia (Suavizado)", 0.0001, 0.0200, 0.0020, format="%.4f", help="Menor = Más detalle, Mayor = Líneas rectas.")

if uploaded_file:
    # Generator
    gen = PatternGenerator(grid_width=grid_size, grid_height=grid_size)
    
    # Run
    # Warning: Re-reading stream requires seek(0) if used multiple times, 
    # but here we pass the object once to process_image which reads it.
    uploaded_file.seek(0)
    
    result = gen.process_image(
        uploaded_file, 
        epsilon_factor=epsilon_val,
        scale=scale_factor,
        closing_size=closing_sz,
        min_area=min_area_val
    )
    
    if "error" in result:
        st.error(result["error"])
    else:
        # Diagnostic View
        st.markdown("#### Diagnóstico de Proceso")
        tab1, tab2, tab3 = st.tabs(["1. Imagen Procesada (Unión)", "2. Vectores Detectados", "3. Código .PAT"])
        
        with tab1:
            st.image(result["debug_closed_img"], caption="Paso 1: Dibujo 'Pegado' (Morphological Closing)", use_column_width=True)
            st.info("Si ves el dibujo muy negro/grueso, baja el 'Grosor de Unión'. Si ves huecos, súbelo.")
            
        with tab2:
            st.image(result["vector_img"], caption="Paso 2: Polilíneas Finales", use_column_width=True)
            st.success(result["stats"])
            
            # Visual check for scale?
            # Scale affects the logical coordinates, difficult to show on a fixed image without grid reference.
            st.caption(f"Visualización renderizada en espacio de imagen original. Coordenadas exportadas a escala: x{scale_factor}")
            
        with tab3:
            pat_data = result["pat_content"]
            st.text_area("Previsualización del Archivo", pat_data, height=300)
            st.download_button("📥 Descargar .PAT", pat_data, "robust_pattern.pat", "text/plain")

else:
    st.info("Sube una imagen para comenzar.")
