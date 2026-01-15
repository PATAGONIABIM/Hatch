
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

# Combined Size control
st.sidebar.markdown("**Dimensiones del Patrón**")
grid_size = st.sidebar.number_input("Tamaño Base (Unidades/Metros)", 1.0, 1000.0, 10.0, help="El tamaño físico que representa TODA la imagen")
# Removed redundant 'Scale' slider or kept it purely as multiplier? 
# User asked for 'Scale' and 'Base Size'. Let's keep them but clarify.
scale_factor = st.sidebar.slider("Factor de Escala", 0.1, 10.0, 1.0, 0.1, help="Multiplicador. Tamaño Final = Base x Escala")

st.sidebar.header("2. Reparación de Dibujo")
closing_sz = st.sidebar.slider("Grosor de Unión (Pixels)", 1, 30, 3, help="Aumenta esto para cerrar huecos entre líneas.")
min_area_val = st.sidebar.number_input("Ignorar formas menores a (px²)", 0, 500, 50)

st.sidebar.header("3. Simplificación")
# Increased Max Epsilon to 0.1 (10%) to ensure it is visible
epsilon_val = st.sidebar.slider("Tolerancia (Suavizado)", 0.0001, 0.1000, 0.0020, format="%.4f", help="Menor = Más detalle, Mayor = Líneas rectas.")

if uploaded_file:
    # Key Fix: Always seek start to avoid empty reads on rerun
    uploaded_file.seek(0)
    
    # Generator
    gen = PatternGenerator(grid_width=grid_size, grid_height=grid_size)
    
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
            st.image(result["vector_img"], caption="Paso 2: Polilíneas Finales (Línea Roja = 1 Unidad)", use_column_width=True)
            st.success(result["stats"])
            st.caption("ℹ️ La línea ROJA muestra '1 Unidad' física. Ajusta 'Tamaño Base' o 'Escala' para cambiarla.")
            
        with tab3:
            pat_data = result["pat_content"]
            st.text_area("Previsualización del Archivo", pat_data, height=300)
            st.download_button("📥 Descargar .PAT", pat_data, "robust_pattern.pat", "text/plain")

else:
    st.info("Sube una imagen para comenzar.")
