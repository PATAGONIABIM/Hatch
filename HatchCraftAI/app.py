import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Pro v3.4", layout="wide")

st.title("HatchCraft Pro: Zero-Overlap Edition 🧱")
st.markdown("### Solución final para traslapes y errores de tileado")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada")
    uploaded_file = st.file_uploader("Subir Imagen", type=["png", "jpg", "jpeg"])
    
    mode = st.radio("Fondo de Imagen", ["Auto-Detectar", "Líneas Negras", "Líneas Blancas"])
    
    st.subheader("2. Geometría Revit")
    grid_size = st.number_input("Tamaño del Tile (cm/pulg)", 1.0, 5000.0, 100.0, help="Debe ser el tamaño real de un 'bloque' de tu patrón.")
    
    st.subheader("3. Ajustes de Calidad")
    do_skeleton = st.checkbox("Usar Esqueletización (Recomendado)", value=True)
    closing_sz = st.slider("Unir Líneas Sueltas", 0, 10, 2)
    epsilon_val = st.slider("Simplificación Vectorial", 0.001, 0.050, 0.005, format="%.3f")
    
    st.subheader("4. Detección de Bordes (Canny)")
    blur_size = st.slider("Suavizado (Blur)", 1, 15, 3, step=2, help="Tamaño del kernel de blur (debe ser impar)")
    canny_low = st.slider("Umbral Bajo", 10, 150, 30, help="Umbral mínimo para detección de bordes")
    canny_high = st.slider("Umbral Alto", 50, 300, 100, help="Umbral máximo para detección de bordes")
    
    st.subheader("5. Filtrado de Segmentos")
    min_contour = st.slider("Longitud Mínima de Contorno (px)", 5, 100, 20, help="Contornos más cortos serán ignorados")
    min_segment = st.slider("Longitud Mínima de Segmento", 0.01, 0.15, 0.025, format="%.3f", help="Segmentos más cortos serán ignorados (0-1)")

if uploaded_file:
    gen = PatternGenerator(grid_size)
    res = gen.process_image(uploaded_file, epsilon_val, closing_sz, mode, do_skeleton, 
                           canny_low, canny_high, blur_size, min_contour, min_segment)
    
    if "error" in res:
        st.error(res["error"])
    else:
        with col_view:
            t1, t2, t3 = st.tabs(["📐 Vista Previa", "🔲 Preview Revit (Tileado)", "📄 Código .PAT"])
            with t1:
                st.image(res["vector_img"], caption="Vectores detectados", use_container_width=True)
                st.success(res["stats"])
            with t2:
                st.image(res["pat_preview"], caption="Simulación de Tileado (3x3 tiles)", use_container_width=True)
                st.info("Esta vista muestra cómo se verá el patrón repetido en Revit")
            with t3:
                st.code(res["pat_content"], language="text")
            
            st.download_button("📥 Descargar .PAT para Revit", res["pat_content"], "Hatch_Pattern.pat", "text/plain")