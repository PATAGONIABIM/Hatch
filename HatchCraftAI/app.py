import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Pro v3.3", layout="wide")

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
    canny_low = st.slider("Umbral Bajo", 10, 150, 30, help="Umbral mínimo para detección de bordes")
    canny_high = st.slider("Umbral Alto", 50, 300, 100, help="Umbral máximo para detección de bordes")

if uploaded_file:
    gen = PatternGenerator(grid_size)
    res = gen.process_image(uploaded_file, epsilon_val, closing_sz, mode, do_skeleton, canny_low, canny_high)
    
    if "error" in res:
        st.error(res["error"])
    else:
        with col_view:
            t1, t2 = st.tabs(["📐 Vista Previa", "📄 Código .PAT"])
            with t1:
                # Corregido a width='stretch' para eliminar avisos de Streamlit
                st.image(res["vector_img"], caption="Tileado detectado (Líneas Negras)", width="stretch")
                st.success(res["stats"])
                st.download_button("📥 Descargar .PAT para Revit", res["pat_content"], "Hatch_Sin_Traslape.pat", "text/plain")
            with t2:
                st.code(res["pat_content"], language="text")