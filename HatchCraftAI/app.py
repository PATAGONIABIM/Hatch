import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Pro v3.2", layout="wide")

st.title("HatchCraft Pro: Seamless Revit Tiling 🏗️")
st.markdown("### Solución de Continuidad Vectorial")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada")
    uploaded_file = st.file_uploader("Subir Imagen", type=["png", "jpg", "jpeg"])
    
    mode = st.radio("Modo de Color", ["Auto-Detectar", "Líneas Negras", "Líneas Blancas"])
    
    st.subheader("2. Geometría")
    unit_system = st.selectbox("Unidades:", ["Centímetros", "Pulgadas"])
    grid_size = st.number_input(f"Tamaño del Tile ({unit_system})", 1.0, 5000.0, 100.0)
    
    st.subheader("3. Refinado")
    do_skeleton = st.checkbox("Usar Esqueletización", value=True)
    closing_sz = st.slider("Unir Líneas", 0, 15, 3)
    epsilon_val = st.slider("Simplificación", 0.001, 0.050, 0.008, format="%.3f")

if uploaded_file:
    gen = PatternGenerator(grid_size, unit_system)
    res = gen.process_image(uploaded_file, epsilon_val, closing_sz, mode, do_skeleton)
    
    if "error" in res:
        st.error(res["error"])
    else:
        with col_view:
            t1, t2 = st.tabs(["📐 Previsualización Técnica", "📄 Código .PAT"])
            with t1:
                # Actualizado a width="stretch" para eliminar el aviso de obsolescencia
                st.image(res["vector_img"], caption="Simulación de Tiling Infinito", width="stretch")
                st.info(res["stats"])
                st.download_button("📥 Descargar .PAT", res["pat_content"], "Hatch_Perfecto.pat", "text/plain")
            with t2:
                st.code(res["pat_content"], language="text")