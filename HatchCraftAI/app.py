
import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Clean-Line", layout="wide")

st.title("HatchCraft: Clean-Line Vector Generator 📐")
st.markdown("""
**Transforma bocetos a mano en patrones técnicos para Revit (.pat).**
Este algoritmo usa *Skeletonization* para extraer el eje central de las líneas.
""")

col_conf, col_prev = st.columns([1, 2])

with col_conf:
    st.subheader("Configuración")
    uploaded_file = st.file_uploader("1. Sube Imagen (PNG/JPG)", type=["png", "jpg"])
    
    st.markdown("---")
    grid_base = st.number_input("Ancho Base Módulo (m)", 1.0, 100.0, 10.0)
    scale = st.slider("Escala Patrón", 0.1, 5.0, 1.0, 0.1, help="Multiplica el tamaño final")
    epsilon = st.slider("Simplificación (Epsilon)", 0.001, 0.05, 0.005, format="%.4f", help="Valores altos = Líneas más rectas (Low Poly)")
    
    st.markdown("---")
    st.info("El algoritmo 'Skeletonize' reduce trazos gruesos a líneas simples.")

if uploaded_file:
    gen = PatternGenerator(grid_width=grid_base, grid_height=grid_base)
    
    # Process
    with st.spinner("Adelgazando líneas y vectorizando..."):
        res = gen.process_image(uploaded_file, epsilon_factor=epsilon, scale=scale)
    
    if "error" in res:
        st.error(res["error"])
    else:
        with col_prev:
            st.subheader("Previsualización Vectorial")
            st.image(res["preview_img"], caption="Resultado (Vectores Negros)", use_column_width=True)
            
            st.success(res["stats"])
            
            pat_data = res["pat_content"]
            st.download_button("📥 Descargar .PAT", pat_data, "clean_pattern.pat", "text/plain")
            
            with st.expander("Ver código generado"):
                st.code(pat_data, language="text")
