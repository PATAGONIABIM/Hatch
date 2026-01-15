import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Pro v2.1", layout="wide")

st.title("HatchCraft Pro: Vectorizer 🧬")
st.markdown("### Previsualización de Patrón .PAT (Líneas Negras / Fondo Blanco)")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada")
    uploaded_file = st.file_uploader("Subir PNG/JPG", type=["png", "jpg", "jpeg"])
    
    mode = st.radio("Modo de Color", ["Auto-Detectar", "Fuerza Fondo Blanco (Líneas Negras)", "Fuerza Fondo Negro (Líneas Blancas)"])
    
    st.subheader("2. Geometría Revit")
    grid_size = st.number_input("Tamaño de Celda (unidades)", 1.0, 1000.0, 100.0)
    
    st.subheader("3. Refinado")
    do_skeleton = st.checkbox("Usar Esqueletización (Línea Central)", value=True)
    closing_sz = st.slider("Conectividad (Unir líneas)", 0, 10, 2)
    epsilon_val = st.slider("Simplificación (Tolerancia)", 0.001, 0.050, 0.010, format="%.3f")

if uploaded_file:
    gen = PatternGenerator(grid_size)
    
    with st.spinner("Generando previsualización vectorial..."):
        res = gen.process_image(
            uploaded_file, 
            epsilon_factor=epsilon_val,
            closing_size=closing_sz,
            mode=mode,
            use_skeleton=do_skeleton
        )
    
    if "error" in res:
        st.error(res["error"])
    else:
        with col_view:
            t1, t2 = st.tabs(["📐 Previsualización Técnica", "📄 Código .PAT"])
            
            with t1:
                # Mostramos el resultado vectorial (Líneas negras sobre blanco)
                st.image(res["vector_img"], caption="Simulación de impresión CAD", use_container_width=True)
                
                st.info(res["stats"])
                if res.get("warning"):
                    st.warning(res["warning"])
                
                st.download_button("📥 Descargar Archivo .PAT", res["pat_content"], "hatch_profesional.pat", "text/plain")
                
            with t2:
                st.text_area("Contenido del archivo", res["pat_content"], height=400)