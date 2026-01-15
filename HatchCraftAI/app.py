import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Pro v3.1", layout="wide")

st.title("HatchCraft Pro: Revit Seamless 🏗️")
st.markdown("### Generador de Patrones de Modelo (Centímetros/Pulgadas)")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada")
    uploaded_file = st.file_uploader("Subir Imagen", type=["png", "jpg", "jpeg"])
    
    mode = st.radio("Modo de Color", ["Auto-Detectar", "Fuerza Fondo Blanco (Líneas Negras)", "Fuerza Fondo Negro (Líneas Blancas)"])
    
    st.subheader("2. Unidades y Escala")
    unit_system = st.selectbox("Trabajar en:", ["Centímetros", "Pulgadas"])
    # Tamaño de la celda de repetición
    grid_size = st.number_input(f"Tamaño del Patrón ({unit_system})", 1.0, 5000.0, 100.0, help="Define el tamaño del 'cuadrado' que se repite en Revit.")
    
    st.subheader("3. Refinado")
    do_skeleton = st.checkbox("Usar Esqueletización (Línea Central)", value=True)
    closing_sz = st.slider("Conectividad (Unir líneas)", 0, 15, 3)
    epsilon_val = st.slider("Simplificación (Tolerancia)", 0.001, 0.050, 0.008, format="%.3f")

if uploaded_file:
    gen = PatternGenerator(grid_size, unit_system)
    
    with st.spinner("Calculando repetición infinita..."):
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
                # Corregido: width='stretch' elimina el error de 2025/2026
                st.image(res["vector_img"], caption="Simulación de Repetición CAD", width='stretch')
                
                st.info(res["stats"])
                st.download_button("📥 Descargar Archivo .PAT", res["pat_content"], "Hatch_Revit_CM.pat", "text/plain")
                
            with t2:
                st.text_area("Contenido para Revit", res["pat_content"], height=400)