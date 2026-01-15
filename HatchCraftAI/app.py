
import streamlit as st
import numpy as np
import cv2
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft AI", layout="wide")

st.title("HatchCraft AI 🏗️")
st.markdown("### Generador de Patrones .PAT para Revit desde Imágenes")

# Sidebar for controls
st.sidebar.header("Configuración")
grid_size = st.sidebar.number_input("Tamaño de Módulo (Unidades)", min_value=0.1, value=1.0, step=0.1)
epsilon = st.sidebar.slider("Suavizado (Simplificación)", 0.001, 0.050, 0.005, format="%.3f")

uploaded_file = st.file_uploader("Sube tu imagen (PNG/JPG) con el patrón (negro sobre blanco funciona mejor)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Instantiate Generator
    generator = PatternGenerator(grid_width=grid_size, grid_height=grid_size)
    
    # Process
    with st.spinner('Procesando imagen y vectorizando...'):
        result = generator.process_image(uploaded_file, epsilon_factor=epsilon)
    
    if "error" in result:
        st.error(result["error"])
    else:
        # Layout: Split columns for Preview and Code
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Previsualización de Vectores")
            st.image(result["preview_img"], caption="Vectores Detectados", use_column_width=True)
            st.info(result["stats"])
            
            # Tiling Preview (Simulated)
            st.markdown("#### Simulación de Repetición (3x3)")
            # Create a simple tile mosaic from the preview image for visual check
            tile = result["preview_img"]
            # Resize just for display performance if needed, but 3x3 is fine
            # h, w = tile.shape[:2]
            # mosaic = np.tile(tile, (3, 3, 1)) # This works if tile is numpy array
            
            # OpenCV tile
            row1 = np.hstack([tile, tile, tile])
            mosaic = np.vstack([row1, row1, row1])
            st.image(mosaic, caption="Efecto Mosaico (Seamless Check)", use_column_width=True)

        with col2:
            st.subheader("Archivo .PAT Generado")
            pat_content = result["pat_content"]
            st.text_area("Contenido:", pat_content, height=400)
            
            st.download_button(
                label="📥 Descargar archivo .pat",
                data=pat_content,
                file_name="hatchcraft_pattern.pat",
                mime="text/plain"
            )

st.sidebar.markdown("---")
st.sidebar.info("Tips: \n- Usa imágenes de alto contraste.\n- Para 'seamless' asegurate que el dibujo coincida en los bordes.\n- Ajusta el 'Suavizado' para reducir el número de líneas.")
