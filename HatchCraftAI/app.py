
import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator

st.set_page_config(page_title="HatchCraft Pro", layout="wide")

st.title("HatchCraft Pro: Vectorizer 🧬")
st.markdown("### Generador de Patrones CAD de Alta Fidelidad")

# Layout
col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada y Pre-Proceso")
    uploaded_file = st.file_uploader("Imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    manual_inv = st.checkbox("🔄 Invertir Colores Manualmente", value=False, help="Úsalo si el fondo aparece negro y las líneas blancas en la previsualización.")
    
    st.subheader("2. Geometría")
    grid_size = st.number_input("Tamaño Base (m)", 0.1, 100.0, 1.0, step=0.1)
    scale_factor = st.slider("Escala", 0.1, 5.0, 1.0, 0.1)
    
    st.subheader("3. Pipeline de Limpieza")
    closing_sz = st.slider("Grosor de Unión (Gluing)", 1, 15, 3, help="Cierra huecos antes de adelgazar la línea.")
    epsilon_val = st.slider("Simplificación (Douglas-Peucker)", 0.001, 0.100, 0.005, format="%.3f")

if uploaded_file:
    uploaded_file.seek(0)
    gen = PatternGenerator(grid_width=grid_size, grid_height=grid_size)
    
    with st.spinner("Procesando: Auto-Invert -> Unión -> Esqueleto -> Vector..."):
        res = gen.process_image(
            uploaded_file, 
            epsilon_factor=epsilon_val,
            scale=scale_factor,
            closing_size=closing_sz,
            manual_invert=manual_inv
        )
    
    if "error" in res:
        st.error(res["error"])
    else:
        with col_view:
            # Dual View
            t1, t2 = st.tabs(["👁️ Visión Computadora (Esqueleto)", "📐 Resultado Vectorial"])
            
            with t1:
                st.image(res["processed_img"], caption="Paso Intermedio: Esqueleto (Lo que se vectoriza)", use_column_width=True)
                st.info("Deberías ver líneas blancas finas sobre fondo negro. Si ves bloques sólidos, reduce el 'Grosor de Unión' o revisa 'Invertir'.")
                
            with t2:
                st.image(res["vector_img"], caption="Vectores Finales", use_column_width=True)
                
                if res.get("warning"):
                    st.warning(res["warning"])
                else:
                    st.success(res["stats"])
                
                pat_data = res["pat_content"]
                st.download_button("📥 Descargar .PAT Optimizado", pat_data, "hatchcraft_pro.pat", "text/plain")
                
                with st.expander("Ver Datos .PAT"):
                    st.text(pat_data[:1000] + "\n... (más líneas) ...")

else:
    with col_view:
        st.info("👈 Sube una imagen para comenzar.")
        st.markdown("""
        **Guía Rápida:**
        1.  **Invertir**: Asegura que el fondo sea negro en la vista 'Esqueleto'.
        2.  **Grosor**: Sube si el dibujo se rompe. Baja si se empasta.
        3.  **Simplificación**: Sube si Revit se queja de un patrón muy pesado.
        """)
