import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator, AIPatternGenerator

st.set_page_config(page_title="HatchCraft AI v4.0", layout="wide")

# Inicializar session state para API key
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

st.title("HatchCraft AI 🧱✨")
st.markdown("### Generación de patrones con Inteligencia Artificial")

# Sidebar para configuración de API
with st.sidebar:
    st.header("⚙️ Configuración API")
    
    # Mostrar estado actual
    if st.session_state.gemini_api_key:
        st.success("✅ API Key configurada")
        key_preview = st.session_state.gemini_api_key[:8] + "..." + st.session_state.gemini_api_key[-4:]
        st.caption(f"Key: {key_preview}")
        
        if st.button("🗑️ Eliminar API Key"):
            st.session_state.gemini_api_key = ""
            st.rerun()
    else:
        st.warning("⚠️ Sin API Key")
    
    st.divider()
    
    # Formulario para agregar/cambiar API key
    with st.expander("🔑 Agregar/Cambiar API Key"):
        new_key = st.text_input("Gemini API Key", type="password", 
                               help="Obtén tu key en https://aistudio.google.com/apikey")
        if st.button("💾 Guardar API Key"):
            if new_key:
                st.session_state.gemini_api_key = new_key
                st.success("API Key guardada!")
                st.rerun()
            else:
                st.error("Ingresa una API key válida")
    
    st.divider()
    st.caption("Powered by Gemini Vision")

# Columnas principales
col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada")
    uploaded_file = st.file_uploader("Subir Imagen", type=["png", "jpg", "jpeg"])
    
    # Selector de modo
    st.subheader("2. Modo de Generación")
    generation_mode = st.radio(
        "Método",
        ["🤖 IA (Gemini Vision)", "🔧 Clásico (Detección de bordes)"],
        help="IA genera patrones geométricos limpios. Clásico traza los contornos de la imagen."
    )
    
    use_ai = "IA" in generation_mode
    
    if use_ai and not st.session_state.gemini_api_key:
        st.error("⚠️ Configura tu API Key en la barra lateral para usar el modo IA")
    
    st.subheader("3. Geometría Revit")
    grid_size = st.number_input("Tamaño del Tile (cm/pulg)", 1.0, 5000.0, 100.0)
    
    # Mostrar opciones según el modo
    if not use_ai:
        st.subheader("4. Ajustes Clásicos")
        mode = st.radio("Fondo de Imagen", ["Auto-Detectar", "Líneas Negras", "Líneas Blancas"])
        do_skeleton = st.checkbox("Usar Esqueletización", value=True)
        closing_sz = st.slider("Unir Líneas Sueltas", 0, 10, 2)
        epsilon_val = st.slider("Simplificación Vectorial", 0.001, 0.050, 0.005, format="%.3f")
        
        st.subheader("5. Detección de Bordes")
        blur_size = st.slider("Suavizado (Blur)", 1, 15, 3, step=2)
        canny_low = st.slider("Umbral Bajo", 10, 150, 30)
        canny_high = st.slider("Umbral Alto", 50, 300, 100)
        
        st.subheader("6. Filtrado")
        min_contour = st.slider("Long. Mín. Contorno (px)", 5, 100, 20)
        min_segment = st.slider("Long. Mín. Segmento", 0.01, 0.15, 0.025, format="%.3f")

# Procesar imagen
if uploaded_file:
    # Leer bytes de la imagen
    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset para poder leer de nuevo
    
    if use_ai and st.session_state.gemini_api_key:
        # Modo IA
        with st.spinner("🤖 Analizando imagen con Gemini Vision..."):
            ai_gen = AIPatternGenerator(st.session_state.gemini_api_key)
            res = ai_gen.analyze_and_generate(image_bytes, grid_size)
        
        if "error" in res:
            st.error(res["error"])
        else:
            with col_view:
                t1, t2 = st.tabs(["🔲 Preview Revit (Tileado)", "📄 Código .PAT"])
                with t1:
                    st.image(res["pat_preview"], caption="Simulación de Tileado (3x3 tiles)", use_container_width=True)
                    st.success(res["stats"])
                with t2:
                    st.code(res["pat_content"], language="text")
                
                st.download_button("📥 Descargar .PAT para Revit", res["pat_content"], "Hatch_AI_Pattern.pat", "text/plain")
    
    elif not use_ai:
        # Modo Clásico
        gen = PatternGenerator(grid_size)
        res = gen.process_image(uploaded_file, epsilon_val, closing_sz, mode, do_skeleton, 
                               canny_low, canny_high, blur_size, min_contour, min_segment)
        
        if "error" in res:
            st.error(res["error"])
        else:
            with col_view:
                t1, t2, t3 = st.tabs(["📐 Vista Previa", "🔲 Preview Revit", "📄 Código .PAT"])
                with t1:
                    st.image(res["vector_img"], caption="Vectores detectados", use_container_width=True)
                    st.success(res["stats"])
                with t2:
                    st.image(res["pat_preview"], caption="Simulación de Tileado (3x3 tiles)", use_container_width=True)
                with t3:
                    st.code(res["pat_content"], language="text")
                
                st.download_button("📥 Descargar .PAT para Revit", res["pat_content"], "Hatch_Pattern.pat", "text/plain")
    else:
        with col_view:
            st.info("👈 Configura tu API Key en la barra lateral para usar el modo IA")