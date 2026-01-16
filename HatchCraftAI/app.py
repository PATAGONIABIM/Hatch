import streamlit as st
import cv2
import numpy as np
from core_logic import PatternGenerator, AIPatternClassifier, PATTERN_TEMPLATES, generate_pat_from_template, render_pat_preview

st.set_page_config(page_title="HatchCraft AI v4.1", layout="wide")

if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

st.title("HatchCraft AI 🧱✨")
st.markdown("### Generación de patrones con Inteligencia Artificial")

with st.sidebar:
    st.header("⚙️ Configuración API")
    
    if st.session_state.gemini_api_key:
        st.success("✅ API Key configurada")
        if st.button("🗑️ Eliminar API Key"):
            st.session_state.gemini_api_key = ""
            st.rerun()
    else:
        st.warning("⚠️ Sin API Key (modo manual)")
    
    with st.expander("🔑 Agregar/Cambiar API Key"):
        new_key = st.text_input("Gemini API Key", type="password")
        if st.button("💾 Guardar"):
            if new_key:
                st.session_state.gemini_api_key = new_key
                st.rerun()
    
    st.divider()
    st.caption("Powered by Gemini Vision")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Entrada")
    uploaded_file = st.file_uploader("Subir Imagen", type=["png", "jpg", "jpeg"])
    
    st.subheader("2. Modo de Generación")
    generation_mode = st.radio(
        "Método",
        ["🎯 Template (Selección Manual)", "🤖 IA (Auto-detectar)", "🔧 Clásico (Detección de bordes)"]
    )
    
    st.subheader("3. Geometría Revit")
    grid_size = st.number_input("Tamaño del Tile (cm/pulg)", 1.0, 5000.0, 100.0)
    
    # Mostrar selector de template si es modo manual
    if "Template" in generation_mode:
        st.subheader("4. Seleccionar Patrón")
        template_options = {
            "running_bond": "🧱 Running Bond (Ladrillos escalonados)",
            "stack_bond": "📦 Stack Bond (Ladrillos alineados)",
            "herringbone_45": "📐 Herringbone 45° (Espiga diagonal)",
            "basketweave": "🧺 Basketweave (Canasta)",
            "square_tile": "⬜ Square Tile (Baldosas cuadradas)",
            "diagonal_tile": "◇ Diagonal Tile (Baldosas diagonales)"
        }
        selected_template = st.selectbox(
            "Tipo de patrón",
            list(template_options.keys()),
            format_func=lambda x: template_options[x]
        )
    
    # Opciones para modo clásico
    if "Clásico" in generation_mode:
        st.subheader("4. Ajustes Clásicos")
        mode = st.radio("Fondo", ["Auto-Detectar", "Líneas Negras", "Líneas Blancas"])
        do_skeleton = st.checkbox("Esqueletización", value=True)
        closing_sz = st.slider("Unir Líneas", 0, 10, 2)
        epsilon_val = st.slider("Simplificación", 0.001, 0.050, 0.005, format="%.3f")
        blur_size = st.slider("Blur", 1, 15, 3, step=2)
        canny_low = st.slider("Canny Bajo", 10, 150, 30)
        canny_high = st.slider("Canny Alto", 50, 300, 100)
        min_contour = st.slider("Min Contorno", 5, 100, 20)
        min_segment = st.slider("Min Segmento", 0.01, 0.15, 0.025, format="%.3f")

# Procesar
if "Template" in generation_mode:
    # Modo template manual - no necesita imagen
    pat_content = generate_pat_from_template(selected_template)
    pat_preview = render_pat_preview(pat_content)
    
    with col_view:
        t1, t2 = st.tabs(["🔲 Preview Revit", "📄 Código .PAT"])
        with t1:
            st.image(pat_preview, caption=f"Preview: {PATTERN_TEMPLATES[selected_template]['name']}", use_container_width=True)
            st.success(f"Template: {PATTERN_TEMPLATES[selected_template]['name']}")
        with t2:
            st.code(pat_content, language="text")
        
        st.download_button("📥 Descargar .PAT", pat_content, f"{selected_template}.pat", "text/plain")

elif uploaded_file:
    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    if "IA" in generation_mode:
        if not st.session_state.gemini_api_key:
            with col_view:
                st.error("⚠️ Configura tu API Key para usar el modo IA")
        else:
            with st.spinner("🤖 Clasificando patrón..."):
                classifier = AIPatternClassifier(st.session_state.gemini_api_key)
                res = classifier.generate_from_classification(image_bytes)
            
            if "error" in res:
                st.error(res["error"])
            else:
                with col_view:
                    t1, t2 = st.tabs(["🔲 Preview Revit", "📄 Código .PAT"])
                    with t1:
                        col_orig, col_pat = st.columns(2)
                        with col_orig:
                            st.image(image_bytes, caption="Imagen Original", use_container_width=True)
                        with col_pat:
                            st.image(res["pat_preview"], caption="Patrón Generado", use_container_width=True)
                        st.success(res["stats"])
                        st.info(f"Tipo detectado: **{res['pattern_type']}**. Si no es correcto, usa el modo Template manual.")
                    with t2:
                        st.code(res["pat_content"], language="text")
                    
                    st.download_button("📥 Descargar .PAT", res["pat_content"], f"{res['pattern_type']}.pat", "text/plain")
    
    elif "Clásico" in generation_mode:
        gen = PatternGenerator(grid_size)
        res = gen.process_image(uploaded_file, epsilon_val, closing_sz, mode, do_skeleton, 
                               canny_low, canny_high, blur_size, min_contour, min_segment)
        
        if "error" in res:
            st.error(res["error"])
        else:
            with col_view:
                t1, t2, t3 = st.tabs(["📐 Vectores", "🔲 Preview Revit", "📄 Código .PAT"])
                with t1:
                    st.image(res["vector_img"], caption="Vectores detectados", use_container_width=True)
                    st.success(res["stats"])
                with t2:
                    st.image(res["pat_preview"], caption="Preview Tileado", use_container_width=True)
                with t3:
                    st.code(res["pat_content"], language="text")
                
                st.download_button("📥 Descargar .PAT", res["pat_content"], "hatch_pattern.pat", "text/plain")