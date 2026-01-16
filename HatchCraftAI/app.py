import streamlit as st
import numpy as np
from PIL import Image
from core_logic import GeminiPatternGenerator, render_pat_preview

st.set_page_config(page_title="HatchCraft AI v5.0", layout="wide")

# Session state
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

st.title("HatchCraft AI 🧱✨")
st.markdown("### Generación de patrones con Gemini 3 Pro")

# Sidebar - API Key
with st.sidebar:
    st.header("⚙️ Configuración")
    
    if st.session_state.gemini_api_key:
        st.success("✅ API Key configurada")
        if st.button("🗑️ Eliminar API Key"):
            st.session_state.gemini_api_key = ""
            st.rerun()
    else:
        st.warning("⚠️ Configura tu API Key")
    
    with st.expander("🔑 API Key", expanded=not st.session_state.gemini_api_key):
        new_key = st.text_input("Gemini API Key", type="password")
        if st.button("💾 Guardar"):
            if new_key:
                st.session_state.gemini_api_key = new_key
                st.rerun()
    
    st.divider()
    st.caption("Modelo: Gemini 3 Pro Preview")
    st.caption("(El más potente)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Imagen del Patrón")
    
    uploaded_file = st.file_uploader(
        "Sube una imagen del patrón a replicar",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        if not st.session_state.gemini_api_key:
            st.error("⚠️ Configura tu API Key en la barra lateral")
        else:
            if st.button("🚀 Generar Patrón con IA", type="primary", use_container_width=True):
                with st.spinner("🤖 Analizando imagen con Gemini 3 Pro..."):
                    # Leer bytes
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    # Generar
                    generator = GeminiPatternGenerator(st.session_state.gemini_api_key)
                    result = generator.generate_pattern(image_bytes)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state.last_result = result
                        st.success(result["stats"])

with col2:
    st.subheader("🔲 Resultado")
    
    if 'last_result' in st.session_state and st.session_state.last_result:
        result = st.session_state.last_result
        
        # Preview
        st.image(result["pat_preview"], caption="Preview tileado (3x3)", use_container_width=True)
        
        # Tabs
        tab1, tab2 = st.tabs(["📄 Código .PAT", "📥 Descargar"])
        
        with tab1:
            st.code(result["pat_content"], language="text")
        
        with tab2:
            st.download_button(
                "📥 Descargar .PAT para Revit",
                result["pat_content"],
                "HatchCraft_AI.pat",
                "text/plain",
                use_container_width=True
            )
            st.info("**En Revit:** Manage → Additional Settings → Fill Patterns → Import")
    else:
        # Placeholder
        empty_img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        st.image(empty_img, caption="El patrón generado aparecerá aquí")
        st.info("👈 Sube una imagen y haz clic en 'Generar Patrón'")

# Footer
st.divider()
st.markdown("""
**Instrucciones:**
1. Obtén tu API Key gratis en [Google AI Studio](https://aistudio.google.com/apikey)
2. Sube una imagen del patrón que quieres replicar
3. Clic en "Generar Patrón con IA"
4. Descarga el archivo .PAT e impórtalo en Revit
""")