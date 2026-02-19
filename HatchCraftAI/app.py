import streamlit as st
import numpy as np
from core_logic import DXFtoPatConverter, ImageToPatConverter, render_pat_preview
import tempfile
import os

st.set_page_config(page_title="HatchCraft - Pattern Generator", layout="wide")

st.title("HatchCraft 📐✨")
st.markdown("### Convierte dibujos y imágenes a patrones para Revit")

# Selector de modo
mode = st.radio("Selecciona el modo:", 
                ["📁 DXF (AutoCAD)", "🖼️ Imagen (Canny/Skeleton)"], 
                horizontal=True)

col1, col2 = st.columns([1, 1])

with col1:
    if mode == "📁 DXF (AutoCAD)":
        st.subheader("📁 Subir DXF")
        st.caption("Dibuja líneas en AutoCAD y guarda como DXF")
        
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo DXF aquí",
            type=["dxf"],
            key="dxf_uploader"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
            
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf', mode='wb') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                with st.spinner("🔄 Convirtiendo DXF a PAT..."):
                    converter = DXFtoPatConverter()
                    result = converter.convert(tmp_path)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.result = result
                    st.success(result["stats"])
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    else:  # Modo Imagen
        st.subheader("🖼️ Subir Imagen")
        st.caption("Detecta bordes y líneas a partir de imágenes")
        
        uploaded_file = st.file_uploader(
            "Arrastra una imagen del patrón",
            type=["png", "jpg", "jpeg"],
            key="img_uploader"
        )
        
        if uploaded_file:
            st.caption("⚙️ Modo de Detección")
            detect_method = st.radio(
                "Elige el algoritmo:", 
                ["📏 Líneas Rectas (Hough - Ideal para geometría, baldosas)", 
                 "🌿 Formas Orgánicas (Contornos - Ideal para piedra, texturas)"],
                label_visibility="collapsed"
            )
            
            st.caption("⚙️ Parámetros de visión")
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                canny_low = st.slider("Canny Low (Bordes)", 10, 200, 50, key="canny_low")
                canny_high = st.slider("Canny High", 50, 300, 150, key="canny_high")
                blur_size = st.slider("Desenfoque (Limpiar ruido)", 1, 11, 3, 2, key="blur")
                
            with col_s2:
                if "Rectas" in detect_method:
                    p1 = st.slider("Longitud mín. de línea", 5, 100, 20, key="hough_min")
                    p2 = st.slider("Unir huecos en líneas (Gap)", 1, 50, 5, key="hough_gap")
                    method_key = "hough"
                else:
                    p1 = st.slider("Longitud mín. contorno", 5, 100, 20, key="cont_min")
                    p2 = st.slider("Suavizado de curvas", 0.001, 0.05, 0.005, format="%.3f", key="cont_eps")
                    method_key = "contour"
            
            # Procesar automáticamente al cambiar cualquier slider
            converter = ImageToPatConverter()
            image_bytes = uploaded_file.getvalue()
            result = converter.convert(image_bytes, method=method_key, canny_low=canny_low, canny_high=canny_high, 
                                       blur_size=blur_size, param1=p1, param2=p2)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.result = result
                st.caption(result["stats"])

with col2:
    st.subheader("🔲 Resultado")
    
    if 'result' in st.session_state and st.session_state.result:
        result = st.session_state.result
        
        tab_debug, tab_preview, tab_code, tab_download = st.tabs([
            "🔍 Debug", "🔲 Preview", "📄 Código", "📥 Descargar"
        ])
        
        with tab_debug:
            if "debug_img" in result:
                st.image(result["debug_img"], use_container_width=True)
            else:
                st.info("Sin imagen de debug")
        
        with tab_preview:
            preview_scale = st.slider("🔍 Escala", 0.1, 10.0, 1.0, 0.1)
            pat_preview = render_pat_preview(result["pat_content"], tile_count=3, 
                                             preview_size=600, manual_scale=preview_scale)
            st.image(pat_preview, caption="Preview tileado (3x3)", use_container_width=True)
        
        with tab_code:
            st.code(result["pat_content"], language="text")
        
        with tab_download:
            st.download_button(
                "📥 Descargar .PAT para Revit",
                result["pat_content"],
                "HatchCraft.pat",
                "text/plain",
                use_container_width=True
            )
            st.info("**En Revit:** Manage → Additional Settings → Fill Patterns → Import")
    else:
        empty_img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        st.image(empty_img, caption="El patrón aparecerá aquí")
        st.info("👈 Sube un archivo para comenzar")

st.divider()
st.markdown("""
**Modos disponibles:**
- **DXF**: Dibuja en AutoCAD con líneas precisas. Ángulos cada 15°.
- **Imagen**: Detecta bordes automáticamente. Ideal para texturas orgánicas.
""")