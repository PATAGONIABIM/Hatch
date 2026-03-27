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
            # ── Método de detección ──
            st.caption("⚙️ Modo de Detección")
            detect_method = st.radio(
                "Algoritmo:", 
                ["📏 Líneas Rectas (Hough)", 
                 "⚡ Líneas Precisas (LSD - Sub-pixel)",
                 "🌿 Formas Orgánicas (Contornos)"],
                label_visibility="collapsed"
            )
            
            # ── Pre-procesamiento avanzado ──
            with st.expander("🔬 Pre-procesamiento", expanded=False):
                col_pre1, col_pre2 = st.columns(2)
                with col_pre1:
                    use_clahe = st.checkbox("CLAHE (Mejorar contraste)", value=False,
                                           help="Ideal para fotos con iluminación desigual")
                    if use_clahe:
                        clahe_clip = st.slider("CLAHE Intensidad", 1.0, 8.0, 2.0, 0.5, key="clahe_clip")
                    else:
                        clahe_clip = 2.0
                
                with col_pre2:
                    use_skeleton = st.checkbox("Skeleton (Zhang-Suen)", value=False,
                                              help="Adelgaza bordes a 1 pixel")
                    merge_segments = st.checkbox("Unir segmentos colineales", value=False,
                                                help="Reduce fragmentación uniendo líneas similares")

                use_adaptive = st.checkbox("Adaptive Threshold (en vez de Canny)", value=False,
                                          help="Mejor para dibujos escaneados con trazos gruesos")
                if use_adaptive:
                    col_at1, col_at2 = st.columns(2)
                    with col_at1:
                        adaptive_block = st.slider("Bloque Adaptivo", 3, 51, 11, 2, key="adap_block")
                    with col_at2:
                        adaptive_c = st.slider("Constante C", 0, 20, 2, key="adap_c")
                else:
                    adaptive_block, adaptive_c = 11, 2
                
                dedup_threshold = st.slider("🚫 Filtro líneas dobles (px)", 0, 30, 8, 1, key="dedup",
                                           help="Distancia mínima entre líneas para NO considerarlas duplicadas")
            
            # ── Parámetros principales ──
            st.caption("⚙️ Parámetros de visión")
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                canny_low = st.slider("Canny Low (Bordes)", 10, 200, 50, key="canny_low",
                                      disabled=use_adaptive)
                canny_high = st.slider("Canny High", 50, 300, 150, key="canny_high",
                                       disabled=use_adaptive)
                blur_size = st.slider("Desenfoque (Limpiar ruido)", 1, 11, 3, 2, key="blur")
                
            with col_s2:
                if "Hough" in detect_method:
                    p1 = st.slider("Longitud mín. de línea", 5, 100, 20, key="hough_min")
                    p2 = st.slider("Unir huecos en líneas (Gap)", 1, 50, 5, key="hough_gap")
                    method_key = "hough"
                elif "LSD" in detect_method:
                    p1 = st.slider("Longitud mín. de línea", 5, 100, 15, key="lsd_min")
                    p2 = 0  # LSD no usa segundo parámetro
                    method_key = "lsd"
                else:
                    p1 = st.slider("Longitud mín. contorno", 5, 100, 20, key="cont_min")
                    p2 = st.slider("Suavizado de curvas", 0.001, 0.05, 0.005, format="%.3f", key="cont_eps")
                    method_key = "contour"
            
            # ── Procesar automáticamente ──
            converter = ImageToPatConverter()
            image_bytes = uploaded_file.getvalue()
            result = converter.convert(
                image_bytes, method=method_key, 
                canny_low=canny_low, canny_high=canny_high, 
                blur_size=blur_size, param1=p1, param2=p2,
                use_clahe=use_clahe, clahe_clip=clahe_clip,
                use_adaptive=use_adaptive, adaptive_block=adaptive_block, adaptive_c=adaptive_c,
                use_skeleton=use_skeleton,
                merge_segments=merge_segments,
                dedup_threshold=float(dedup_threshold)
            )
            
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
- **Imagen → Hough**: Detecta líneas rectas. Ideal para baldosas y geometría.
- **Imagen → LSD**: Detector sub-pixel auto-tuning. El más preciso para líneas.
- **Imagen → Contornos**: Formas orgánicas (piedra, texturas).
- **Pre-procesamiento**: CLAHE, Adaptive Threshold, Skeleton, Merge.
""")