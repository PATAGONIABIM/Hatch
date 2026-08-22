import streamlit as st
import numpy as np
from core_logic import DXFtoPatConverter, ImageToPatConverter, render_pat_preview
import tempfile
import os
import hashlib

@st.cache_data(show_spinner="Procesando imagen...")
def cached_convert(image_hash, **kwargs):
    """Wrapper cacheado: si los params no cambian, devuelve resultado instantáneo."""
    converter = ImageToPatConverter()
    return converter.convert(**kwargs)

st.set_page_config(page_title="HatchCraft - Pattern Generator", layout="wide")

st.title("HatchCraft 📐✨")
st.markdown("### Convierte dibujos y imágenes a patrones para Revit")

mode = st.radio("Selecciona el modo:",
                ["📁 DXF (AutoCAD)", "🖼️ Imagen (Canny/Skeleton/FLD)"],
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
                    st.session_state.pop("original_image", None)
                    st.success(result["stats"])
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    else:
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
                "Algoritmo:",
                ["📏 Líneas Rectas (Hough)",
                 "⚡ Líneas Precisas (LSD - Sub-pixel)",
                 "🚀 FastLineDetector (contrib)",
                 "🌿 Formas Orgánicas (Contornos)"],
                label_visibility="collapsed"
            )

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
                    merge_segments = st.checkbox("Unir segmentos colineales", value=True,
                                                help="Reduce fragmentación uniendo líneas similares")

                filter_mode = st.radio(
                    "Filtro de ruido:",
                    ["Gaussiano", "Bilateral (preserva bordes)", "Ninguno"],
                    horizontal=True,
                    help="Bilateral suaviza sin desdibujar los bordes del patrón"
                )
                filter_key = {"Gaussiano": "gaussian",
                              "Bilateral (preserva bordes)": "bilateral",
                              "Ninguno": "none"}[filter_mode]

                use_blackhat = st.checkbox("Black-hat (trazos oscuros finos)", value=False,
                                           help="Extrae marcas oscuras sobre fondo claro (tinta, sellos)")
                blackhat_ksize = 15
                if use_blackhat:
                    blackhat_ksize = st.slider("Tamaño kernel black-hat", 5, 51, 15, 2,
                                               key="bh_k",
                                               help="Debe ser mayor que el grosor del trazo")

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

                use_auto_canny = False
                if not use_adaptive:
                    use_auto_canny = st.checkbox("Auto-Canny (mediana ±33%)", value=True,
                                                 help="Calcula umbrales automáticamente desde la imagen")
                if not use_auto_canny:
                    canny_low = st.slider("Canny Low", 10, 200, 50, key="canny_low2")
                    canny_high = st.slider("Canny High", 50, 300, 150, key="canny_high2")
                else:
                    canny_low, canny_high = 50, 150

                dedup_auto = st.checkbox("Dedup automático por grosor de trazo", value=True,
                                         help="Estima el grosor del trazo y filtra duplicados acorde")
                if dedup_auto:
                    dedup_k = st.slider("Sensibilidad dedup (× grosor)", 1.0, 6.0, 2.5, 0.5,
                                        key="dedup_k")
                    dedup_threshold = 8.0
                else:
                    dedup_k = 2.5
                    dedup_threshold = st.slider("🚫 Filtro líneas dobles (px)", 0, 30, 8, 1,
                                                key="dedup",
                                                help="Distancia mínima entre líneas para NO considerarlas duplicadas")

            st.caption("⚙️ Parámetros de visión")
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                blur_size = st.slider("Desenfoque (Limpiar ruido)", 1, 11, 3, 2, key="blur")

            with col_s2:
                if "Hough" in detect_method:
                    p1 = st.slider("Longitud mín. de línea", 5, 100, 20, key="hough_min")
                    p2 = st.slider("Unir huecos en líneas (Gap)", 1, 50, 5, key="hough_gap")
                    method_key = "hough"
                elif "LSD" in detect_method:
                    p1 = st.slider("Longitud mín. de línea", 5, 100, 15, key="lsd_min")
                    p2 = 0
                    method_key = "lsd"
                elif "FastLineDetector" in detect_method:
                    p1 = st.slider("Longitud mín. de línea (FLD)", 5, 100, 20, key="fld_min")
                    p2 = 0
                    method_key = "fld"
                else:
                    p1 = st.slider("Longitud mín. contorno", 5, 100, 20, key="cont_min")
                    p2 = st.slider("Suavizado de curvas", 0.001, 0.05, 0.005, format="%.3f",
                                   key="cont_eps")
                    method_key = "contour"

            st.caption("📐 Alinear Patrón y Escala")
            col_off1, col_off2 = st.columns(2)
            with col_off1:
                offset_x = st.slider("Offset X", 0.0, 1.0, 0.0, 0.01, key="off_x")
            with col_off2:
                offset_y = st.slider("Offset Y", 0.0, 1.0, 0.0, 0.01, key="off_y")

            col_esc1, col_esc2 = st.columns(2)
            with col_esc1:
                tile_mm = st.number_input(
                    "Escala real del tile (mm)", 1.0, 5000.0, 100.0, 10.0, key="tile_mm",
                    help="Tamaño físico que tendrá el tile completo al importarlo en Revit"
                )
            with col_esc2:
                max_res = st.slider("Resolución de trabajo (px)", 200, 800, 600, 50,
                                    key="max_res",
                                    help="Menor = más rápido. Mayor = más detalle.")

            with st.expander("📏 Avanzado: dashes mínimos (mm)", expanded=False):
                c_md1, c_md2 = st.columns(2)
                with c_md1:
                    min_dash_mm = st.number_input("Dash mínimo (mm)", 0.05, 10.0, 0.3, 0.05,
                                                  key="min_dash_mm")
                with c_md2:
                    min_gap_mm = st.number_input("Gap mínimo (mm)", 0.05, 10.0, 0.3, 0.05,
                                                 key="min_gap_mm")

            image_bytes = uploaded_file.getvalue()
            img_hash = hashlib.md5(image_bytes).hexdigest()
            result = cached_convert(
                img_hash,
                image_bytes=image_bytes, method=method_key,
                canny_low=canny_low, canny_high=canny_high,
                blur_size=blur_size, param1=p1, param2=p2,
                use_clahe=use_clahe, clahe_clip=clahe_clip,
                use_adaptive=use_adaptive, adaptive_block=adaptive_block, adaptive_c=adaptive_c,
                use_auto_canny=use_auto_canny,
                filter_mode=filter_key,
                use_blackhat=use_blackhat, blackhat_ksize=blackhat_ksize,
                use_skeleton=use_skeleton,
                merge_segments=merge_segments,
                dedup_auto=dedup_auto, dedup_k=dedup_k,
                dedup_threshold=float(dedup_threshold),
                offset_x=offset_x, offset_y=offset_y,
                max_resolution=max_res,
                tile_mm=float(tile_mm),
                min_dash_mm=float(min_dash_mm),
                min_gap_mm=float(min_gap_mm)
            )

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.result = result
                st.session_state.original_image = image_bytes
                st.caption(result["stats"])

with col2:
    st.subheader("🔲 Resultado")

    if 'result' in st.session_state and st.session_state.result:
        result = st.session_state.result

        for w in result.get("warnings") or []:
            st.warning(w)

        tab_debug, tab_preview, tab_code, tab_download = st.tabs([
            "🔍 Comparativa", "🔲 Preview", "📄 Código", "📥 Descargar"
        ])

        with tab_debug:
            col_o, col_d, col_r = st.columns(3)
            with col_o:
                st.caption("Original")
                if "original_image" in st.session_state and st.session_state.original_image:
                    st.image(st.session_state.original_image, use_container_width=True)
                elif "debug_img" in result:
                    st.image(result["debug_img"], use_container_width=True)
                else:
                    st.info("Sin imagen original")
            with col_d:
                st.caption("Detectado")
                if "debug_img" in result and "original_image" in st.session_state:
                    st.image(result["debug_img"], use_container_width=True)
                else:
                    st.info("Modo DXF: ver Original")
            with col_r:
                st.caption("Simulación Revit (fiel)")
                pat_preview = render_pat_preview(result["pat_content"], tile_count=3,
                                                 preview_size=400, manual_scale=1.0)
                st.image(pat_preview, use_container_width=True)

            px = result.get("period_x_px")
            py = result.get("period_y_px")
            if px or py:
                sx = f"{px:.0f}px" if px else "—"
                sy = f"{py:.0f}px" if py else "—"
                st.info(
                    f"🔁 Período detectado: X={sx} · Y={sy}. "
                    f"Ajusta el Offset para centrar el tile en un ciclo del patrón."
                )

        with tab_preview:
            preview_scale = st.slider("🔍 Escala", 0.1, 10.0, 1.0, 0.1)
            pat_preview = render_pat_preview(result["pat_content"], tile_count=3,
                                             preview_size=600, manual_scale=preview_scale)
            st.image(pat_preview, caption="Simulación fiel del tileado en Revit (3x3)",
                     use_container_width=True)

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
- **DXF**: explosión completa de entidades (INSERT, polilíneas, arcos, splines). Ángulos exactos, sin cuantizar.
- **Imagen → Hough**: Detecta líneas rectas. Ideal para baldosas y geometría.
- **Imagen → LSD**: Detector sub-pixel auto-tuning. El más preciso para líneas.
- **Imagen → FLD**: FastLineDetector (opencv-contrib), rápido y robusto con fusión de segmentos.
- **Imagen → Contornos**: Formas orgánicas (piedra, texturas).
- **Pre-procesamiento**: CLAHE, Black-hat, Bilateral, Auto-Canny, Adaptive Threshold, Skeleton, Merge.
- **Escala física**: define el tamaño real del tile en mm para cotas correctas en Revit.
""")
