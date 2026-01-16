import streamlit as st
import numpy as np
from core_logic import DXFtoPatConverter, render_pat_preview
import tempfile
import os

st.set_page_config(page_title="HatchCraft - DXF to PAT", layout="wide")

st.title("HatchCraft DXF → PAT 📐")
st.markdown("### Convierte dibujos de AutoCAD a patrones para Revit")

st.info("""
**Instrucciones:**
1. Dibuja tu patrón en **AutoCAD** usando solo **líneas** (LINE o POLYLINE)
2. Guarda como **DXF** (File → Save As → DXF)
3. Sube el archivo DXF aquí
4. Descarga el archivo .PAT para Revit

**Tip:** Dibuja el patrón en un cuadrado de 1x1 unidades para mejor escala.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Subir DXF")
    
    uploaded_file = st.file_uploader(
        "Arrastra tu archivo DXF aquí",
        type=["dxf"]
    )
    
    if uploaded_file:
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        
        # Guardar temporalmente el archivo
        tmp_path = None
        try:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf', mode='wb') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Convertir
            with st.spinner("🔄 Convirtiendo DXF a PAT..."):
                converter = DXFtoPatConverter()
                result = converter.convert(tmp_path)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.result = result
                st.success(result["stats"])
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
        finally:
            # Limpiar archivo temporal
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

with col2:
    st.subheader("🔲 Resultado")
    
    # Slider de escala
    preview_scale = st.slider("🔍 Escala del Preview", 0.1, 10.0, 1.0, 0.1, 
                              help="Ajusta para ver mejor el patrón")
    
    if 'result' in st.session_state and st.session_state.result:
        result = st.session_state.result
        
        # Regenerar preview con la escala seleccionada
        pat_preview = render_pat_preview(result["pat_content"], tile_count=3, 
                                         preview_size=600, manual_scale=preview_scale)
        
        # Preview
        st.image(pat_preview, caption="Preview tileado (3x3)", use_container_width=True)
        
        # Tabs
        tab1, tab2 = st.tabs(["📄 Código .PAT", "📥 Descargar"])
        
        with tab1:
            st.code(result["pat_content"], language="text")
        
        with tab2:
            st.download_button(
                "📥 Descargar .PAT para Revit",
                result["pat_content"],
                "HatchCraft_DXF.pat",
                "text/plain",
                use_container_width=True
            )
            st.info("**En Revit:** Manage → Additional Settings → Fill Patterns → Import")
    else:
        # Placeholder
        empty_img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        st.image(empty_img, caption="El patrón convertido aparecerá aquí")
        st.info("👈 Sube un archivo DXF para convertir")

# Footer
st.divider()
st.markdown("""
**Formatos soportados:**
- Entidades LINE
- Entidades LWPOLYLINE (polylines)
- Coordenadas en cualquier unidad (se normalizan automáticamente)

**Limitaciones:**
- Solo líneas rectas (no arcos, círculos o splines)
- Ángulos se redondean a 0°, 45°, 90° o 135°
""")