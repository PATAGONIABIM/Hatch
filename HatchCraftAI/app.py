import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import io
from core_logic import lines_to_pat, render_pat_preview, extract_lines_from_canvas

st.set_page_config(page_title="HatchCraft Editor", layout="wide")

st.title("HatchCraft Editor ✏️🧱")
st.markdown("### Dibuja tu patrón y conviértelo a .PAT para Revit")

# Inicializar session state
if 'lines' not in st.session_state:
    st.session_state.lines = []
if 'bg_image' not in st.session_state:
    st.session_state.bg_image = None

# Sidebar - Controles
with st.sidebar:
    st.header("🎨 Herramientas")
    
    # Cargar imagen de fondo
    st.subheader("📷 Imagen de Fondo")
    uploaded_bg = st.file_uploader("Cargar imagen para calcar", type=["png", "jpg", "jpeg"])
    
    if uploaded_bg:
        bg_img = Image.open(uploaded_bg)
        # Redimensionar a cuadrado
        size = min(bg_img.size)
        bg_img = bg_img.crop((0, 0, size, size))
        bg_img = bg_img.resize((500, 500))
        st.session_state.bg_image = bg_img
        st.success("✅ Imagen cargada")
    
    bg_opacity = st.slider("Opacidad del fondo", 0.1, 1.0, 0.5)
    
    if st.button("🗑️ Quitar fondo"):
        st.session_state.bg_image = None
        st.rerun()
    
    st.divider()
    
    # Modo de dibujo
    st.subheader("✏️ Modo de Dibujo")
    drawing_mode = st.radio(
        "Herramienta",
        ["Línea libre", "Línea recta"],
        help="Línea recta: clic-arrastrar para dibujar"
    )
    
    ortho_mode = st.checkbox("🔲 Modo Ortogonal (solo H/V)", value=False,
                             help="Fuerza líneas a 0° o 90°")
    
    stroke_width = st.slider("Grosor de línea", 1, 5, 2)
    stroke_color = st.color_picker("Color de línea", "#000000")
    
    st.divider()
    
    # Acciones
    st.subheader("⚡ Acciones")
    canvas_size = 500
    
    if st.button("🗑️ Limpiar Canvas", use_container_width=True):
        st.session_state.lines = []
        st.rerun()

# Columnas principales
col_canvas, col_preview = st.columns([1, 1])

with col_canvas:
    st.subheader("📐 Canvas de Dibujo")
    
    # Preparar imagen de fondo como numpy array
    background_image = None
    if st.session_state.bg_image:
        bg = st.session_state.bg_image.copy()
        bg = bg.convert("RGB")
        # Aplicar opacidad mezclando con blanco
        bg_array = np.array(bg, dtype=np.float32)
        white = np.ones_like(bg_array) * 255
        blended = (bg_array * bg_opacity + white * (1 - bg_opacity)).astype(np.uint8)
        background_image = Image.fromarray(blended)
    
    # Canvas de dibujo
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        background_image=background_image,
        height=canvas_size,
        width=canvas_size,
        drawing_mode="line" if drawing_mode == "Línea recta" else "freedraw",
        key="canvas",
    )
    
    st.caption("Dibuja las líneas del patrón. Para calcar, carga una imagen de fondo.")

with col_preview:
    st.subheader("🔲 Preview del Patrón")
    
    # Extraer líneas del canvas
    lines = extract_lines_from_canvas(canvas_result)
    
    # Aplicar modo ortogonal si está activo
    if ortho_mode and lines:
        ortho_lines = []
        for line in lines:
            dx = abs(line['x2'] - line['x1'])
            dy = abs(line['y2'] - line['y1'])
            if dx > dy:
                # Hacer horizontal
                line['y2'] = line['y1']
            else:
                # Hacer vertical
                line['x2'] = line['x1']
            ortho_lines.append(line)
        lines = ortho_lines
    
    if lines:
        # Generar PAT
        pat_content = lines_to_pat(lines, canvas_size)
        
        # Renderizar preview
        pat_preview = render_pat_preview(pat_content, tile_count=3, preview_size=500)
        
        st.image(pat_preview, caption=f"Preview tileado (3x3) - {len(lines)} líneas", use_container_width=True)
        
        # Tabs para código y descarga
        tab1, tab2 = st.tabs(["📄 Código .PAT", "📥 Descargar"])
        
        with tab1:
            st.code(pat_content, language="text")
        
        with tab2:
            st.download_button(
                "📥 Descargar .PAT para Revit",
                pat_content,
                "HatchCraft_Manual.pat",
                "text/plain",
                use_container_width=True
            )
            st.info("Importa este archivo en Revit: Manage → Additional Settings → Fill Patterns")
    else:
        st.info("👈 Dibuja líneas en el canvas para generar el patrón")
        
        # Mostrar canvas vacío como preview
        empty_preview = np.ones((500, 500, 3), dtype=np.uint8) * 240
        st.image(empty_preview, caption="Sin líneas dibujadas", use_container_width=True)

# Lista de líneas editable (en expansor)
with st.expander("📋 Lista de Líneas (Editar manualmente)"):
    if lines:
        st.markdown("**Líneas detectadas:**")
        for i, line in enumerate(lines):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.text(f"#{i+1}")
            with col2:
                st.text(f"Inicio: ({line['x1']:.0f}, {line['y1']:.0f})")
            with col3:
                st.text(f"Fin: ({line['x2']:.0f}, {line['y2']:.0f})")
            with col4:
                dx = line['x2'] - line['x1']
                dy = line['y2'] - line['y1']
                length = (dx**2 + dy**2) ** 0.5
                st.text(f"L: {length:.1f}px")
    else:
        st.info("No hay líneas dibujadas")