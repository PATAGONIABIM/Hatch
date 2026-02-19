# HatchCraft AI 📐✨

**HatchCraft AI** is a powerful tool designed to convert architectural drawings (DXF) and raster images (JPG/PNG) into **Revit Fill Patterns (.PAT)**. It simplifies the creation of complex custom hatches, supporting both precise geometric designs and organic textures.

## Features

- **DXF Mode**: Convert AutoCAD line drawings into precise .PAT files.
  - Automatically handles angles and segmentation.
  - Ideal for floor tiles, brick layouts, and technical grids.
- **Image Mode**: precise Computer Vision algorithms to detect patterns from images.
  - **Hough Transform Mode**: Detects perfect straight lines (great for geometric scans).
  - **Contour Mode**: Detects organic shapes and curves (great for stone walls, vegetation, soil).
  - **Smart Filtering**: Removes double lines and optimizes geometry for Revit performance.
- **Real-time Preview**: Vizualize your pattern before exporting.
- **Revit Ready**: Generates files compatible with Revit's `Import Fill Pattern` tool.

---

## Installation

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1. Select your mode: **DXF** or **Image**.
2. Upload your file.
3. Adjust parameters (Thresholds, Blur, Min Length, etc.) in real-time.
4. Check the **Preview** tab.
5. Download the `.PAT` file and import it into Revit.

---
---

# HatchCraft AI (Español) 📐✨

**HatchCraft AI** es una herramienta potente diseñada para convertir dibujos arquitectónicos (DXF) e imágenes rasterizadas (JPG/PNG) en **Patrones de Relleno de Revit (.PAT)**. Simplifica la creación de hachurados personalizados complejos, soportando tanto diseños geométricos precisos como texturas orgánicas.

## Características

- **Modo DXF**: Convierte dibujos de líneas de AutoCAD en archivos .PAT precisos.
  - Maneja automáticamente ángulos y segmentación.
  - Ideal para baldosas, aparejos de ladrillo y grillas técnicas.
- **Modo Imagen**: Algoritmos de Visión por Computadora para detectar patrones desde fotos.
  - **Modo Hough Transform**: Detecta líneas rectas perfectas (ideal para escaneos geométricos).
  - **Modo Contornos**: Detecta formas orgánicas y curvas (ideal para muros de piedra, vegetación, tierra).
  - **Filtrado Inteligente**: Elimina líneas dobles ("fantasmas") y optimiza la geometría para el rendimiento de Revit.
- **Vista Previa en Tiempo Real**: Visualiza tu patrón antes de exportar.
- **Listo para Revit**: Genera archivos compatibles con la herramienta `Import Fill Pattern` de Revit.

## Instalación

1. Clona el repositorio.
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Ejecuta la aplicación Streamlit:

```bash
streamlit run app.py
```

1. Selecciona tu modo: **DXF** o **Imagen**.
2. Sube tu archivo.
3. Ajusta los parámetros (Umbrales, Desenfoque, Longitud Mínima, etc.) en tiempo real.
4. Revisa la pestaña de **Vista Previa (Preview)**.
5. Descarga el archivo `.PAT` e impórtalo en Revit.
