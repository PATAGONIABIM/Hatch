# HATCH.it

**HATCH.it** es una herramienta que convierte dibujos arquitectónicos (DXF) e imágenes rasterizadas (JPG/PNG) en **Patrones de Relleno de Revit (.PAT)**. Simplifica la creación de hachurados personalizados, soportando diseños geométricos precisos y texturas orgánicas.

## Características

- **Modo DXF**: Convierte dibujos de líneas de AutoCAD en archivos .PAT precisos.
  - Maneja automáticamente ángulos y segmentación (explosión de INSERT, polilíneas, arcos, splines).
  - Ideal para baldosas, aparejos de ladrillo y grillas técnicas.
- **Modo Imagen**: Algoritmos de visión por computadora para detectar patrones desde imágenes.
  - **Hough**: Detecta líneas rectas perfectas (ideal para escaneos geométricos).
  - **LSD**: Detector sub-pixel auto-tuning, el más preciso para líneas.
  - **FLD**: FastLineDetector (opencv-contrib), rápido y robusto.
  - **Contornos**: Formas orgánicas y curvas (muros de piedra, vegetación, tierra).
  - **Pre-procesamiento**: CLAHE, Black-hat, Bilateral, Auto-Canny, Adaptive Threshold, Skeleton, Merge.
  - **Escala física**: define el tamaño real del tile en mm para cotas correctas en Revit.
- **Filtrado inteligente**: elimina líneas dobles y optimiza la geometría para el rendimiento de Revit.
- **Vista previa en tiempo real**: simulación fiel del tileado 3x3 en Revit.
- **Listo para Revit**: genera archivos compatibles con `Import Fill Pattern`.

## Arquitectura

- `core_logic.py` — motor de conversión (DXF e imagen a PAT), independiente de la interfaz.
- `pat_compiler.py` — compilador de entradas PAT.
- `pat_sim.py` — simulador fiel del renderizado de Revit.
- `api/main.py` — API FastAPI (JSON) que expone el motor y sirve el frontend.
- `web/` — frontend custom (HTML/CSS/JS vanilla, sin build step, i18n ES/EN).
- `app.py` — interfaz Streamlit clásica (mantenida como alternativa).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### API + frontend (recomendado)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8501
```

Abre `http://localhost:8501` — el frontend HATCH.it se sirve desde `/`.

### Streamlit (alternativa)

```bash
streamlit run app.py
```

## Endpoints API

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/health` | GET | Estado del servicio |
| `/api/convert/image` | POST (multipart) | Imagen → PAT (parámetros CV como form fields) |
| `/api/convert/dxf` | POST (multipart) | DXF → PAT |
| `/api/preview` | POST (JSON) | Renderiza un PAT con la simulación Revit (escala) |

## Tests

```bash
python -m pytest
```

## Importar en Revit

1. **Manage → Additional Settings → Fill Patterns**.
2. Clic en **New Pattern**.
3. **Custom → Import** y selecciona el archivo `.PAT`.
4. Asigna el patrón a tu material o categoría.

---

# HATCH.it (Español)

**HATCH.it** es una herramienta que convierte dibujos arquitectónicos (DXF) e imágenes rasterizadas (JPG/PNG) en **Patrones de Relleno de Revit (.PAT)**. Simplifica la creación de hachurados personalizados, soportando diseños geométricos precisos y texturas orgánicas.

## Características

- **Modo DXF**: Convierte dibujos de líneas de AutoCAD en archivos .PAT precisos.
  - Maneja automáticamente ángulos y segmentación.
  - Ideal para baldosas, aparejos de ladrillo y grillas técnicas.
- **Modo Imagen**: Algoritmos de visión por computadora para detectar patrones desde imágenes.
  - **Hough Transform**: Detecta líneas rectas perfectas (ideal para escaneos geométricos).
  - **Contornos**: Detecta formas orgánicas y curvas (ideal para muros de piedra, vegetación, tierra).
  - **Filtrado inteligente**: Elimina líneas dobles y optimiza la geometría para el rendimiento de Revit.
- **Vista previa en tiempo real**: Visualiza tu patrón antes de exportar.
- **Listo para Revit**: Genera archivos compatibles con la herramienta `Import Fill Pattern` de Revit.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### API + frontend (recomendado)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8501
```

Abre `http://localhost:8501` — el frontend HATCH.it se sirve desde `/`.

### Streamlit (alternativa)

```bash
streamlit run app.py
```

1. Selecciona tu modo: **DXF** o **Imagen**.
2. Sube tu archivo.
3. Ajusta los parámetros (umbrales, desenfoque, longitud mínima, etc.).
4. Revisa la vista previa.
5. Descarga el archivo `.PAT` e impórtalo en Revit.