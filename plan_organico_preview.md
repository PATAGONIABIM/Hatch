# Plan: preview 3×3 fiel + detección de líneas orgánicas

## 0. Diagnóstico (verificado con scripts)

### El preview funciona: el `.pat` está mal
El tile inferior izquierdo sale bien y los otros 8 salen como "ruido". El simulador reproduce lo que haría Revit con ese `.pat`: **el problema está en el compilador**.

Prueba con un segmento por ángulo, midiendo qué parte de cada tile del 3×3 queda cubierta ([diag_tiles.py](file:///C:/Users/chris/.gemini/antigravity-ide/brain/64f42e61-5ec6-464b-ac3a-7e9cbd0d4084/scratch/diag_tiles.py)):

| Ángulo | Tile (0,0) | Otros 8 tiles | Copias dibujadas (deberían ser 9) |
|---|---|---|---|
| 0° / 90° | 100% | 100% | 9 ✅ |
| 45° | 100% | 100% | **25** (líneas fantasma) |
| 77° / 86° / 100° | 100% | **0%** | 9–11, todas mal ubicadas |

**Causa 1: el compilador.** [compile_pat_perline](file:///d:/Hatch/HatchCraftAI/pat_compiler.py#L329-L428) emite `delta=(tile,tile)` y `dash+gap=tile` para cualquier ángulo. Eso solo repite bien a 0° y 90°. A 86°, el siguiente dash cae en `T·(cos86, sin86) = (0.07T, 0.998T)` y no en `(0, T)`. Cada tile se va corriendo más, y Revit dibujará exactamente lo mismo. Además, [get_pat_shift](file:///d:/Hatch/HatchCraftAI/pat_compiler.py#L312-L326) usa `0.7071·T` a 45°, lo que duplica la densidad. Como en las imágenes orgánicas casi todo cae entre 77° y 104°, casi todo se rompe.

**Causa 2: el simulador (estaba oculta).** [iter_entry_segments](file:///d:/Hatch/HatchCraftAI/pat_sim.py#L93-L106) calcula el rango de miembros `k` proyectando sobre el vector delta y no sobre la perpendicular a la línea. Hoy no se nota, pero apenas se arregle el compilador, el preview perdería copias.

### Solución ya probada en prototipo
Una familia PAT solo se repite con el tile si su dirección es un **vector del retículo** `(n·Tw, m·Th)`, con `n` y `m` coprimos. Entonces:
- `θ = atan2(m·Th, n·Tw)`. Se elige el par `(n, m)` de menor `|L|` que deje un error en los extremos ≤ `tol_mm`.
- Ciclo de dash `= |L|`: `dash = len`, `gap = -(|L| - len)`.
- Bézout: `n·q − m·p = 1`, de donde `dy = Tw·Th/|L|` (perpendicular) y `dx = (p·n·Tw² + q·m·Th²)/|L|` (paralelo).

Resultado con el compilador prototipo y el simulador corregido ([test_proto.py](file:///C:/Users/chris/.gemini/antigravity-ide/brain/64f42e61-5ec6-464b-ac3a-7e9cbd0d4084/scratch/test_proto.py)): **100% de cobertura en los 9 tiles y exactamente 9 copias** a 0°, 45°, 77°, 86°, 89°, 100°, 104° y 33.3°. El error angular máximo fue de 0.044 mm.

````carousel
![Antes: con vetas sintéticas se reproduce tu bug (solo el tile (0,0) sale bien)](C:/Users/chris/.gemini/antigravity-ide/brain/64f42e61-5ec6-464b-ac3a-7e9cbd0d4084/preview_antes.png)
<!-- slide -->
![Después: entradas por retículo + simulador corregido, 9 tiles idénticos](C:/Users/chris/.gemini/antigravity-ide/brain/64f42e61-5ec6-464b-ac3a-7e9cbd0d4084/preview_despues.png)
````

---

## Fase 1: Arreglar el preview y el `.pat` (prioridad alta)

### 1.1 Compilador por retículo: [pat_compiler.py](file:///d:/Hatch/HatchCraftAI/pat_compiler.py)
- Nuevas funciones `lattice_direction(theta, length, tw, th, tol_mm, n_max, ratio_max)` y `lattice_entry(seg, tw, th, tol_mm)`, a partir del prototipo [lattice_proto.py](file:///C:/Users/chris/.gemini/antigravity-ide/brain/64f42e61-5ec6-464b-ac3a-7e9cbd0d4084/scratch/lattice_proto.py).
- `compile_pat_perline` pasa a usar `lattice_entry` para **todo ángulo que no sea exactamente 0° o 90°**. Las entradas a 0° y 90° quedan byte a byte iguales, así que los goldens ortogonales no cambian.
- Se elimina `get_pat_shift` (0.7071), que provoca los fantasmas a 45°.
- Casos borde:
  - Si `len ≥ |L|`, la línea es continua (`dashes=[]`).
  - Si ningún `(n, m)` cumple la tolerancia, el segmento se **divide en dos** de forma recursiva (cada mitad tolera más error angular).
  - Tiles no cuadrados (F7): la fórmula general en `Tw, Th` ya los cubre.
- Nuevo parámetro `angle_tol_mm`, por defecto `min(0.05, sagitta_mm/2)`.
- Topes `n_max≈64` y `ratio_max=|L|/T≈60`, para no generar familias demasiado densas en Revit (ver 1.6).

### 1.2 Cuantización que respete las cadenas (conectividad)
Rotar cada cuerda sobre su punto medio abre microgaps (≤ tol) en los vértices compartidos. Propuesta:
- El pipeline expone `chain_ids`, en paralelo a `segments_px`.
- Al compilar, se cuantiza la cadena entera en secuencia: el inicio de cada cuerda se ancla al final de la anterior y el error se reparte. La deriva queda acotada por `tol_mm`.

### 1.3 Simulador fiel: [pat_sim.py](file:///d:/Hatch/HatchCraftAI/pat_sim.py)
- Rango `k` calculado con la proyección **perpendicular** `v=(-uy,ux)` dividida por `dy` (ya probado en [sim_fixed.py](file:///C:/Users/chris/.gemini/antigravity-ide/brain/64f42e61-5ec6-464b-ac3a-7e9cbd0d4084/scratch/sim_fixed.py)).
- `max_segments` adaptado al número de entradas. Si se alcanza el tope, devolver `truncated=True` para que la UI avise, en vez de cortar en silencio.

### 1.4 Aplicarlo a todas las rutas
| Ruta | Archivo |
|---|---|
| Imagen → PAT | [core_logic.py:L788](file:///d:/Hatch/HatchCraftAI/core_logic.py#L788-L796) |
| Edición en lienzo (RO/EX/LINEA/DEL) | [api/main.py:L418](file:///d:/Hatch/HatchCraftAI/api/main.py#L418-L422) |
| DXF → PAT (15°: 15/30/60/75° también se rompen hoy) | [core_logic.py:L169](file:///d:/Hatch/HatchCraftAI/core_logic.py#L169) |

### 1.5 Retroalimentación en la UI: [main.js](file:///d:/Hatch/HatchCraftAI/web/js/main.js)
- Badge **"Fidelidad de tiling: 100%"**, calculado en el backend al comparar cada tile con el (0,0).
- Aviso cuando `truncated` o cuando haya entradas con `|L|/T` alto ("patrón denso para Revit").

### 1.6 Tests y validación en Revit
- Nuevo `tests/test_tiling_fidelity.py`, con ángulos de 0 a 180° cada 0.5°, segmentos aleatorios y tiles cuadrados y no cuadrados. Comprueba cobertura = 100% y copias = N por tile.
- Volver a correr `test_perline_pat.py`, `test_compiler.py`, `test_golden_*` y `test_api.py`.
- **Prueba manual en Revit 2025/2026:** importar `.pat` con 86°, 100°, 33° y la imagen de vetas. Confirmar que no aparece "líneas demasiado juntas" y calibrar `ratio_max`.

---

## Fase 2: Detección de líneas orgánicas

Lo que muestra tu captura: trazos azules fragmentados, piezas grises corridas en paralelo, puntos verdes y violetas sueltos, y "Dominantes: 77, 86, 89, 96, 100, 104" con `Auto[snap 2.0°, 155°]`.

### 2.1 Clasificador orgánico / arquitectónico: [regularize.py](file:///d:/Hatch/HatchCraftAI/vectorize/regularize.py) + [pipeline.py](file:///d:/Hatch/HatchCraftAI/vectorize/pipeline.py)
- **Problema:** con 6 "dominantes" en solo 27°, la compuerta de cobertura (`min_coverage=0.6`) deja pasar el snap, y el snap endereza las vetas a la fuerza. `snap_junctions` también deforma la veta, porque crea uniones T/L falsas.
- **Propuesta:** función `structure_kind()` basada en:
  - la dispersión de los dominantes (≥3 dentro de 30° → una sola familia orgánica);
  - la fracción de curvas/cuerdas;
  - la coherencia del campo de orientación (tensor de estructura).
- **En modo `organic`:**
  - sin snap angular, sin lattice y sin `snap_junctions`;
  - `corner_angle≈120°`, para cortar solo en esquinas reales (155° fragmenta la veta);
  - el auto-tune excluye combinaciones con snap.

### 2.2 Preproceso por crestas: [preprocess.py](file:///d:/Hatch/HatchCraftAI/vectorize/preprocess.py)
- Sauvola binariza localmente y corta las vetas tenues en pedazos.
- En modo orgánico se agrega un filtro multiescala de crestas oscuras (`skimage.filters.sato` o `meijering`, con `black_ridges=True` y `sigmas≈[0.5, 1, 1.5]·sw`), seguido de **histéresis** (`apply_hysteresis_threshold`). Así se conservan los tramos débiles conectados a tramos fuertes.
- Opcional: suavizado que realza la coherencia a lo largo de la orientación local, para unir los grises fragmentados.

### 2.3 Cierre de gaps global: [centerline.py](file:///d:/Hatch/HatchCraftAI/vectorize/centerline.py#L391-L468)
- **Hoy:** como máximo **una unión por pasada y 4 pasadas**, es decir, a lo sumo 4 gaps cerrados en toda la imagen. Además el alcance es 2.2·sw y el puente es recto.
- **Propuesta:**
  - Candidatos con KD-tree (`scipy.spatial.cKDTree`).
  - Costo = distancia/sw + desvío de tangente + diferencia con el campo de orientación.
  - **Matching greedy global** por costo, sin límite de uniones.
  - Alcance hasta 5·sw en modo orgánico.
  - Puente con **Hermite cúbico** que respete las tangentes, en vez de una recta de píxeles.

### 2.4 Filtrado de ruido guiado por la orientación
Se descartan cadenas cortas (< 3·sw) que se desvíen más de 25° del campo local o tengan poco contraste de tinta. Son los puntos verdes y violetas y las piezas grises sueltas.

### 2.5 Dedup que no se coma vetas vecinas: [core_logic.py:L647-L702](file:///d:/Hatch/HatchCraftAI/core_logic.py#L647-L702)
- `dedup_eff = 2.5·sw` borra rectas paralelas cercanas que se solapan. En la madera, dos vetas reales pueden estar a menos de esa distancia.
- **En orgánico:** usar `k≈1.0` y exigir un **solape ≥ 70%** de longitud, no un simple "se tocan".

### 2.6 Presupuesto adaptativo: [discretize.py](file:///d:/Hatch/HatchCraftAI/vectorize/discretize.py#L123-L133)
- Hoy el presupuesto es `budget=400` global. Si se supera, la sagita se degrada hasta ×64 y las vetas quedan poligonales.
- **Propuesta:** el presupuesto escala con la longitud total del trazo, y la UI muestra "sagita efectiva" cuando se degrada.

### 2.7 Continuidad en los bordes (las imágenes orgánicas no son periódicas)
Paso opcional "seam match": los extremos que tocan x=0 se emparejan con los de x=T (lo mismo en y), con tolerancia en la coordenada y en el ángulo, y se ajustan para que la veta siga en el tile vecino. Así no se ve la grilla en Revit.

### 2.8 Métricas y goldens: [metrics.py](file:///d:/Hatch/HatchCraftAI/metrics.py) + [test_golden_images.py](file:///d:/Hatch/HatchCraftAI/tests/test_golden_images.py)
- Métricas nuevas:
  - **recall de tinta** (esqueleto cubierto a ≤ sw), con `f1_line_pixels` como base;
  - **fragmentación** (segmentos por trazo real);
  - **fidelidad de tiling**.
- Goldens orgánicos: [madera.png](file:///d:/Hatch/HatchCraftAI/madera.png), `Flux2-Klein_00837_.png` y la imagen de tu captura.
- Usar estas métricas como objetivo del `auto_tune` ([tuning.py](file:///d:/Hatch/HatchCraftAI/vectorize/tuning.py)).

---

## Orden de ejecución y criterios de aceptación

| # | Entregable | Criterio |
|---|---|---|
| 1 | 1.1 + 1.3 + test de fidelidad | 9/9 tiles idénticos con cualquier ángulo; goldens ortogonales sin cambios |
| 2 | 1.4 + 1.5 | Imagen, lienzo y DXF con el preview correcto; badge visible |
| 3 | 1.6 Revit | `.pat` orgánico importa y se ve igual que el preview |
| 4 | 2.1 + 2.5 | Sin snap en vetas; ninguna veta paralela eliminada por dedup |
| 5 | 2.3 + 2.4 | Fragmentación −50% y sin puntos sueltos en la imagen de prueba |
| 6 | 2.2 + 2.6 | Recall de tinta ≥ 90% en goldens orgánicos |
| 7 | 2.7 + 2.8 | Bordes continuos (opcional); métricas en CI |

## Preguntas abiertas
1. **Tolerancia angular:** ¿te sirve `0.05 mm` de error máximo en los extremos? Con más tolerancia, el `.pat` queda más liviano en Revit.
2. **DXF:** ¿aplico también la corrección a la ruta DXF? Cambia la salida de los patrones a 15/30/60/75°, que hoy también se rompen.
3. **Continuidad de bordes (2.7):** ¿la incluimos ahora o después?
4. ¿Me pasas la imagen original de la captura para usarla como golden orgánico?
