"""F1 — Preprocesamiento de precisión.

Reemplaza a Canny genérico para el método "precision":

1. Escalado de trabajo: agranda imágenes pequeñas (x2-x4, INTER_CUBIC) para
   que el refinamiento sub-píxel tenga resolución espacial de sobra, con
   tope para no disparar el coste.
2. Corrección de iluminación: dividir por el fondo gaussiano (viñeteo /
   escaneo desigual).
3. Suavizado bilateral (elimina ruido sin fundir trazos vecinos).
4. Binarización local Sauvola (adaptativa, robusta al fondo variable).
5. Campo de tinta float 0..1 (oscuro = tinta) para el refinamiento
   sub-píxel posterior.
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def _odd(n):
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def upscale_factor(side, min_work=800, max_work=2000):
    """Factor de escalado entero según el lado de entrada (px)."""
    if side < 250:
        f = 4
    elif side < 450:
        f = 3
    elif side < min_work:
        f = 2
    else:
        f = 1
    while f > 1 and side * f > max_work:
        f -= 1
    return f


def preprocess(img_bgr, min_work=800, max_work=2000, organic=False):
    """Devuelve dict con gray (uint8 suavizado+corregido), ink (float 0..1),
    mask (uint8 0/1, 1=tinta), scale (int) y size (h, w) de trabajo.

    organic=True (Fase 2.2): además de Sauvola se recupera la tinta tenue
    con filtro de crestas oscuras (sato/meijering) + histéresis: los tramos
    débiles conectados a píxeles fuertes se conservan, las manchas de fondo
    no entran.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    f = upscale_factor(min(h, w), min_work, max_work)
    if f > 1:
        gray = cv2.resize(gray, (w * f, h * f), interpolation=cv2.INTER_CUBIC)
    gray = np.ascontiguousarray(gray)
    h, w = gray.shape

    # 1) Corrección de iluminación: dividir por el fondo gaussiano.
    sigma = max(15.0, 0.05 * min(h, w))
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    background = np.maximum(background.astype(np.float32), 30.0)
    corrected = np.clip(
        gray.astype(np.float32) * 255.0 / background, 0.0, 255.0
    ).astype(np.uint8)

    # 2) Suavizado que preserva bordes.
    smoothed = cv2.bilateralFilter(corrected, d=5, sigmaColor=25, sigmaSpace=25)

    # 3) Binarización local adaptativa (Sauvola).
    win = _odd(max(15.0, min(h, w) / 20.0))
    win = min(win, 151)
    thresh = threshold_sauvola(smoothed, window_size=win)
    mask = (smoothed < thresh).astype(np.uint8)

    # 4) Campo de tinta: intensidad de oscuridad 0..1 (para sub-píxel).
    ink = np.clip((255.0 - smoothed.astype(np.float32)) / 255.0, 0.0, 1.0)

    # 5) Fase 2.2 (modo orgánico): rescatar vetas tenues que Sauvola corta.
    if organic:
        mask = _recover_faint_ridges(smoothed, mask, ink)

    return {
        "gray": smoothed,
        "ink": np.ascontiguousarray(ink),
        "mask": np.ascontiguousarray(mask),
        "scale": int(f),
        "size": (h, w),
    }


def _recover_faint_ridges(smoothed, mask, ink):
    """Suma a `mask` las crestas oscuras débiles conectadas a tinta fuerte.

    Usa skimage.sato (multiescala, black_ridges) sobre la imagen corregida
    y histéresis: semillas = respuesta alta (trazos ya visibles), crecimiento
    = respuesta media (vetas tenues unidas a ellas). El filtro de ink evita
    semillas sobre fondo liso.
    """
    from skimage.filters import apply_hysteresis_threshold, sato

    inv = (mask == 0).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dt[mask > 0]
    sw = 2.0 * float(np.median(vals)) if vals.size else 3.0
    sw = max(2.0, sw)
    sigmas = [max(0.5, 0.5 * sw), max(1.0, sw), max(1.5, 1.5 * sw)]

    resp = sato(smoothed.astype(np.float64) / 255.0,
                sigmas=sigmas, black_ridges=True)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
    pos = resp[resp > 0.0]
    if pos.size == 0:
        return mask
    high = float(np.percentile(pos, 98.0))
    if high <= 0.0:
        return mask
    ridge = apply_hysteresis_threshold(resp, 0.3 * high, high)
    ridge &= ink >= 0.12
    if not ridge.any():
        return mask
    return np.maximum(mask, ridge.astype(np.uint8))
