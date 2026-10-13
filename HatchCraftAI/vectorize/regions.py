"""F3b — Regiones: k-means Lab + eje medio de juntas compartidas.

Para fotos de materiales (adoquín, piedra, madera) la junta entre piezas
es una BANDA de color distinto (mortero) o una FRONTERA entre regiones.
La línea que debe ir al PAT es el EJE MEDIO de esa banda, compartido
entre las dos piezas que une (una sola línea por junta, sin dobles).

Estrategia (Vector Magic / plan F3b):
1. k-means sobre Lab del color suavizado (bilateral).
2. Filtro de mayoría espacial sobre el mapa de etiquetas (mata speckle).
3. Fase WEB: etiqueta cuyo esqueleto es una red con uniones reales
   (T/X) y cadenas largas entre uniones = mortero/juntas. Su EJE MEDIO
   (esqueleto de la fase) es la red de juntas: una línea por junta.
   El mortero puede ser mayoría (p. ej. 60-80% de la foto): lo que lo
   define como junta es la TOPOLOGÍA de red, no su grosor relativo.
4. Sin fase web (checkerboard, regiones compactas que se tocan) ->
   banda de frontera entre etiquetas distintas (línea compartida).
5. El esqueleto de la máscara de juntas es el eje medio; de ahí en
   fuera se reutiliza TODA la maquinaria F3a/F5/F6 (cadenas,
   primitivas, regularización, discretización).
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .centerline import build_graph


def _odd(n):
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1


def _majority_labels(labels, ksize):
    """Filtro de mayoría por etiqueta (box one-hot por slice)."""
    k = _odd(ksize)
    n = int(labels.max()) + 1
    votes = np.stack(
        [cv2.boxFilter((labels == i).astype(np.float32), -1, (k, k))
         for i in range(n)],
        axis=2)
    return np.argmax(votes, axis=2).astype(np.uint8)


def _label_stats(mask_bool):
    """(stroke_width, fracción) de una etiqueta binaria."""
    if not mask_bool.any():
        return 0.0, 0.0
    m = mask_bool.astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_5)
    skel = skeletonize(mask_bool)
    if skel.any():
        sw = 2.0 * float(np.median(dt[skel]))
    else:
        sw = 2.0 * float(np.median(dt[m > 0]))
    return sw, float(mask_bool.mean())


def _web_metrics(mask_bool):
    """(uniones reales, longitud de esqueleto) de una etiqueta.

    Una fase WEB (red de mortero, vetas) tiene un esqueleto con muchas
    uniones T/X y cadenas largas; una fase COMPACTA (piedras, celdas)
    tiene esqueletos de estrella cortos por blob.
    """
    skel = skeletonize(mask_bool)
    if not skel.any():
        return 0, 0
    try:
        _, _, _, junction = build_graph(skel.astype(np.uint8))
        j3 = int(junction.sum())
    except Exception:
        j3 = 0
    return j3, int(skel.sum())


def _boundary_band(labels):
    """Píxeles donde se tocan dos etiquetas distintas (banda 2px)."""
    h, w = labels.shape
    b = np.zeros((h, w), dtype=np.uint8)
    vr = (labels[:-1, :] != labels[1:, :]).astype(np.uint8)
    hr = (labels[:, :-1] != labels[:, 1:]).astype(np.uint8)
    b[:-1, :] |= vr
    b[1:, :] |= vr
    b[:, :-1] |= hr
    b[:, 1:] |= hr
    return b


def extract_joint_mask(img_bgr, ksize_hint=15, region_k=0,
                       min_joint_frac=0.02, max_joint_frac=0.95,
                       min_pixels_frac=0.001, max_kmeans_px=150000,
                       min_junctions=8):
    """Máscara de juntas (uint8 0/1) + info, o (None, info) si nada útil.

    region_k=0 -> auto (prueba K=2, luego K=3); >=2 -> K fijo.
    El ksize del filtro de mayoría es adaptativo: en fotos el grosor de
    Sauvola no representa las juntas, así que se ancla al lado de la
    imagen (side//32) con un piso 3·ksize_hint y techo 41.

    Selección de juntas:
    - "web": etiqueta con esqueleto de red (>= min_junctions uniones
      reales) y la mayor longitud media de cadena (skel/j3) = mortero.
      Su máscara ES la junta; el eje medio lo da joint_skeleton().
    - "boundary": sin fase web -> banda de frontera entre etiquetas.
    - "none": nada útil.
    """
    smooth = cv2.bilateralFilter(img_bgr, 9, 60, 60)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    if region_k and int(region_k) >= 2:
        ks = [int(region_k)]
    else:
        ks = [2, 3]
    ks = [k for k in ks if 2 <= k <= 8] or [2]
    side = min(h, w)
    ksize = _odd(min(41, max(3 * max(float(ksize_hint), 1.0),
                              9.0, side // 32)))
    ksize = min(ksize, _odd(side // 2))
    min_px = max(64, int(min_pixels_frac * h * w))
    # submuestreo para el ajuste k-means (asignación completa luego)
    n = h * w
    stride = max(1, int((n / max_kmeans_px) ** 0.5))
    if stride > 1:
        sample = np.concatenate(
            [flat[i * w:(i + 1) * w:stride] for i in range(0, h, stride)],
            axis=0)
    else:
        sample = flat

    first_smooth = None
    first_info = None
    for K in ks:
        try:
            _, _, centers = cv2.kmeans(
                sample, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            continue
        # asignar TODOS los píxeles al centro más cercano
        dists = np.stack(
            [((flat - centers[k]) ** 2).sum(axis=1) for k in range(K)],
            axis=1)
        labels = np.argmin(dists, axis=1).reshape(h, w).astype(np.uint8)
        labels_s = _majority_labels(labels, ksize)
        sws, fracs, j3s, skels = [], [], [], []
        for i in range(K):
            mask_i = (labels_s == i)
            sw, frac = _label_stats(mask_i)
            j3, skel_len = _web_metrics(mask_i)
            sws.append(sw)
            fracs.append(frac)
            j3s.append(j3)
            skels.append(skel_len)
        info = {"k": K, "fracs": [round(f, 4) for f in fracs],
                "sws": [round(s, 2) for s in sws],
                "j3": j3s, "skel": skels}
        if first_smooth is None:
            first_smooth = labels_s
            first_info = dict(info)
        # fase WEB: red de uniones + cadenas largas (mortero/vetas)
        best_i, best_mean, best_skel = -1, 0.0, 0
        for i in range(K):
            if (j3s[i] >= min_junctions
                    and min_joint_frac <= fracs[i] <= max_joint_frac):
                mean_chain = skels[i] / max(j3s[i], 1)
                if (mean_chain > best_mean + 1e-9
                        or (abs(mean_chain - best_mean) <= 1e-9
                            and skels[i] > best_skel)):
                    best_i, best_mean, best_skel = i, mean_chain, skels[i]
        if best_i >= 0 and skels[best_i] >= min_px // 4:
            info["source"] = "web"
            info["web_label"] = best_i
            info["mean_chain"] = round(best_mean, 1)
            return (labels_s == best_i).astype(np.uint8), info

    # Sin fase web: banda de frontera del primer K probado
    if first_smooth is None:
        return None, {"source": "none", "k": 0, "fracs": [], "sws": []}
    band = _boundary_band(first_smooth)
    if int(band.sum()) >= min_px:
        first_info["source"] = "boundary"
        return band, first_info
    return None, {"source": "none", "k": first_info.get("k", 0),
                  "fracs": first_info.get("fracs", []),
                  "sws": first_info.get("sws", [])}


def joint_skeleton(joint_mask):
    """(skeleton bool, stroke_width) de una máscara de juntas."""
    jb = joint_mask.astype(bool)
    if not jb.any():
        return jb, 0.0
    dt = cv2.distanceTransform(joint_mask.astype(np.uint8),
                               cv2.DIST_L2, cv2.DIST_MASK_5)
    skel = skeletonize(jb)
    if skel.any():
        sw = 2.0 * float(np.median(dt[skel]))
    else:
        sw = 2.0 * float(np.median(dt[jb]))
    return skel, max(sw, 1.0)
