"""F3a — Centro de trazo sub-píxel (modo centerline).

Pipeline sobre el esqueleto de la tinta:

1. Grafo de píxeles del esqueleto: nodos = extremos (grado 1) y uniones
   (grado >= 3, agrupando blobs de unión adyacentes); aristas = cadenas de
   píxeles de grado 2.
2. Poda iterativa de espolones (< 1.5 x grosor).
3. Fusión de cadenas a través de uniones degeneradas (2 cadenas vivas).
4. Cierre de gaps entre extremos colgantes (distancia + tangente +
   cobertura de tinta en el puente).
5. Refinamiento sub-píxel: centroide ponderado por tinta^2 del perfil
   perpendicular al trazo, muestreado bilinealmente.
6. Colocación exacta de uniones: intersección en mínimos cuadrados de las
   rectas TLS terminales de las cadenas incidentes.
7. Cadena -> primitivas: RDP con tolerancia ABSOLUTA + ajuste TLS; en S1
   las curvas quedan como cuerdas acotadas (arcos/Bézier en S2).
"""

import math

import numpy as np
import cv2

from geometry import fit_line_tls, simplify_polyline_adaptive, max_perp_deviation

_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# --------------------------------------------------------------- grafo


def _direction_clusters(flat, w, idx, tol_deg=60.0):
    """Número de grupos de direcciones de los vecinos 8-conn de `idx`.

    Una unión REAL tiene >= 3 direcciones bien separadas. Los píxeles con
    grado >= 3 pero solo 1-2 grupos (esquinas de escalera del esqueleto en
    diagonales rasas) son uniones ESPURIAS: el trazo simplemente continúa.
    """
    import math as _m
    h = flat.shape[0] // w
    y, x = divmod(idx, w)
    angs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and flat[ny * w + nx]:
                angs.append(_m.degrees(_m.atan2(dy, dx)) % 360.0)
    if len(angs) < 3:
        return len(angs)
    angs.sort()
    big_gaps = 0
    for i in range(len(angs)):
        gap = (angs[(i + 1) % len(angs)] - angs[i]) % 360.0
        if gap > tol_deg:
            big_gaps += 1
    return big_gaps if big_gaps > 0 else 1


def build_graph(skel):
    """Grados y clasificación de píxeles del esqueleto (índices planos).

    node = extremos (grado 1) y uniones REALES (grado >= 3 con >= 3 grupos
    de direcciones). Las uniones espurias (escaleras) no son nodo: las
    cadenas las atraviesan con preferencia direccional.
    """
    h, w = skel.shape
    flat = skel.ravel().astype(bool)
    deg = np.zeros(flat.shape[0], np.uint16)
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = skel.astype(np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w].ravel()
    node = flat & (deg == 1)
    junction = flat & (deg >= 3)
    hi = np.flatnonzero(flat & (deg >= 3))
    for p in hi:
        if _direction_clusters(flat, w, int(p)) >= 3:
            node[p] = True
        else:
            junction[p] = False
    return flat, deg, node, junction


def _neighbors(flat, w, idx):
    h = flat.shape[0] // w
    y, x = divmod(idx, w)
    out = []
    for dy, dx in _N8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            j = ny * w + nx
            if flat[j]:
                out.append(j)
    return out


def _walk_step(flat, deg, node, visited, w, prev, cur):
    """Siguiente píxel de una cadena desde `cur` viniendo de `prev`.

    Reglas: no volver a `prev`, no repetir visitados, parar en nodos.
    Entre varios candidatos (píxeles espurios de escalera) prefiere la
    continuación más colineal con la dirección de entrada. Devuelve None
    si no hay continuación.
    """
    in_dir = np.array(divmod(cur, w)[::-1], dtype=np.float64) - \
        np.array(divmod(prev, w)[::-1], dtype=np.float64)
    n_in = np.hypot(*in_dir)
    best_j = None
    best_score = -2.0
    for j in _neighbors(flat, w, cur):
        if j == prev or visited[j]:
            continue
        if not node[j] and deg[j] < 2:
            continue
        if n_in < 1e-9:
            return j  # sin dirección de entrada: primero disponible
        d = np.array(divmod(j, w)[::-1], dtype=np.float64) - \
            np.array(divmod(cur, w)[::-1], dtype=np.float64)
        nd = np.hypot(*d)
        if nd < 1e-9:
            continue
        score = float(np.dot(in_dir / n_in, d / nd))
        if score > best_score:
            best_score = score
            best_j = j
    return best_j


def walk_chains(flat, deg, node, w):
    """Cadenas de píxeles ordenadas entre nodos (+ lazos cerrados).

    Devuelve (chains, closed): chains = listas de índices planos cuyo
    primer y último elemento son nodos (o el punto de arranque si es lazo).
    """
    chains = []
    closed = []
    visited = np.zeros(flat.shape[0], bool)

    node_idx_list = np.flatnonzero(node)
    direct_seen = set()
    for p in node_idx_list:
        p = int(p)
        for q in _neighbors(flat, w, p):
            if node[q]:
                key = (min(p, q), max(p, q))
                if key not in direct_seen:
                    direct_seen.add(key)
                    chains.append([p, int(q)])
                    closed.append(False)
                continue
            if visited[q] or deg[q] < 2:
                continue
            chain = [p]
            prev, cur = p, int(q)
            while True:
                chain.append(cur)
                if node[cur]:
                    break
                visited[cur] = True
                nxt = _walk_step(flat, deg, node, visited, w, prev, cur)
                if nxt is None:
                    break
                prev, cur = cur, nxt
            chains.append(chain)
            closed.append(False)

    # Lazos cerrados sin ningún nodo (círculos, etc.).
    remaining = np.flatnonzero(flat & (deg == 2) & ~visited)
    for start in remaining:
        start = int(start)
        if visited[start]:
            continue
        nbs = [j for j in _neighbors(flat, w, start) if deg[j] == 2]
        if len(nbs) < 2:
            visited[start] = True
            continue
        chain = [start]
        visited[start] = True
        prev, cur = start, int(nbs[0])
        while True:
            if visited[cur]:
                break
            chain.append(cur)
            visited[cur] = True
            nxt = None
            for j in _neighbors(flat, w, cur):
                if j == prev or visited[j]:
                    continue
                if flat[j] and deg[j] == 2:
                    nxt = j
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
        first, last = chain[0], chain[-1]
        is_ring = len(chain) > 4 and any(
            j == first for j in _neighbors(flat, w, last))
        chains.append(chain)
        closed.append(is_ring)

    return chains, closed


def cluster_nodes(node, junction, w):
    """Etiqueta cada nodo con un cluster. Los píxeles de unión adyacentes se
    fusionan; los extremos siempre quedan en clusters solitarios."""
    total = node.shape[0]
    labels = np.full(total, -1, np.int32)
    h = total // w
    next_id = 0
    for p in np.flatnonzero(node):
        p = int(p)
        if labels[p] != -1:
            continue
        labels[p] = next_id
        stack = [p]
        while stack:
            q = stack.pop()
            y, x = divmod(q, w)
            for dy, dx in _N8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    j = ny * w + nx
                    if node[j] and labels[j] == -1 and junction[j] and junction[q]:
                        labels[j] = next_id
                        stack.append(j)
        next_id += 1
    return labels, next_id


# --------------------------------------------------------------- cirugía


def _purge_chain(live, ci):
    for n in list(live.keys()):
        inc = live.get(n)
        if not inc:
            continue
        live[n] = [t for t in inc if t[0] != ci]
        if not live[n]:
            del live[n]


def prune_spurs(chains, chain_clusters, live, sw, factor=1.5):
    """Poda iterativa de ramas cortas muertas (espolones)."""
    removed = 0
    changed = True
    while changed:
        changed = False
        for ci in range(len(chains)):
            if chains[ci] is None:
                continue
            if len(chains[ci]) >= factor * max(sw, 1.0):
                continue
            c0, c1 = chain_clusters[ci]
            if c0 is None and c1 is None:
                continue
            l0 = len(live.get(c0, ())) if c0 is not None else 0
            l1 = len(live.get(c1, ())) if c1 is not None else 0
            if l0 == 1 or l1 == 1:
                chains[ci] = None
                if c0 is not None:
                    live[c0] = [t for t in live.get(c0, ()) if t[0] != ci]
                    if not live[c0]:
                        del live[c0]
                if c1 is not None and c1 != c0:
                    live[c1] = [t for t in live.get(c1, ()) if t[0] != ci]
                    if not live[c1]:
                        del live[c1]
                removed += 1
                changed = True
    return removed


def merge_through_deg2(chains, chain_clusters, live):
    """Fusiona cadenas que quedan pegadas a una unión con solo 2 cadenas
    vivas (p. ej. tras podar un espolón de una T)."""
    merged = 0
    changed = True
    while changed:
        changed = False
        for n in list(live.keys()):
            inc = live.get(n, [])
            if len(inc) != 2:
                continue
            (c1, e1), (c2, e2) = inc
            if c1 == c2:
                continue
            if chains[c1] is None or chains[c2] is None:
                continue
            a = list(chains[c1]) if e1 == 1 else list(chains[c1])[::-1]
            b = list(chains[c2]) if e2 == 0 else list(chains[c2])[::-1]
            chains[c1] = a + b[1:]
            far1 = chain_clusters[c1][0] if e1 == 1 else chain_clusters[c1][1]
            far2 = chain_clusters[c2][1] if e2 == 0 else chain_clusters[c2][0]
            chain_clusters[c1] = (far1, far2)
            _purge_chain(live, c1)
            _purge_chain(live, c2)
            live.setdefault(far1, []).append((c1, 0))
            live.setdefault(far2, []).append((c1, 1))
            chains[c2] = None
            merged += 1
            changed = True
            break
    return merged


def contract_junctions(chains, chain_clusters, live, sw, factor=2.0):
    """Colapsa rombos de cruce: cadenas cortas entre dos uniones REALES se
    contraen fusionando ambos clusters (las cuerdas internas del rombo se
    descartan y los brazos convergen en una única unión colocada por LS).
    """
    merged = 0
    changed = True
    while changed:
        changed = False
        for ci in range(len(chains)):
            if chains[ci] is None:
                continue
            c0, c1 = chain_clusters[ci]
            if c0 is None or c1 is None or c0 == c1:
                continue
            if len(chains[ci]) > factor * max(sw, 1.0):
                continue
            if len(live.get(c0, ())) < 3 or len(live.get(c1, ())) < 3:
                continue
            inc = live.pop(c1, [])
            live.setdefault(c0, []).extend(inc)
            for cj in range(len(chains)):
                if chains[cj] is None:
                    continue
                a, b = chain_clusters[cj]
                if a == c1:
                    a = c0
                if b == c1:
                    b = c0
                chain_clusters[cj] = (a, b)
            merged += 1
            changed = True
            break
    # descartar cuerdas internas cortas (ambos extremos en el mismo cluster)
    dropped = 0
    for ci in range(len(chains)):
        if chains[ci] is None:
            continue
        c0, c1 = chain_clusters[ci]
        if (c0 is not None and c0 == c1
                and len(chains[ci]) <= factor * max(sw, 1.0)):
            _purge_chain(live, ci)
            chains[ci] = None
            dropped += 1
    return merged, dropped


def _bilinear(img, xs, ys):
    """Muestreo bilinear vectorizado. xs/ys: arrays float de misma forma."""
    h, w = img.shape
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    v00 = img[y0, x0]
    v01 = img[y0, x0 + 1]
    v10 = img[y0 + 1, x0]
    v11 = img[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy + v11 * fx * fy)


def _ink_coverage(ink, p, q, step=0.5):
    """Cobertura media de tinta a lo largo del segmento p→q (float px)."""
    d = q - p
    dist = float(np.hypot(*d))
    if dist < 1e-6:
        return 0.0
    ts = np.arange(0.0, dist + step, step)
    xs = p[0] + d[0] / dist * ts
    ys = p[1] + d[1] / dist * ts
    vals = _bilinear(ink, xs, ys)
    return float(np.mean(vals))


def close_gaps(chains, chain_clusters, live, ink, w, sw,
               max_dist_factor=2.2, max_angle_deg=18.0,
               min_coverage=0.35, max_passes=4, organic=False):
    """Une extremos colgantes cercanos con tangentes alineadas y tinta
    continua en el puente (saltos de esqueleto por umbralizado).

    Fase 2.3: con organic=True el matching es GLOBAL (cKDTree + coste
    greedy sin límite de uniones por pasada, alcance 5·sw y puente
    Hermite cúbico que respeta las tangentes de los extremos). El modo
    geométrico clásico (≤1 unión/pasada, puente recto) queda intacto.
    """
    if organic:
        return _close_gaps_organic(chains, chain_clusters, live, ink, w, sw,
                                   max_angle_deg, min_coverage)
    joined_total = 0

    def tip_and_tangent(chain, end):
        idxs = chain if end == 0 else chain[::-1]
        tip = np.array(divmod(idxs[0], w)[::-1], dtype=np.float64)
        back_i = idxs[min(3, len(idxs) - 1)]
        back = np.array(divmod(back_i, w)[::-1], dtype=np.float64)
        t = tip - back
        n = np.hypot(*t)
        if n < 1e-9:
            return tip, None
        return tip, t / n

    for _ in range(max_passes):
        ends = []
        for ci in range(len(chains)):
            if chains[ci] is None:
                continue
            c0, c1 = chain_clusters[ci]
            for end in (0, 1):
                cl = c0 if end == 0 else c1
                if cl is None:
                    continue
                if len(live.get(cl, ())) != 1:
                    continue
                tip, t = tip_and_tangent(chains[ci], end)
                if t is None:
                    continue
                ends.append((ci, end, cl, tip, t))
        joined = False
        for i in range(len(ends)):
            if joined:
                break
            ci, ei, _, pi, ti = ends[i]
            for j in range(i + 1, len(ends)):
                cj, ej, _, pj, tj = ends[j]
                if ci == cj:
                    continue
                d = pj - pi
                dist = float(np.hypot(*d))
                if dist > max_dist_factor * max(sw, 1.0) or dist < 1e-6:
                    continue
                v = d / dist
                if float(np.degrees(np.arccos(np.clip(np.dot(ti, v), -1, 1)))) > max_angle_deg:
                    continue
                if float(np.degrees(np.arccos(np.clip(np.dot(tj, -v), -1, 1)))) > max_angle_deg:
                    continue
                if _ink_coverage(ink, pi, pj) < min_coverage:
                    continue
                # unir: ci con extremo ei al final; cj con extremo ej al inicio
                a = list(chains[ci]) if ei == 1 else list(chains[ci])[::-1]
                b = list(chains[cj]) if ej == 0 else list(chains[cj])[::-1]
                bridge = []
                n_b = int(dist)
                for k in range(1, n_b):
                    bx = int(round(pi[0] + v[0] * k))
                    by = int(round(pi[1] + v[1] * k))
                    bridge.append(by * w + bx)
                _join_chains(chains, chain_clusters, live, ci, ei, cj, ej,
                             a, b, bridge)
                joined_total += 1
                joined = True
                break
        if not joined:
            break
    return joined_total


def _join_chains(chains, chain_clusters, live, ci, ei, cj, ej, a, b, bridge):
    """Funde la cadena cj en ci (puente en medio) y actualiza clusters/live."""
    chains[ci] = a + bridge + b[1:]
    far_i = chain_clusters[ci][0] if ei == 1 else chain_clusters[ci][1]
    far_j = chain_clusters[cj][1] if ej == 0 else chain_clusters[cj][0]
    chain_clusters[ci] = (far_i, far_j)
    _purge_chain(live, ci)
    _purge_chain(live, cj)
    live.setdefault(far_i, []).append((ci, 0))
    live.setdefault(far_j, []).append((ci, 1))
    chains[cj] = None


def _hermite_bridge(pi, ti, pj, tj, w, dist):
    """Muestrea un puente Hermite cúbico p→q con tangentes ti/tj (px)."""
    n_b = max(3, int(round(dist)))
    bridge = []
    for k in range(1, n_b):
        t = k / n_b
        h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
        h10 = t ** 3 - 2.0 * t ** 2 + t
        h01 = -2.0 * t ** 3 + 3.0 * t ** 2
        h11 = t ** 3 - t ** 2
        x = h00 * pi[0] + h10 * (ti * dist)[0] + h01 * pj[0] + h11 * (tj * dist)[0]
        y = h00 * pi[1] + h10 * (ti * dist)[1] + h01 * pj[1] + h11 * (tj * dist)[1]
        idx = int(round(y)) * w + int(round(x))
        if not bridge or bridge[-1] != idx:
            bridge.append(idx)
    return bridge


def _close_gaps_organic(chains, chain_clusters, live, ink, w, sw,
                        max_angle_deg=18.0, min_coverage=0.35,
                        reach_factor=5.0, max_rounds=30):
    """Fase 2.3: matching global de extremos libres con cKDTree.

    - alcance 5·sw (vs 2.2·sw del modo geométrico);
    - coste = distancia/sw + desviación de tangente (greedy global, SIN
      límite de una unión por pasada);
    - puente Hermite cúbico que respeta las tangentes de los extremos.
    """
    from scipy.spatial import cKDTree

    reach = reach_factor * max(sw, 1.0)
    joined_total = 0
    for _ in range(max_rounds):
        ends = []
        for ci in range(len(chains)):
            if chains[ci] is None:
                continue
            c0, c1 = chain_clusters[ci]
            for end in (0, 1):
                cl = c0 if end == 0 else c1
                if cl is None:
                    continue
                if len(live.get(cl, ())) != 1:
                    continue
                idxs = chains[ci] if end == 0 else chains[ci][::-1]
                tip = np.array(divmod(idxs[0], w)[::-1], dtype=np.float64)
                back = np.array(divmod(idxs[min(3, len(idxs) - 1)], w)[::-1],
                                dtype=np.float64)
                t = tip - back
                n = np.hypot(*t)
                if n < 1e-9:
                    continue
                ends.append((ci, end, tip, t / n))
        if len(ends) < 2:
            break
        pts = np.array([e[2] for e in ends], dtype=np.float64)
        pairs = cKDTree(pts).query_pairs(r=reach, output_type="ndarray")
        if len(pairs) == 0:
            break
        cands = []
        for i, j in pairs:
            ci, ei, pi, ti = ends[i]
            cj, ej, pj, tj = ends[j]
            if ci == cj:
                continue
            d = pj - pi
            dist = float(np.hypot(*d))
            if dist < 1e-6:
                continue
            v = d / dist
            a1 = float(np.degrees(np.arccos(np.clip(np.dot(ti, v), -1, 1))))
            a2 = float(np.degrees(np.arccos(np.clip(np.dot(tj, -v), -1, 1))))
            if a1 > max_angle_deg or a2 > max_angle_deg:
                continue
            if _ink_coverage(ink, pi, pj) < min_coverage:
                continue
            cost = dist / max(sw, 1.0) + (a1 + a2) / max_angle_deg
            cands.append((cost, i, j))
        if not cands:
            break
        cands.sort(key=lambda c: c[0])
        used = set()
        for _, i, j in cands:
            ci, ei, pi, ti = ends[i]
            cj, ej, pj, tj = ends[j]
            if chains[ci] is None or chains[cj] is None:
                continue
            if ci in used or cj in used:
                continue
            dist = float(np.hypot(*(pj - pi)))
            a = list(chains[ci]) if ei == 1 else list(chains[ci])[::-1]
            b = list(chains[cj]) if ej == 0 else list(chains[cj])[::-1]
            bridge = _hermite_bridge(pi, ti, pj, tj, w, dist)
            _join_chains(chains, chain_clusters, live, ci, ei, cj, ej,
                         a, b, bridge)
            used.add(ci)
            used.add(cj)
            joined_total += 1
        if not used:
            break
    return joined_total


def drop_stray_chains(chains, chain_clusters, live, ink, sw,
                      min_span_factor=3.0, dot_factor=1.5,
                      max_dev_deg=25.0, min_contrast=0.15):
    """Fase 2.4: descarta cadenas cortas aisladas (puntos sueltos).

    Solo toca cadenas con extensión < min_span_factor·sw:
    - < dot_factor·sw = punto suelto (sin dirección definida) → fuera;
    - si su eje principal se desvía > max_dev_deg del campo de
      orientación local (tensor de estructura de la tinta), o su
      contraste medio < min_contrast → fuera.
    Las cadenas largas (vetas reales) nunca se tocan.
    """
    if not live:
        return 0
    span_ang = None  # lazily: (gx2, gxy, gy2) suavizados
    h, w = ink.shape
    s = max(1.0, float(sw))

    def field_angle(x, y):
        nonlocal span_ang
        if span_ang is None:
            gx = cv2.Sobel(ink, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(ink, cv2.CV_32F, 0, 1, ksize=3)
            k = (0, 0)
            jxx = cv2.GaussianBlur(gx * gx, k, s)
            jxy = cv2.GaussianBlur(gx * gy, k, s)
            jyy = cv2.GaussianBlur(gy * gy, k, s)
            span_ang = (jxx, jxy, jyy)
        jxx, jxy, jyy = span_ang
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        a = 0.5 * math.atan2(2.0 * float(jxy[yi, xi]),
                             float(jxx[yi, xi]) - float(jyy[yi, xi]))
        # tangente = perpendicular al gradiente dominante
        return math.degrees(a) + 90.0

    dropped = 0
    for ci in range(len(chains)):
        if chains[ci] is None or len(chains[ci]) < 2:
            continue
        idxs = chains[ci]
        xs = np.array([i % w for i in idxs], dtype=np.float64)
        ys = np.array([i // w for i in idxs], dtype=np.float64)
        span = float(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        if span >= min_span_factor * s:
            continue
        c0, c1 = chain_clusters[ci]
        if any(cl is not None and len(live.get(cl, ())) >= 3
               for cl in (c0, c1)):
            # conectada a un cruce real: forma parte de la red
            continue
        drop = False
        if span < dot_factor * s:
            # punto suelto: sin dirección definida
            drop = True
        else:
            contrast = float(np.mean(_bilinear(ink, xs, ys)))
            if contrast < min_contrast:
                drop = True
            else:
                mx, my = float(xs.mean()), float(ys.mean())
                dx, dy = xs - mx, ys - my
                cxx = float(np.mean(dx * dx))
                cyy = float(np.mean(dy * dy))
                cxy = float(np.mean(dx * dy))
                chain_ang = math.degrees(0.5 * math.atan2(2.0 * cxy, cxx - cyy))
                dev = abs(chain_ang - field_angle(mx, my)) % 180.0
                dev = min(dev, 180.0 - dev)
                if dev > max_dev_deg:
                    drop = True
        if drop:
            _purge_chain(live, ci)
            chains[ci] = None
            chain_clusters[ci] = (None, None)
            dropped += 1
    return dropped


# --------------------------------------------------------------- refinamiento


def refine_chains(chain_px_lists, ink, w, sw, near_junction,
                  profile_factor=0.75, sample_step=0.5, k=2):
    """Refinamiento sub-píxel de cada cadena.

    Para cada píxel (no cercano a una unión) se calcula el centroide
    ponderado por tinta^2 del perfil perpendicular al trazo (ventana de
    +-profile_factor*grosor, muestreo bilinear). Devuelve por cadena:
    dict(coords (N,2) float, refined (N,) bool, closed bool).
    """
    results = []
    half = profile_factor * max(sw, 1.0)
    offs = np.arange(-half, half + sample_step * 0.5, sample_step)
    for chain in chain_px_lists:
        n = len(chain)
        coords = np.empty((n, 2), dtype=np.float64)
        coords[:, 0] = [i % w for i in chain]
        coords[:, 1] = [i // w for i in chain]
        refined = np.zeros(n, bool)
        if n >= 5:
            pad = np.vstack([
                np.repeat(coords[:1], k, axis=0), coords,
                np.repeat(coords[-1:], k, axis=0)
            ])
            tangents = pad[2 * k:2 * k + n] - pad[:n]
            norms = np.hypot(tangents[:, 0], tangents[:, 1])
            usable = (~near_junction[chain]) & (norms > 1e-9)
            if np.any(usable):
                idxs = np.flatnonzero(usable)
                normals = np.empty((len(idxs), 2))
                normals[:, 0] = -tangents[idxs, 1] / norms[idxs]
                normals[:, 1] = tangents[idxs, 0] / norms[idxs]
                pts = coords[idxs]
                xs = pts[:, 0:1] + np.outer(normals[:, 0], offs)
                ys = pts[:, 1:2] + np.outer(normals[:, 1], offs)
                vals = _bilinear(ink, xs, ys)
                wgt = vals * vals
                denom = wgt.sum(axis=1)
                good = denom > 1e-6
                shift = np.zeros(len(idxs))
                shift[good] = (wgt[good] * offs).sum(axis=1) / denom[good]
                coords[idxs, 0] += normals[:, 0] * shift
                coords[idxs, 1] += normals[:, 1] * shift
                refined[idxs] = True
        results.append({"coords": coords, "refined": refined, "closed": False})
    return results


def place_junctions(chain_data, junction_incidents, sw, window=15):
    """Intersección LS de las rectas TLS terminales; coloca los extremos de
    las cadenas incidentes en la proyección de la unión sobre su propia
    recta y endereza el tramo terminal contaminado por el cruce.

    El ajuste TLS usa una ventana que SALTA la zona del cruce (los perfiles
    de tinta se contaminan con el trazo cruzado): se toman los píxeles
    refinados entre 0.6·grosor y 0.6·grosor+window desde la unión.
    """
    skip = int(math.ceil(0.6 * max(sw, 1.0)))
    tail = int(math.ceil(1.2 * max(sw, 1.0)))
    for inc in junction_incidents:
        fits = []
        for (k, end) in inc:
            cd = chain_data[k]
            n = len(cd["coords"])
            if end == 0:
                idxs = np.arange(min(skip, n), min(skip + window, n))
            else:
                idxs = np.arange(max(0, n - skip - window), max(0, n - skip))
            idxs = idxs[cd["refined"][idxs]] if idxs.size else idxs
            if idxs.size < 5:
                idxs = np.flatnonzero(cd["refined"])
            if idxs.size < 5:
                continue
            fit = fit_line_tls(cd["coords"][idxs])
            if fit is None:
                continue
            fits.append((k, end, fit))
        if len(fits) < 2:
            continue
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for (_, _, fit) in fits:
            d = np.asarray(fit["dir"])
            p = np.asarray(fit["point"])
            M = np.eye(2) - np.outer(d, d)
            A += M
            b += M @ p
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        J = np.linalg.solve(A, b)
        for (k, end, fit) in fits:
            cd = chain_data[k]
            d = np.asarray(fit["dir"])
            p = np.asarray(fit["point"])
            n = len(cd["coords"])
            # extremo exacto: proyección de J sobre la recta de la cadena
            proj = p + d * float((J - p) @ d)
            cd["coords"][0 if end == 0 else n - 1] = proj
            # enderezar el tramo terminal contaminado por el cruce
            lo = 0 if end == 0 else n - tail
            hi = tail if end == 0 else n
            for i in range(max(0, lo), min(n, hi)):
                q = cd["coords"][i]
                cd["coords"][i] = p + d * float((q - p) @ d)


# --------------------------------------------------------------- primitivas


def _tls_segment(pts):
    """Segmento recto extremo-a-extremo ajustado por TLS (o cuerda cruda)."""
    fit = fit_line_tls(pts)
    if fit is None:
        return (float(pts[0][0]), float(pts[0][1]),
                float(pts[-1][0]), float(pts[-1][1]))
    p0 = np.asarray(fit["point"])
    d = np.asarray(fit["dir"])
    a = p0 + d * float((np.asarray(pts[0]) - p0) @ d)
    b = p0 + d * float((np.asarray(pts[-1]) - p0) @ d)
    if np.hypot(*(b - a)) < 1e-9:
        return None
    return (float(a[0]), float(a[1]), float(b[0]), float(b[1]))


def chains_to_segments(chain_data, sw, rdp_tol=0.5, min_len=None):
    """Cadena -> segmentos rectos. RDP absoluta; si la cadena es recta
    (desviación máxima <= tol) emite UN segmento TLS extremo a extremo.

    min_len aquí es solo un filtro de emisión (fragmentos >= 2px); el
    filtrado definitivo de trazos cortos lo hace el pipeline DESPUÉS de
    merge_collinear_segments para poder curar gaps de escalera primero.
    """
    if rdp_tol is None or rdp_tol <= 0:
        rdp_tol = 0.5
    if min_len is None:
        min_len = 2.0
    segs = []
    for cd in chain_data:
        pts = cd["coords"]
        n = len(pts)
        if n < 2:
            continue
        simplified = simplify_polyline_adaptive(pts, rdp_tol)
        emit = []
        if len(simplified) == 2:
            emit.append(_tls_segment(pts))
        else:
            fit = fit_line_tls(pts)
            # Rectitud robusta: el raster propio (cuantización/aliasing)
            # mete ruido ~+-1px aunque el trazo sea recto; un TLS de cadena
            # completa lo promedia y da el ángulo exacto. Las curvas reales
            # se desvían mucho más (sagitta = L²/8R) y quedan como cuerdas.
            straight_tol = max(1.25, 0.13 * max(sw, 1.0), rdp_tol)
            if fit is not None and max_perp_deviation(pts, fit) <= straight_tol:
                emit.append(_tls_segment(pts))
            else:
                for i in range(len(simplified) - 1):
                    emit.append(_tls_segment(simplified[i:i + 2]))
        if cd.get("closed") and len(simplified) >= 3:
            emit.append(_tls_segment(np.vstack([simplified[-1], simplified[0]])))
        for s in emit:
            if s is None:
                continue
            if math_hypot_seg(s) >= min_len:
                segs.append(s)
    return segs


def math_hypot_seg(s):
    return math.hypot(s[2] - s[0], s[3] - s[1])


def merge_collinear_segments(segments, sw=1.0, angle_tol=1.0,
                             perp_tol=None, gap_tol=None):
    """Une segmentos colineales que se tocan por los extremos (cadenas
    partidas al cruzar uniones del esqueleto) en un único portador."""
    from collections import defaultdict

    n = len(segments)
    if n < 2:
        return [tuple(map(float, s)) for s in segments]
    if perp_tol is None:
        perp_tol = max(1.5, 0.35 * max(sw, 1.0))
    if gap_tol is None:
        gap_tol = max(2.5, 0.75 * max(sw, 1.0))

    lines = []
    for (x1, y1, x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy)
        if ln < 1e-9:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        ux, uy = dx / ln, dy / ln
        lines.append((float(x1), float(y1), float(x2), float(y2),
                      ang, ux, uy))
    n = len(lines)
    if n < 2:
        return [tuple(map(float, s)) for s in segments]

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def ang_diff(a, b):
        d = abs((a - b) % 180.0)
        return min(d, 180.0 - d)

    buckets = defaultdict(list)
    for i, L in enumerate(lines):
        buckets[int(L[4]) % 180].append(i)

    for bi in list(buckets.keys()):
        cand = []
        for off in (-1, 0, 1):
            cand.extend(buckets.get((bi + off) % 180, []))
        if len(cand) < 2:
            continue
        for ii in range(len(cand)):
            i = cand[ii]
            for jj in range(ii + 1, len(cand)):
                j = cand[jj]
                if find(i) == find(j):
                    continue
                Li, Lj = lines[i], lines[j]
                if ang_diff(Li[4], Lj[4]) > angle_tol:
                    continue
                pairs = ((Li[2], Li[3], Lj[0], Lj[1]),
                         (Li[0], Li[1], Lj[2], Lj[3]),
                         (Li[0], Li[1], Lj[0], Lj[1]),
                         (Li[2], Li[3], Lj[2], Lj[3]))
                hit = False
                for (ax, ay, bx, by) in pairs:
                    if math.hypot(bx - ax, by - ay) > gap_tol:
                        continue
                    # colinealidad mutua de extremos
                    t = (bx - Li[0]) * Li[5] + (by - Li[1]) * Li[6]
                    px, py = Li[0] + Li[5] * t, Li[1] + Li[6] * t
                    d1 = math.hypot(bx - px, by - py)
                    t2 = (ax - Lj[0]) * Lj[5] + (ay - Lj[1]) * Lj[6]
                    qx, qy = Lj[0] + Lj[5] * t2, Lj[1] + Lj[6] * t2
                    d2 = math.hypot(ax - qx, ay - qy)
                    if d1 <= perp_tol and d2 <= perp_tol:
                        hit = True
                        break
                if hit:
                    union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append((lines[g[0]][0], lines[g[0]][1],
                        lines[g[0]][2], lines[g[0]][3]))
            continue
        pts = []
        for i in g:
            pts.append((lines[i][0], lines[i][1]))
            pts.append((lines[i][2], lines[i][3]))
        fit = fit_line_tls(pts)
        if fit is None or max_perp_deviation(pts, fit) > 1.5 * perp_tol:
            for i in g:
                out.append((lines[i][0], lines[i][1],
                            lines[i][2], lines[i][3]))
            continue
        p0 = np.asarray(fit["point"])
        d = np.asarray(fit["dir"])
        ts = [(px - p0[0]) * d[0] + (py - p0[1]) * d[1] for (px, py) in pts]
        a = p0 + d * min(ts)
        b = p0 + d * max(ts)
        if math.hypot(*(b - a)) > 1e-9:
            out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
        else:
            for i in g:
                out.append((lines[i][0], lines[i][1],
                            lines[i][2], lines[i][3]))
    return out


# --------------------------------------------------------------- extractor


def extract_chains(skeleton, ink, stroke_width, organic=False):
    """F3a completo: esqueleto bool -> lista de dict(coords, refined, closed).

    Devuelve (chain_data, info) donde info lleva contadores de cirugía y el
    mapa near_junction para depuración/tests. organic=True (Fase 2.3/2.4):
    cierre de gaps global con Hermite y descarte de puntos sueltos.
    """
    sw = max(float(stroke_width), 1.0)
    flat, deg, node, junction = build_graph(skeleton)
    h, w = skeleton.shape
    if not np.any(node) and not np.any(flat & (deg == 2)):
        return [], {"chains": 0}
    chains, closed_flags = walk_chains(flat, deg, node, w)
    labels, _ = cluster_nodes(node, junction, w)

    chain_clusters = []
    live = {}
    for ci, chain in enumerate(chains):
        if closed_flags[ci]:
            chain_clusters.append((None, None))
            continue
        c0 = int(labels[chain[0]]) if labels[chain[0]] >= 0 else None
        c1 = int(labels[chain[-1]]) if labels[chain[-1]] >= 0 else None
        chain_clusters.append((c0, c1))
        if c0 is not None:
            live.setdefault(c0, []).append((ci, 0))
        if c1 is not None:
            if c1 == c0:
                live[c0].append((ci, 1))
            else:
                live.setdefault(c1, []).append((ci, 1))

    info = {"chains_raw": len(chains)}
    info["spurs_pruned"] = prune_spurs(chains, chain_clusters, live, sw)
    jc_m, jc_d = contract_junctions(chains, chain_clusters, live, sw)
    info["junctions_contracted"] = jc_m
    info["rhombus_chords_dropped"] = jc_d
    info["deg2_merged"] = merge_through_deg2(chains, chain_clusters, live)
    info["gaps_closed"] = close_gaps(chains, chain_clusters, live, ink, w, sw,
                                     organic=organic)
    info["strays_dropped"] = 0
    if organic:
        info["strays_dropped"] = drop_stray_chains(
            chains, chain_clusters, live, ink, sw)

    # estructuras activas finales
    active = []
    for ci, chain in enumerate(chains):
        if chain is None or len(chain) < 2:
            continue
        c0, c1 = chain_clusters[ci]
        j0 = c0 is not None and len(live.get(c0, ())) >= 3
        j1 = c1 is not None and len(live.get(c1, ())) >= 3
        active.append({
            "px": chain,
            "closed": closed_flags[ci],
            "junction_ends": (j0, j1),
            "end_clusters": (c0 if j0 else None, c1 if j1 else None),
        })
    info["chains_live"] = len(active)
    if not active:
        return [], info

    # mapa de proximidad a uniones (para saltar el refinamiento allí)
    near_junction = np.zeros(flat.shape[0], bool)
    junc_px = np.flatnonzero(node & junction)
    if junc_px.size:
        jm = np.zeros((h, w), np.uint8)
        jy, jx = divmod(junc_px, w)
        jm[jy, jx] = 1
        jm = cv2.dilate(jm, np.ones((5, 5), np.uint8), iterations=1)
        near_junction = jm.ravel().astype(bool)

    chain_data = refine_chains([a["px"] for a in active], ink, w, sw,
                               near_junction)
    for cd, a in zip(chain_data, active):
        cd["closed"] = a["closed"]

    # incidentes por cluster de unión (>= 3 cadenas vivas)
    inc_by_cluster = {}
    for k, a in enumerate(active):
        for end in (0, 1):
            cl = a["end_clusters"][end]
            if cl is not None:
                inc_by_cluster.setdefault(cl, []).append((k, end))
    place_junctions(chain_data, list(inc_by_cluster.values()), sw)

    return chain_data, info
