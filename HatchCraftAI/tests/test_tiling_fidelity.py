"""Fidelidad de tiling del compilador PAT per-line (Fase 1).

Bloquea el bug del "tablero de ajedrez": con el delta histórico (tile, tile)
o (0.7071·tile, 0.7071·tile) solo 0°/90°/45° repetían bien y el resto de
ángulos dejaban 8 de los 9 tiles vacíos (o con fantasmas).

Invariante: para CUALQUIER ángulo, la familia emitida cubre exactamente los
3×3 tiles con UNA copia por tile (9 en total), sin huecos ni fantasmas.
"""

import math

import pytest

from pat_compiler import compile_pat_perline, is_lattice_entry
from pat_sim import segments_in_rect

TILE = 100.0
SEG_LEN = 40.0
ANCHOR = (20.0, 20.0)


def _segment(deg, length=SEG_LEN, center=(TILE / 2.0, TILE / 2.0)):
    """Segmento de `length` centrado en el tile (queda dentro para todo ángulo)."""
    a = math.radians(deg)
    cx, cy = center
    x1 = cx - length * math.cos(a) / 2.0
    y1 = cy - length * math.sin(a) / 2.0
    return [(x1, y1, x1 + length * math.cos(a), y1 + length * math.sin(a))]


def _fully_inside(seg, x0, y0, tw, th, tol=1e-6):
    return (x0 - tol <= seg[0] <= x0 + tw + tol
            and x0 - tol <= seg[2] <= x0 + tw + tol
            and y0 - tol <= seg[1] <= y0 + th + tol
            and y0 - tol <= seg[3] <= y0 + th + tol)


def _local_sets(segs, tile_w, tile_h):
    """Segmentos de cada tile traducidos al sistema local del tile."""
    sets = []
    for j in range(3):
        for i in range(3):
            local = []
            for s in segs:
                if _fully_inside(s, i * tile_w, j * tile_h, tile_w, tile_h):
                    local.append((s[0] - i * tile_w, s[1] - j * tile_h,
                                  s[2] - i * tile_w, s[3] - j * tile_h))
            local.sort()
            sets.append(local)
    return sets


def _same_set(a, b, tol=1e-3):
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for sa in a:
        for i, sb in enumerate(b):
            if used[i]:
                continue
            if max(abs(x - y) for x, y in zip(sa, sb)) <= tol:
                used[i] = True
                break
        else:
            return False
    return True


def _assert_tiles_identical(sim, tile_w, tile_h, deg):
    """Invariante de fidelidad: los 3×3 tiles dibujan EXACTAMENTE lo mismo."""
    sets = _local_sets(sim, tile_w, tile_h)
    ref = sets[0]
    assert ref, f"{deg}°: el tile (0,0) queda vacío"
    for k, s in enumerate(sets):
        assert _same_set(ref, s), (
            f"{deg}°: tile {k % 3},{k // 3} con {len(s)} piezas "
            f"≠ tile 0,0 con {len(ref)} piezas")
    return len(ref)


@pytest.mark.parametrize("deg", [i * 0.5 for i in range(360)])
def test_every_angle_tiles_3x3(deg):
    r = compile_pat_perline(_segment(deg), name="T", desc="fidelity",
                            tile_size=TILE, coord_nd=6, dash_nd=6)
    sim = segments_in_rect(r["pat_content"], (0.0, 0.0, 3 * TILE, 3 * TILE))
    n = _assert_tiles_identical(sim, TILE, TILE, deg)
    if len(r["entries"]) == 1:
        # Sin división del segmento: UNA copia exacta por tile.
        assert n == 1, f"{deg}°: {n} copias por tile"
    for e in r["entries"]:
        assert is_lattice_entry(e, TILE, TILE), e


@pytest.mark.parametrize("deg", [0, 15, 33.3, 45, 77, 86, 100, 120, 150])
def test_non_square_tile_tiles_3x3(deg):
    tw, th = 100.0, 50.0
    r = compile_pat_perline(_segment(deg, center=(tw / 2.0, th / 2.0)),
                            name="T", desc="fidelity",
                            tile_size=tw, tile_w=tw, tile_h=th,
                            coord_nd=6, dash_nd=6)
    sim = segments_in_rect(r["pat_content"], (0.0, 0.0, 3 * tw, 3 * th))
    n = _assert_tiles_identical(sim, tw, th, deg)
    if len(r["entries"]) == 1:
        assert n == 1, f"{deg}°: {n} copias por tile"
    for e in r["entries"]:
        assert is_lattice_entry(e, tw, th), e


def test_orthogonal_entries_keep_legacy_delta():
    # 0°/90° exactos conservan la emisión histórica byte a byte.
    r = compile_pat_perline([(0.0, 0.0, 10.0, 0.0)], name="T", desc="d",
                            tile_size=10.0)
    e = r["entries"][0]
    assert e["dx"] == 10.0 and e["dy"] == 10.0
    assert "10.0,10.0" in r["pat_content"]


def test_long_segment_is_continuous():
    # Segmento más largo que |L|: la familia cubre el portador entero.
    r = compile_pat_perline(_segment(45.0, length=200.0), name="T",
                            desc="fidelity", tile_size=TILE)
    assert r["entries"][0]["dashes"] == []
    img_free = segments_in_rect(r["pat_content"], (0.0, 0.0, 3 * TILE, 3 * TILE))
    # La línea continua cruza los 3 tiles de cada diagonal.
    assert len(img_free) >= 3


def test_near_orthogonal_short_segment_stays_exact():
    # 0.3° sobre un segmento corto se cuantiza EXACTAMENTE a 0° (n=1,m=0).
    r = compile_pat_perline(_segment(0.3, length=9.0), name="T",
                            desc="fidelity", tile_size=TILE)
    e = r["entries"][0]
    assert e["angle"] == 0.0
    assert e["dx"] == TILE and e["dy"] == TILE


def test_chain_ids_close_vertex_gaps():
    # Una cadena de piezas debe anclar el inicio de cada una al final de la
    # anterior: sin huecos en los vértices compartidos.
    segs = []
    x, y = 10.0, 10.0
    for k in range(8):
        a = math.radians(33.0 + 0.2 * k)
        nx, ny = x + 8.0 * math.cos(a), y + 8.0 * math.sin(a)
        segs.append((x, y, nx, ny))
        x, y = nx, ny
    r = compile_pat_perline(segs, name="T", desc="chain", tile_size=TILE,
                            chain_ids=[7] * len(segs))
    entries = r["entries"]
    assert len(entries) == len(segs)
    for prev, nxt in zip(entries, entries[1:]):
        a = math.radians(prev["angle"])
        ex = prev["ox"] + prev["dashes"][0] * math.cos(a)
        ey = prev["oy"] + prev["dashes"][0] * math.sin(a)
        gap = math.hypot(nxt["ox"] - ex, nxt["oy"] - ey)
        assert gap <= 0.1, f"hueco en vértice: {gap}"


def test_render_faithful_reports_truncation():
    """Fase 1.4: el preview avisa cuando recorta por límite de segmentos."""
    from pat_sim import render_faithful
    r = compile_pat_perline(_segment(15.0), name="T", desc="fidelity",
                            tile_size=TILE, coord_nd=6, dash_nd=6)
    img, stats = render_faithful(r["pat_content"], TILE, TILE, tiles=3,
                                 size=300, return_stats=True, max_segments=2)
    assert stats["truncated"] is True
    assert stats["segments"] == 2


def test_render_faithful_no_truncation_under_cap():
    from pat_sim import render_faithful
    r = compile_pat_perline(_segment(15.0), name="T", desc="fidelity",
                            tile_size=TILE, coord_nd=6, dash_nd=6)
    img, stats = render_faithful(r["pat_content"], TILE, TILE, tiles=3,
                                 size=300, return_stats=True)
    assert stats["truncated"] is False
    assert stats["segments"] > 0


def test_render_pat_preview_returns_stats():
    from core_logic import render_pat_preview
    r = compile_pat_perline(_segment(45.0), name="T", desc="fidelity",
                            tile_size=TILE, coord_nd=6, dash_nd=6)
    img, stats = render_pat_preview(r["pat_content"], return_stats=True)
    assert stats["truncated"] is False
    assert stats["entries"] == 1
