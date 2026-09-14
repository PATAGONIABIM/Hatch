"""Tests unitarios de metrics.py (Fase 0)."""

import math

import ezdxf

from metrics import chord_error, count_overlap, count_overlap_entries
from core_logic import DXFtoPatConverter


def test_count_overlap_identical_segments():
    segs = [(0.0, 0.0, 10.0, 0.0), (0.0, 0.0, 10.0, 0.0)]
    assert count_overlap(segs) == 1


def test_count_overlap_parallel_distinct():
    segs = [(0.0, 0.0, 10.0, 0.0), (0.0, 5.0, 10.0, 5.0)]
    assert count_overlap(segs) == 0


def test_count_overlap_colinear_disjoint():
    segs = [(0.0, 0.0, 4.0, 0.0), (6.0, 0.0, 10.0, 0.0)]
    assert count_overlap(segs) == 0


def test_count_overlap_partial_overlap():
    segs = [(0.0, 0.0, 6.0, 0.0), (4.0, 0.0, 10.0, 0.0)]
    assert count_overlap(segs) == 1


def test_count_overlap_entries_mod_tile():
    entries = [
        {"angle": 0.0, "ox": 0.0, "oy": 0.0, "dx": 10.0, "dy": 10.0, "dashes": [1.0, -9.0]},
        {"angle": 0.0, "ox": 0.0, "oy": 10.0, "dx": 10.0, "dy": 10.0, "dashes": [1.0, -9.0]},
        {"angle": 90.0, "ox": 0.0, "oy": 0.0, "dx": 10.0, "dy": 10.0, "dashes": [1.0, -9.0]},
    ]
    # Las dos horizontales (offset 0 y 10 ≡ 0 mod tile) coinciden; la vertical no.
    assert count_overlap_entries(entries, tile=10.0) == 1


def test_chord_error_perfect():
    pts = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    assert chord_error(pts, pts) < 1e-9


def test_chord_error_arc_approximation():
    # Fuente: arco denso; simplificado: cuerda única. El error = apófisis (sagitta).
    r = 10.0
    n = 64
    src = [(r * math.cos(math.pi * i / n), r * math.sin(math.pi * i / n)) for i in range(n + 1)]
    simplified = [src[0], src[-1]]
    # Arco de 0..pi: src[0]=(r,0), src[-1]=(-r,0); la cuerda es el diámetro (eje x).
    # El punto más lejano a la cuerda es la cima (0,r): distancia = r.
    sagitta = r - math.hypot((src[0][0] + src[-1][0]) / 2, (src[0][1] + src[-1][1]) / 2)
    err = chord_error(src, simplified)
    assert abs(err - sagitta) < 1e-6


def test_dxf_arc_converts(tmp_path):
    """Un ARC se aplana y convierte sin error (base para fidelidad orgánica)."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_arc(center=(5, 5), radius=5, start_angle=0, end_angle=180)
    p = tmp_path / "arc.dxf"
    doc.saveas(str(p))
    r = DXFtoPatConverter().convert(str(p), pattern_name="ARC")
    assert "error" not in r
    assert r["num_entries"] > 1  # el arco se aplana en varios segmentos
