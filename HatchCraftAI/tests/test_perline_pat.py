"""Tests del formato de emisión PAT "una entrada por línea" (el que Revit acepta).

Bloquea la regresión al formato v2 (portadoras con delta axis-aligned y multi-dash),
que Revit rechazaba y que producía el solapado de líneas en la simulación.
"""

import io

import numpy as np
import pytest

import pat_compiler
from core_logic import DXFtoPatConverter, ImageToPatConverter
from pat_sim import parse_pat, render_faithful


def _entries(pat_content):
    name, entries = parse_pat(pat_content)
    return name, entries


def test_compile_pat_perline_basic_format():
    r = pat_compiler.compile_pat_perline(
        [(0.0, 0.0, 10.0, 0.0)], name="T", desc="d", tile_size=10.0
    )
    pc = r["pat_content"]
    assert pc.startswith("*T, d\r\n;%TYPE=MODEL\r\n")
    assert pc.endswith("\r\n")
    # delta = (tile, tile); dash=length; gap=-0.001
    assert "10.0,10.0" in pc
    name, entries = _entries(pc)
    assert name == "T"
    assert len(entries) == 1
    e = entries[0]
    assert e["dx"] == 10.0 and e["dy"] == 10.0
    assert abs(e["dashes"][0] - 10.0) < 1e-6
    assert e["dashes"][1] < 0  # gap negativo


def test_compile_pat_perline_drafting_type():
    r = pat_compiler.compile_pat_perline(
        [(0.0, 0.0, 5.0, 0.0)], name="T", desc="d", tile_size=10.0,
        pattern_type="DRAFTING",
    )
    assert ";%TYPE=DRAFTING" in r["pat_content"]


def test_compile_pat_perline_quantizes_15deg():
    # 8 grados debe cuantizar a 15; 100 -> 105 -> (>=180? no) queda 105
    r = pat_compiler.compile_pat_perline(
        [(0.0, 0.0, 9.0, 1.26)], name="T", desc="d", tile_size=20.0,
        quantize_angles=pat_compiler._quantize_angle_15,
    )
    _, entries = _entries(r["pat_content"])
    assert entries[0]["angle"] in (0, 15)


def test_compile_pat_perline_swap_over_180():
    # Ángulo 200° -> se resta 180 -> 20°, y se intercambian puntos.
    r = pat_compiler.compile_pat_perline(
        [(0.0, 0.0, -9.0, -3.0)], name="T", desc="d", tile_size=20.0
    )
    _, entries = _entries(r["pat_content"])
    assert 0.0 <= entries[0]["angle"] < 180.0


def test_dxf_converter_emits_perline_delta(tmp_path):
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((0, 0), (0, 100))
    msp.add_line((100, 0), (100, 100))
    p = tmp_path / "rect.dxf"
    doc.saveas(str(p))

    res = DXFtoPatConverter().convert(str(p), pattern_name="TestRect")
    assert "error" not in res, res.get("error")
    pc = res["pat_content"]
    assert pc.startswith("*TestRect,")
    assert ";%TYPE=MODEL" in pc
    name, entries = _entries(pc)
    # Cada entrada: dx==dy==tile_size (delta=(tile,tile))
    tile = max(e["dx"] for e in entries)
    for e in entries:
        assert abs(e["dx"] - e["dy"]) < 1e-6, "delta no es (tile,tile)"
    # No solapado por delta axis-aligned: todas las entradas usan el mismo tile.
    assert all(abs(e["dx"] - tile) < 1e-6 for e in entries)
    # Render fiel no debe fallar
    img = render_faithful(pc, tile, tile, tiles=3, size=300)
    assert img.shape == (300, 300, 3)


def _make_grid_image(size=300, period=60):
    import cv2
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    for x in range(0, size, period):
        cv2.line(img, (x, 0), (x, size), (20, 20, 20), 2)
    for y in range(0, size, period):
        cv2.line(img, (0, y), (size, y), (20, 20, 20), 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_image_converter_emits_perline_delta():
    data = _make_grid_image()
    res = ImageToPatConverter().convert(image_bytes=data, method="hough", tile_mm=100.0)
    assert "error" not in res, res.get("error")
    pc = res["pat_content"]
    assert pc.startswith("*Image_Pattern,")
    assert ";%TYPE=MODEL" in pc
    assert pc.endswith("\r\n")
    _, entries = _entries(pc)
    assert entries, "sin entradas"
    for e in entries:
        assert abs(e["dx"] - e["dy"]) < 1e-4, "delta imagen no es (tile,tile)"
        assert e["dashes"][0] > 0
        assert e["dashes"][1] < 0


def test_diagonal_simulation_rotated_frame():
    from pat_sim import segments_in_rect, render_faithful
    # Entrada diagonal a 45° con delta=(100,100)
    pat = "*Diag,Diagonal 45 test\r\n;%TYPE=MODEL\r\n45.0, 0.0,0.0, 100.0,100.0, 141.421356,-0.001\r\n"
    img = render_faithful(pat, tile_w=100.0, tile_h=100.0, tiles=3, size=300)
    assert img.shape == (300, 300, 3)
    # Debe haber tinta visible en la simulación
    ink = int(np.count_nonzero(np.all(img < 128, axis=2)))
    assert ink > 100
    # En un rect de 300x300, con espaciado perpendicular de 100, debe haber múltiples diagonales paralelas
    segs = segments_in_rect(pat, (0.0, 0.0, 300.0, 300.0))
    assert len(segs) >= 3, f"esperadas múltiples diagonales paralelas, obtenidas {len(segs)}"
