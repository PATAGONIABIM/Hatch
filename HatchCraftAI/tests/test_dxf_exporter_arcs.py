"""Tests del exportador DXF con entidades ricas (Fase 5: chains, arcos, círculos)."""

import io
import math

import cv2
import ezdxf
import numpy as np

from dxf_exporter import chain_segments, image_bytes_to_dxf, write_dxf_entities


def _read_doc(dxf_bytes):
    return ezdxf.read(io.StringIO(dxf_bytes.decode("utf-8")))


def _entity_types(dxf_bytes):
    doc = _read_doc(dxf_bytes)
    return [e.dxftype() for e in doc.modelspace()]


def test_chain_segments_closes_square():
    segs = [
        (0.0, 0.0, 10.0, 0.0),
        (10.0, 0.0, 10.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
        (0.0, 10.0, 0.0, 0.0),
    ]
    chains = chain_segments(segs, tol=0.5)
    assert len(chains) == 1
    pts = chains[0]["points"]
    assert len(pts) == 5  # 4 esquinas + cierre
    assert math.dist(pts[0], pts[-1]) < 0.5
    assert sorted(chains[0]["indices"]) == [0, 1, 2, 3]


def test_chain_segments_keeps_isolated_line():
    segs = [(0.0, 0.0, 10.0, 0.0), (50.0, 50.0, 60.0, 50.0)]
    chains = chain_segments(segs, tol=0.5)
    assert len(chains) == 2
    assert all(len(c["points"]) == 2 for c in chains)


def test_write_dxf_entities_types():
    entities = [
        {"type": "line", "p1": (0.0, 0.0), "p2": (10.0, 0.0)},
        {"type": "polyline", "points": [(0.0, 5.0), (5.0, 5.0), (5.0, 8.0)]},
        {"type": "arc", "center": (20.0, 20.0), "radius": 5.0,
         "start_angle": 0.0, "end_angle": 90.0},
        {"type": "circle", "center": (40.0, 40.0), "radius": 3.0},
    ]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    data = write_dxf_entities(entities, colors=colors)
    types = _entity_types(data)
    assert types == ["LINE", "LWPOLYLINE", "ARC", "CIRCLE"]
    doc = _read_doc(data)
    ents = list(doc.modelspace())
    assert ents[0].rgb == (255, 0, 0)
    arc = ents[2]
    assert abs(arc.dxf.radius - 5.0) < 1e-9
    assert abs(arc.dxf.start_angle - 0.0) < 1e-9
    assert abs(arc.dxf.end_angle - 90.0) < 1e-9


def _make_circle_image(size=400, radius=140):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (0, 0, 0), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_image_to_dxf_contour_fits_arc_or_circle():
    data = _make_circle_image()
    res = image_bytes_to_dxf(data, method="contour", param1=40, param2=0.005,
                             tile_mm=100.0, max_resolution=400)
    assert "error" not in res, res.get("error")
    types = _entity_types(res["dxf_content"])
    assert "ARC" in types or "CIRCLE" in types, f"sin arcos: {types}"
    counts = res["entities"]
    assert counts["arcs"] + counts["circles"] >= 1


def test_image_to_dxf_hough_lines_only():
    size = 300
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    for x in range(0, size, 60):
        cv2.line(img, (x, 0), (x, size), (20, 20, 20), 2)
    ok, buf = cv2.imencode(".png", img)
    data = buf.tobytes()
    res = image_bytes_to_dxf(data, method="hough", max_resolution=300)
    assert "error" not in res, res.get("error")
    types = set(_entity_types(res["dxf_content"]))
    assert types == {"LINE"}
