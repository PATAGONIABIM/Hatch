"""Tests golden con los .dxf reales del proyecto (Fase 0).

Bloquean regresiones del formato per-line aceptado por Revit y fijan la
línea base de solapado (Fase 3 debe reducirla, no aumentarla).
"""

import os

import pytest

from core_logic import DXFtoPatConverter
from metrics import count_overlap_entries
from pat_sim import parse_pat, render_faithful

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_DXF = [
    "ADOCRETO_MODEL.dxf",
    "HERRIBONE.dxf",
    "HILADA.dxf",
    "TABLEADO HORIZONTAL.dxf",
    "TEST.dxf",
]

# Línea base de solapado VISUAL (ángulo + offset ≡ mod tile + dashes que se
# solapan mod tile). Tras Fase 3 (dedup de familias) todos los goldens miden 0:
# los duplicados reales se eliminan y los patrones trabados legítimos (p.ej.
# las juntas escalonadas de TABLEADO) se conservan sin contar como solape.
# Subir por encima de 0 = regresión.
OVERLAP_BASELINE = {
    "ADOCRETO_MODEL.dxf": 0,
    "HERRIBONE.dxf": 0,
    "HILADA.dxf": 0,
    "TABLEADO HORIZONTAL.dxf": 0,
    "TEST.dxf": 0,
}


@pytest.mark.parametrize("fname", GOLDEN_DXF)
def test_golden_dxf_converts_clean(fname):
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    r = DXFtoPatConverter().convert(path, pattern_name="GOLDEN")
    assert "error" not in r
    assert r["segments"] > 0
    assert r["num_entries"] > 0


@pytest.mark.parametrize("fname", GOLDEN_DXF)
def test_golden_dxf_perline_format(fname):
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    r = DXFtoPatConverter().convert(path, pattern_name="GOLDEN")
    content = r["pat_content"]

    # Transporte: CRLF, sin BOM, termina en CRLF.
    assert not content.startswith("\ufeff")
    assert "\r\n" in content
    assert content.endswith("\r\n")
    assert content.replace("\r\n", "").find("\r") == -1

    # Header + tipo.
    assert content.startswith("*GOLDEN, Converted from AutoCAD DXF")
    assert ";%TYPE=MODEL" in content

    # Invariante per-line: una entrada por segmento VÁLIDO (los degenerados se omiten).
    name, entries = parse_pat(content)
    assert name == "GOLDEN"
    tile = max(r["tile_w"], r["tile_h"])
    assert len(entries) == r["num_entries"]
    assert len(entries) <= r["segments"]
    for e in entries:
        assert (abs(e["dx"] - tile) < 1e-4 or abs(e["dx"] - 0.7071067812 * tile) < 1e-4), f"dx inesperado: {e['dx']}"
        assert (abs(e["dy"] - tile) < 1e-4 or abs(e["dy"] - 0.7071067812 * tile) < 1e-4), f"dy inesperado: {e['dy']}"
        assert e["dashes"][0] > 0
        assert e["dashes"][-1] < 0

    # Render fiel no falla.
    img = render_faithful(content, tile, tile, tiles=3, size=300)
    assert img.shape == (300, 300, 3)


@pytest.mark.parametrize("fname", GOLDEN_DXF)
def test_golden_dxf_overlap_baseline(fname):
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    r = DXFtoPatConverter().convert(path, pattern_name="GOLDEN")
    _, entries = parse_pat(r["pat_content"])
    tile = max(r["tile_w"], r["tile_h"])
    overlap = count_overlap_entries(entries, tile)
    assert overlap <= OVERLAP_BASELINE[fname], (
        f"{fname}: solapado {overlap} > baseline {OVERLAP_BASELINE[fname]} (regresión)"
    )


def test_golden_dxf_drafting_type():
    path = os.path.join(ROOT, "HILADA.dxf")
    if not os.path.exists(path):
        pytest.skip("HILADA.dxf no existe")
    r = DXFtoPatConverter().convert(path, pattern_name="GOLDEN", pattern_type="DRAFTING")
    assert ";%TYPE=DRAFTING" in r["pat_content"]


def test_hilada_drops_degenerate_segment():
    """HILADA.dxf contiene 1 segmento de longitud 0; el emisor lo omite
    (un dash 0.0 sería inválido en Revit)."""
    path = os.path.join(ROOT, "HILADA.dxf")
    if not os.path.exists(path):
        pytest.skip("HILADA.dxf no existe")
    r = DXFtoPatConverter().convert(path, pattern_name="GOLDEN")
    assert r["segments"] == 5
    assert r["num_entries"] == 4
    _, entries = parse_pat(r["pat_content"])
    assert all(e["dashes"][0] > 0 for e in entries)
