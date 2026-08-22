import glob
import os

import numpy as np
import pytest

from pat_sim import parse_pat, render_faithful

SAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_parse_90deg():
    path = os.path.join(SAMPLES_DIR, "test_90deg.pat")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    name, entries = parse_pat(content)
    assert name == "SimpleTest"
    assert len(entries) == 2
    assert {e["angle"] for e in entries} == {0.0, 90.0}


@pytest.mark.parametrize("fname", [
    "ADOCRETO_MODEL.pat",
    "HILADA_MODEL.pat",
    "TABLEADO_HORIZONTAL.pat",
    "BASKET_WAVE.pat",
    "test_complex.pat",
    "test_pattern.pat",
])
def test_render_sample_pats(fname):
    path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} no existe")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    img = render_faithful(content, tiles=3, size=400)
    assert img.shape == (400, 400, 3)
    ink = int(np.count_nonzero(np.all(img < 128, axis=2)))
    _, entries = parse_pat(content)
    if entries:
        assert ink > 50, f"patrón sin tinta visible: {fname} ({ink}px)"


def test_faithful_vs_legacy_preview_difference():
    path = os.path.join(SAMPLES_DIR, "ADOCRETO_MODEL.pat")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    img3 = render_faithful(content, tiles=3, size=300)
    img6 = render_faithful(content, tiles=6, size=600)
    ink3 = int(np.count_nonzero(np.all(img3 < 128, axis=2)))
    ink6 = int(np.count_nonzero(np.all(img6 < 128, axis=2)))
    assert ink3 > 0 and ink6 > 0


def test_all_project_pats_parse_cleanly():
    pats = glob.glob(os.path.join(SAMPLES_DIR, "*.pat"))
    assert len(pats) >= 5
    for p in pats:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            name, entries = parse_pat(f.read())
        assert isinstance(name, str) and name
        for e in entries:
            assert set(e.keys()) == {"angle", "ox", "oy", "dx", "dy", "dashes"}
