import math

from pat_compiler import group_collinear, compile_pat
from pat_sim import parse_pat


def test_group_collinear_merges_same_line():
    segs = [(0.0, 0.0, 4.0, 0.0), (6.0, 0.0, 10.0, 0.0),
            (1.0, 1e-4, 9.0, -1e-4)]
    carriers = group_collinear(segs, angle_tol=0.5, perp_tol=0.05)
    horizontals = [c for c in carriers
                   if abs(c["angle"]) < 1.0 or abs(c["angle"] - 180) < 1.0]
    assert len(horizontals) == 1
    merged = sorted(horizontals[0]["intervals"])
    span = max(b for _, b in merged) - min(a for a, _ in merged)
    assert span >= 9.0


def test_group_keeps_parallel_lines_separate():
    segs = [(0.0, 0.0, 10.0, 0.0), (0.0, 2.0, 10.0, 2.0)]
    carriers = group_collinear(segs, angle_tol=0.5, perp_tol=0.05)
    assert len(carriers) == 2


def test_single_segment_entry_exact():
    segs = [(2.0, 3.0, 7.0, 3.0)]
    result = compile_pat(segs, name="EXACT", tile_w=10.0, tile_h=10.0,
                         min_dash=0.01, min_gap=0.01)
    _, entries = parse_pat(result["pat_content"])
    assert len(entries) == 1
    e = entries[0]
    assert abs(e["angle"]) < 1e-6
    assert abs(e["oy"] - 3.0) < 1e-6
    cycle = sum(abs(d) for d in e["dashes"])
    assert abs(cycle - 10.0) < 1e-3
    positive = [d for d in e["dashes"] if d > 0]
    assert any(abs(d - 5.0) < 1e-3 for d in positive)


def test_diagonal_continuous_no_phase_shift():
    segs = [(0.0, 0.0, 10.0, 10.0)]
    result = compile_pat(segs, name="DIAG", tile_w=10.0, tile_h=10.0)
    _, entries = parse_pat(result["pat_content"])
    assert len(entries) == 1
    e = entries[0]
    assert abs(e["angle"] - 45.0) < 0.01
    cross = e["dx"] * math.sin(math.radians(45)) \
        - e["dy"] * math.cos(math.radians(45))
    assert abs(cross) < 1e-6


def test_compile_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        compile_pat([])
