import math

import cv2
import numpy as np
import pytest

from pat_compiler import compile_pat
from pat_sim import parse_pat, render_faithful, segments_in_rect, _clip_segment

EPS = 1e-6


def _point_on_segment_dist(px, py, seg):
    x1, y1, x2, y2 = seg
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < EPS:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _covered(point, segments, tol):
    return any(_point_on_segment_dist(point[0], point[1], s) < tol
               for s in segments)


def _sample_points(seg, n=7):
    x1, y1, x2, y2 = seg
    return [(x1 + (x2 - x1) * (0.08 + 0.84 * i / (n - 1)),
             y1 + (y2 - y1) * (0.08 + 0.84 * i / (n - 1))) for i in range(n)]


def _wrapped_expected(segments, w, h, rect):
    out = []
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            ox, oy = da * w, db * h
            for (x1, y1, x2, y2) in segments:
                clipped = _clip_segment((x1 + ox, y1 + oy),
                                        (x2 + ox, y2 + oy), rect)
                if clipped:
                    out.append((clipped[0][0], clipped[0][1],
                                clipped[1][0], clipped[1][1]))
    return out


def _assert_same_geometry(expected, decoded, tol=5e-3):
    def real(segs):
        return [s for s in segs
                if math.hypot(s[2] - s[0], s[3] - s[1]) > 1e-4]
    exp_pts = [p for s in real(expected) for p in _sample_points(s)]
    dec_pts = [p for s in real(decoded) for p in _sample_points(s)]
    for p in exp_pts:
        assert _covered(p, decoded, tol), f"punto esperado sin cobertura: {p}"
    for p in dec_pts:
        assert _covered(p, expected, tol), f"segmento fantasma en: {p}"


def _random_segments(seed, w, h, count=40):
    import random
    rng = random.Random(seed)
    segs = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.25:
            x1, x2 = rng.uniform(0, w), rng.uniform(0, w)
            y = rng.choice([rng.uniform(0, h), rng.uniform(-2, h + 2)])
            segs.append((x1, y, x2, y))
        elif kind < 0.5:
            y1, y2 = rng.uniform(0, h), rng.uniform(0, h)
            x = rng.choice([rng.uniform(0, w), rng.uniform(-2, w + 2)])
            segs.append((x, y1, x, y2))
        else:
            ang = rng.uniform(0, math.pi)
            r = rng.uniform(0.5, min(w, h))
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            segs.append((cx, cy,
                         cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return segs


@pytest.mark.parametrize("seed", [1, 7, 42, 123])
def test_roundtrip_no_loss(seed):
    w = h = 10.0
    segs = _random_segments(seed, w, h)
    result = compile_pat(segs, name="RT", tile_w=w, tile_h=h,
                         min_dash=0.05, min_gap=0.05,
                         angle_tol=0.3, perp_tol=0.03)
    rect = (0.0, 0.0, w, h)
    decoded = segments_in_rect(result["pat_content"], rect)
    expected = _wrapped_expected(segs, w, h, rect)
    exp_pts = [p for s in expected
               for p in _sample_points(s)
               if math.hypot(s[2] - s[0], s[3] - s[1]) > 1e-4]
    assert exp_pts
    for p in exp_pts:
        assert _covered(p, decoded, 5e-3), f"punto esperado sin cobertura: {p}"


def _tile_dark_counts(pat_content, tile_w, tile_h, tiles=5, size=500):
    img = render_faithful(pat_content, tile_w=tile_w, tile_h=tile_h,
                          tiles=tiles, size=size, show_grid=False)
    tp = size // tiles
    counts = []
    for r in range(tiles):
        for c in range(tiles):
            cell = img[r * tp:(r + 1) * tp, c * tp:(c + 1) * tp]
            counts.append(int((cell < 128).sum()))
    return counts


@pytest.mark.parametrize("seed", [1, 7, 42, 123])
def test_tiling_no_empty_quadrant(seed):
    w = h = 10.0
    segs = _random_segments(seed, w, h)
    result = compile_pat(segs, name="TILE", tile_w=w, tile_h=h,
                         min_dash=0.05, min_gap=0.05,
                         angle_tol=0.3, perp_tol=0.03)
    counts = _tile_dark_counts(result["pat_content"], w, h)
    avg = sum(counts) / len(counts)
    assert avg > 0
    for i, c in enumerate(counts):
        assert c >= 0.25 * avg, (
            f"tile {i} casi vacío: {c} px vs promedio {avg:.0f}")


@pytest.mark.parametrize("seed", [3, 99])
def test_delta_tile_periodic_and_matches_cycle(seed):
    w = h = 10.0
    segs = _random_segments(seed, w, h)
    result = compile_pat(segs, name="INV", tile_w=w, tile_h=h,
                         min_dash=0.05, min_gap=0.05)
    _, entries = parse_pat(result["pat_content"])
    assert entries
    for e in entries:
        rad = math.radians(e["angle"])
        nx, ny = -math.sin(rad), math.cos(rad)
        perp1 = abs(w * nx)
        perp2 = abs(h * ny)
        if perp1 >= perp2:
            assert e["dx"] == pytest.approx(w, abs=1e-6)
            assert e["dy"] == pytest.approx(0.0, abs=1e-6)
        else:
            assert e["dx"] == pytest.approx(0.0, abs=1e-6)
            assert e["dy"] == pytest.approx(h, abs=1e-6)
        assert 0 <= e["angle"] < 180


def test_border_crossing_wraps():
    w = h = 10.0
    segs = [(8.0, 5.0, 12.0, 5.0)]
    result = compile_pat(segs, name="WRAP", tile_w=w, tile_h=h,
                         min_dash=0.01, min_gap=0.01)
    rect = (0.0, 0.0, w, h)
    decoded = segments_in_rect(result["pat_content"], rect)
    left_piece = [s for s in decoded if s[0] < 0.1 and abs(s[1] - 5) < 0.1]
    right_piece = [s for s in decoded if s[2] > 9.9 and abs(s[1] - 5) < 0.1]
    assert left_piece and right_piece


def test_no_ghost_family_lines():
    w = h = 10.0
    segs = [(2.0, 3.0, 8.0, 9.0)]
    result = compile_pat(segs, name="GHOST", tile_w=w, tile_h=h)
    _, entries = parse_pat(result["pat_content"])
    rect = (0.0, 0.0, w, h)
    decoded = segments_in_rect(result["pat_content"], rect)
    assert len(decoded) >= 1
    for s in decoded:
        mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        dist = abs((mx - 2.0) * (9.0 - 3.0) - (my - 3.0) * (8.0 - 2.0)) \
            / math.hypot(6.0, 6.0)
        assert dist < 1e-3, f"línea fantasma detectada en {s}"
