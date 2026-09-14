"""Tests unitarios de segment_cleaner.dedup_duplicate_segments (Fase 3)."""

from segment_cleaner import dedup_duplicate_segments


def test_dedup_exact_duplicates():
    segs = [(0, 0, 10, 0), (0, 0, 10, 0)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0)
    assert len(kept) == 1
    assert removed == 1


def test_dedup_offset_mod_tile():
    # Dos líneas verticales a offset 0 y offset=tile → familias idénticas.
    segs = [(0, 0, 0, 10), (10, 0, 10, 10)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0)
    assert len(kept) == 1
    assert removed == 1


def test_keeps_distinct_parallel_lines():
    # Paralelas a distinto offset (no congruente mod tile) → conservar.
    segs = [(0, 0, 10, 0), (0, 3, 10, 3), (0, 7, 10, 7)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0)
    assert len(kept) == 3
    assert removed == 0


def test_keeps_same_offset_without_interval_overlap():
    # Misma línea de soporte (offset 0) pero tramos disjuntos mod tile → conservar.
    segs = [(0, 0, 2, 0), (5, 0, 7, 0)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0)
    assert len(kept) == 2
    assert removed == 0


def test_dedup_overlapping_intervals():
    # Mismo soporte, tramos solapados → duplicado parcial.
    segs = [(0, 0, 6, 0), (3, 0, 9, 0)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0)
    assert len(kept) == 1
    assert removed == 1


def test_angle_tolerance():
    # Ángulo fuera de tolerancia → no es duplicado.
    segs = [(0, 0, 10, 0), (0, 0, 10, 1)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0, angle_tol=0.5)
    assert len(kept) == 2
    assert removed == 0


def test_drops_degenerate():
    segs = [(1, 1, 1, 1), (0, 0, 10, 0)]
    kept, removed = dedup_duplicate_segments(segs, tile=10.0)
    assert kept == [(0, 0, 10, 0)]


def test_diagonal_mod_tile():
    # Diagonales 45°: offset perpendicular congruente mod tile → duplicado.
    segs = [(0, 0, 10, 10), (10, 0, 20, 10)]
    # offsets: línea y=x → 0; línea y=x-10 → offset 10/√2 ≡ 10/√2 mod (tile=10·√2?)
    # Con tile=10√2 la familia se repite cada tile; aquí solo verificamos
    # que offsets distintos NO se eliminan con tile grande.
    kept, removed = dedup_duplicate_segments(segs, tile=100.0)
    assert len(kept) == 2
    assert removed == 0
