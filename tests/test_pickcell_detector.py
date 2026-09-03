"""``select_segment`` decides which vision segment a detection locks onto:
largest by default, nearest-the-boresight under ``prefer_centred`` — the
mode the conductor's per-pick verify uses so a bigger same-band neighbor at
the frame edge never out-competes the centred target (GPU seed-7)."""

from dataclasses import dataclass, field

import pytest

from pickcell.detector import CENTRED_SEGMENT_MAX_OFFSET_MM, select_segment


@dataclass
class _Centre:
    x: float
    y: float
    z: float = 300.0


@dataclass
class _Geometry:
    center: _Centre


@dataclass
class _Geometries:
    geometries: list[_Geometry]


@dataclass
class _Segment:
    point_cloud: bytes
    geometries: _Geometries = field(
        default_factory=lambda: _Geometries([_Geometry(_Centre(0.0, 0.0))])
    )


def _segment(points: int, x_mm: float, y_mm: float) -> _Segment:
    return _Segment(b"p" * points, _Geometries([_Geometry(_Centre(x_mm, y_mm))]))


def test_default_selection_is_the_largest_cloud():
    big_far = _segment(5000, 280.0, -30.0)
    small_centred = _segment(1200, 8.0, 5.0)
    assert select_segment([small_centred, big_far], prefer_centred=False) is big_far


def test_centred_selection_beats_a_bigger_edge_neighbor():
    big_far = _segment(5000, 280.0, -30.0)
    small_centred = _segment(1200, 8.0, 5.0)
    assert select_segment([big_far, small_centred], prefer_centred=True) is small_centred


def test_centred_selection_admits_the_census_error_bound():
    # a target detected up to ~40 mm off the census pose must still qualify
    off_by_census_error = _segment(1000, 38.0, -20.0)
    assert select_segment([off_by_census_error], prefer_centred=True) is off_by_census_error


def test_centred_selection_refuses_a_neighbor_at_pool_spacing():
    """Both bounds: the 100 mm cap admits census error (<= ~40 mm) and rejects
    the pool's minimum 120 mm block spacing, so a missing target can never be
    silently substituted by its nearest neighbor."""
    neighbor_at_min_spacing = _segment(4000, 120.0, 0.0)
    with pytest.raises(RuntimeError, match="no segment within 100 mm of the look axis"):
        select_segment([neighbor_at_min_spacing], prefer_centred=True)
    assert CENTRED_SEGMENT_MAX_OFFSET_MM < 120.0
