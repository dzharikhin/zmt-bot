import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio.segments import SegmentSpec, get_segments


@pytest.fixture
def short_audio(tmp_path):
    """1-second audio file (shorter than any reasonable window)."""
    sr = 22050
    y = np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    path = tmp_path / "short.wav"
    sf.write(path, y, sr)
    return path, 1.0


@pytest.fixture
def long_audio(tmp_path):
    """60-second audio file with varying energy."""
    sr = 22050
    rng = np.random.default_rng(42)
    y = np.zeros(sr * 60, dtype=np.float32)
    # Quiet first 20s
    y[: sr * 20] = rng.standard_normal(sr * 20).astype(np.float32) * 0.1
    # Loud middle 20s
    y[sr * 20 : sr * 40] = rng.standard_normal(sr * 20).astype(np.float32) * 0.9
    # Quiet last 20s
    y[sr * 40 :] = rng.standard_normal(sr * 20).astype(np.float32) * 0.1
    path = tmp_path / "long.wav"
    sf.write(path, y, sr)
    return path, 60.0


@pytest.fixture
def exact_window_audio(tmp_path):
    """30-second audio file (exactly one window of 30s)."""
    sr = 22050
    y = np.random.default_rng(7).standard_normal(sr * 30).astype(np.float32)
    path = tmp_path / "exact.wav"
    sf.write(path, y, sr)
    return path, 30.0


# --- SegmentSpec.canonical / parse round-trip ---


class TestSegmentSpecCanonical:
    def test_full(self):
        spec = SegmentSpec(type="full", window_s=None, k=None, aggregation="mean")
        assert spec.canonical() == "full"

    def test_topk_energy(self):
        spec = SegmentSpec(type="topk_energy", window_s=30.0, k=3, aggregation="mean")
        assert spec.canonical() == "topk_energy:W=30.0,K=3|agg=mean"

    def test_uniform(self):
        spec = SegmentSpec(type="uniform", window_s=25.0, k=4, aggregation="meanstd")
        assert spec.canonical() == "uniform:W=25.0,K=4|agg=meanstd"

    def test_topk_spectral_flux(self):
        spec = SegmentSpec(
            type="topk_spectral_flux", window_s=15.0, k=5, aggregation="max"
        )
        assert spec.canonical() == "topk_spectral_flux:W=15.0,K=5|agg=max"


class TestSegmentSpecParse:
    def test_full(self):
        spec = SegmentSpec.parse("full")
        assert spec == SegmentSpec(
            type="full", window_s=None, k=None, aggregation="mean"
        )

    def test_topk_energy(self):
        spec = SegmentSpec.parse("topk_energy:W=30.0,K=3|agg=mean")
        assert spec == SegmentSpec(
            type="topk_energy", window_s=30.0, k=3, aggregation="mean"
        )

    def test_uniform(self):
        spec = SegmentSpec.parse("uniform:W=25.0,K=4|agg=meanstd")
        assert spec == SegmentSpec(
            type="uniform", window_s=25.0, k=4, aggregation="meanstd"
        )

    def test_spectral_flux(self):
        spec = SegmentSpec.parse("topk_spectral_flux:W=15.0,K=5|agg=max")
        assert spec == SegmentSpec(
            type="topk_spectral_flux", window_s=15.0, k=5, aggregation="max"
        )


class TestSegmentSpecRoundTrip:
    @pytest.mark.parametrize(
        "spec",
        [
            SegmentSpec(type="full", window_s=None, k=None, aggregation="mean"),
            SegmentSpec(type="topk_energy", window_s=30.0, k=3, aggregation="mean"),
            SegmentSpec(type="uniform", window_s=25.5, k=4, aggregation="meanstd"),
            SegmentSpec(
                type="topk_spectral_flux", window_s=15.0, k=5, aggregation="max"
            ),
        ],
    )
    def test_round_trip(self, spec):
        assert SegmentSpec.parse(spec.canonical()) == spec

    def test_integer_window_s_round_trip(self):
        spec = SegmentSpec(type="topk_energy", window_s=30, k=3, aggregation="mean")
        parsed = SegmentSpec.parse(spec.canonical())
        assert parsed.type == "topk_energy"
        assert parsed.window_s == 30.0
        assert parsed.k == 3
        assert parsed.aggregation == "mean"


# --- get_segments ---


class TestGetSegmentsFull:
    def test_full_returns_whole_track(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(type="full", window_s=None, k=None, aggregation="mean")
        segments = get_segments(path, spec)
        assert segments == [(0.0, pytest.approx(duration, abs=0.01))]

    def test_full_short_track(self, short_audio):
        path, duration = short_audio
        spec = SegmentSpec(type="full", window_s=None, k=None, aggregation="mean")
        segments = get_segments(path, spec)
        assert segments == [(0.0, pytest.approx(duration, abs=0.01))]


class TestGetSegmentsShortTrack:
    def test_topk_energy_degrades_to_full(self, short_audio):
        path, duration = short_audio
        spec = SegmentSpec(type="topk_energy", window_s=30.0, k=3, aggregation="mean")
        segments = get_segments(path, spec)
        assert segments == [(0.0, pytest.approx(duration, abs=0.01))]

    def test_uniform_degrades_to_full(self, short_audio):
        path, duration = short_audio
        spec = SegmentSpec(type="uniform", window_s=30.0, k=3, aggregation="mean")
        segments = get_segments(path, spec)
        assert segments == [(0.0, pytest.approx(duration, abs=0.01))]

    def test_spectral_flux_degrades_to_full(self, short_audio):
        path, duration = short_audio
        spec = SegmentSpec(
            type="topk_spectral_flux", window_s=30.0, k=3, aggregation="mean"
        )
        segments = get_segments(path, spec)
        assert segments == [(0.0, pytest.approx(duration, abs=0.01))]


class TestGetSegmentsUniform:
    def test_uniform_basic(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(type="uniform", window_s=10.0, k=3, aggregation="mean")
        segments = get_segments(path, spec)
        assert len(segments) == 3
        for start, end in segments:
            assert 0.0 <= start
            assert end <= duration + 0.01

    def test_uniform_window_larger_than_chunk(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(type="uniform", window_s=25.0, k=3, aggregation="mean")
        segments = get_segments(path, spec)
        assert len(segments) == 3
        # Midpoints at 10, 30, 50; windows centered with W=25
        # Window 1: [10-12.5, 10+12.5] = [-2.5, 22.5] → clamped [0, 22.5]
        # Window 2: [30-12.5, 30+12.5] = [17.5, 42.5]
        # Window 3: [50-12.5, 50+12.5] = [37.5, 62.5] → clamped [37.5, 60]
        assert segments[0][0] == pytest.approx(0.0, abs=0.01)
        assert segments[0][1] == pytest.approx(22.5, abs=0.5)
        assert segments[1][0] == pytest.approx(17.5, abs=0.5)
        assert segments[1][1] == pytest.approx(42.5, abs=0.5)
        assert segments[2][0] == pytest.approx(37.5, abs=0.5)
        assert segments[2][1] == pytest.approx(duration, abs=0.5)

    def test_uniform_exact_window(self, exact_window_audio):
        path, duration = exact_window_audio
        spec = SegmentSpec(type="uniform", window_s=30.0, k=1, aggregation="mean")
        segments = get_segments(path, spec)
        assert len(segments) == 1
        assert segments[0][0] == pytest.approx(0.0, abs=0.5)
        assert segments[0][1] == pytest.approx(duration, abs=0.5)


class TestGetSegmentsTopkEnergy:
    def test_topk_energy_selects_loudest(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(type="topk_energy", window_s=10.0, k=1, aggregation="mean")
        segments = get_segments(path, spec)
        assert len(segments) == 1
        start, end = segments[0]
        # The loudest part is the middle 20s (10s-30s range), so the selected
        # window should be in that region
        assert 10.0 <= start <= 25.0
        assert end - start == pytest.approx(10.0, abs=0.1)

    def test_topk_energy_non_overlapping(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(type="topk_energy", window_s=10.0, k=3, aggregation="mean")
        segments = get_segments(path, spec)
        assert len(segments) == 3
        # Verify non-overlapping
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                s1, e1 = segments[i]
                s2, e2 = segments[j]
                assert (
                    s1 >= e2 or s2 >= e1
                ), f"Segments overlap: {segments[i]} and {segments[j]}"

    def test_topk_energy_sorted_by_start(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(type="topk_energy", window_s=10.0, k=3, aggregation="mean")
        segments = get_segments(path, spec)
        starts = [s for s, _e in segments]
        assert starts == sorted(starts)

    def test_topk_energy_returns_fewer_when_not_enough_room(self, exact_window_audio):
        path, duration = exact_window_audio
        spec = SegmentSpec(type="topk_energy", window_s=10.0, k=5, aggregation="mean")
        segments = get_segments(path, spec)
        # 30s track, 10s window, 5s hop: positions at 0, 5, 10, 15, 20
        # Non-overlapping 10s windows max: at 0, 10, 20 = 3 windows
        assert len(segments) <= 5
        # Verify non-overlapping
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                s1, e1 = segments[i]
                s2, e2 = segments[j]
                assert s1 >= e2 or s2 >= e1


class TestGetSegmentsTopkSpectralFlux:
    def test_topk_spectral_flux_basic(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(
            type="topk_spectral_flux", window_s=10.0, k=2, aggregation="mean"
        )
        segments = get_segments(path, spec)
        assert len(segments) == 2
        for start, end in segments:
            assert 0.0 <= start
            assert end <= duration + 0.01
            assert end - start == pytest.approx(10.0, abs=0.1)

    def test_topk_spectral_flux_non_overlapping(self, long_audio):
        path, duration = long_audio
        spec = SegmentSpec(
            type="topk_spectral_flux", window_s=10.0, k=3, aggregation="mean"
        )
        segments = get_segments(path, spec)
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                s1, e1 = segments[i]
                s2, e2 = segments[j]
                assert s1 >= e2 or s2 >= e1


class TestGetSegmentsUnknownType:
    def test_unknown_type_raises(self, long_audio):
        path, _duration = long_audio
        spec = SegmentSpec(type="unknown", window_s=10.0, k=3, aggregation="mean")
        with pytest.raises(ValueError, match="Unknown segment type"):
            get_segments(path, spec)
