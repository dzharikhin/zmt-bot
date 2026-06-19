from unittest.mock import patch

import numpy as np

from audio.extractor import CombinedExtractor


class MockPANNsModel:
    def extract(self, audio_path):
        return np.ones(2048, dtype=np.float32)

    def extract_segment(self, audio_path, start_s, end_s):
        return np.ones(2048, dtype=np.float32)


def make_extractor(essentia_dim=50, num_segments=1):
    def essentia_extract_fn(extractor, audio_path):
        return np.ones(essentia_dim, dtype=np.float32)

    def essentia_extract_segment_fn(extractor, audio_path, start, end):
        return np.ones(essentia_dim, dtype=np.float32)

    extractor = CombinedExtractor(
        essentia_extractor=None,
        panns_model=MockPANNsModel(),
        essentia_extract_fn=essentia_extract_fn,
        essentia_extract_segment_fn=essentia_extract_segment_fn,
    )

    def fake_get_segments(audio_path, spec):
        if spec.type == "full":
            return [(0.0, 10.0)]
        duration = 10.0
        if spec.type == "uniform":
            k = spec.k or 3
            window_s = spec.window_s or duration / k
            segments = []
            for i in range(k):
                start = i * window_s
                end = start + window_s
                if end > duration:
                    end = duration
                segments.append((start, end))
            return segments
        return [(0.0, duration)]

    return extractor, fake_get_segments


class TestCombinedExtractorFull:
    def test_single_segment_mean(self):
        extractor, fake_get_segments = make_extractor(essentia_dim=50)
        with patch("audio.extractor.get_segments", fake_get_segments):
            result = extractor("fake_path.mp3")

        assert result.shape == (50 + 2048,)
        np.testing.assert_array_equal(result[:50], np.ones(50))
        np.testing.assert_array_equal(result[50:], np.ones(2048))

    def test_single_segment_meanstd(self):
        extractor, fake_get_segments = make_extractor(essentia_dim=50)
        spec = type(
            "Spec",
            (),
            {"type": "full", "window_s": None, "k": None, "aggregation": "meanstd"},
        )()
        with patch("audio.extractor.get_segments", fake_get_segments):
            result = extractor("fake_path.mp3", segment_spec=spec)

        assert result.shape == (50 * 2 + 2048 * 2,)
        np.testing.assert_array_equal(result[:50], np.ones(50))
        np.testing.assert_array_equal(result[50:100], np.zeros(50))
        np.testing.assert_array_equal(result[100 : 100 + 2048], np.ones(2048))
        np.testing.assert_array_equal(result[100 + 2048 :], np.zeros(2048))

    def test_single_segment_max(self):
        extractor, fake_get_segments = make_extractor(essentia_dim=50)
        spec = type(
            "Spec",
            (),
            {"type": "full", "window_s": None, "k": None, "aggregation": "max"},
        )()
        with patch("audio.extractor.get_segments", fake_get_segments):
            result = extractor("fake_path.mp3", segment_spec=spec)

        assert result.shape == (50 + 2048,)
        np.testing.assert_array_equal(result[:50], np.ones(50))
        np.testing.assert_array_equal(result[50:], np.ones(2048))


class TestCombinedExtractorMultiSegment:
    def test_multiple_segments_mean(self):
        extractor, fake_get_segments = make_extractor(essentia_dim=30, num_segments=3)
        spec = type(
            "Spec",
            (),
            {"type": "uniform", "window_s": 3.0, "k": 3, "aggregation": "mean"},
        )()
        with patch("audio.extractor.get_segments", fake_get_segments):
            result = extractor("fake_path.mp3", segment_spec=spec)

        assert result.shape == (30 + 2048,)

    def test_multiple_segments_meanstd(self):
        extractor, fake_get_segments = make_extractor(essentia_dim=30, num_segments=3)
        spec = type(
            "Spec",
            (),
            {"type": "uniform", "window_s": 3.0, "k": 3, "aggregation": "meanstd"},
        )()
        with patch("audio.extractor.get_segments", fake_get_segments):
            result = extractor("fake_path.mp3", segment_spec=spec)

        assert result.shape == (30 * 2 + 2048 * 2,)

    def test_output_is_concatenation_not_aggregation(self):
        def essentia_extract_fn(extractor, audio_path):
            return np.ones(10, dtype=np.float32) * 2.0

        extractor, fake_get_segments = make_extractor(essentia_dim=10)
        extractor.essentia_extract_fn = essentia_extract_fn

        with patch("audio.extractor.get_segments", fake_get_segments):
            result = extractor("fake_path.mp3")

        assert result.shape == (10 + 2048,)
        np.testing.assert_array_equal(result[:10], np.ones(10) * 2.0)
        np.testing.assert_array_equal(result[10:], np.ones(2048))
