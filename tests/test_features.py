import numpy as np
import pytest
import soundfile as sf

from audio import features as audio_features
from audio.features import (
    _essentia_pool_to_vector,
    _summarize_matrix_rowstats,
    _summarize_stats4,
    assert_schema_dim_consistent,
    extract_essentia_features_segment,
    schema_fingerprint,
)

try:
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


def make_mock_pool(scalars=None, vectors_1d=None, vectors_2d=None, strings=None):
    if scalars is None:
        scalars = {}
    if vectors_1d is None:
        vectors_1d = {}
    if vectors_2d is None:
        vectors_2d = {}
    if strings is None:
        strings = {}

    class MockPool:
        def __init__(self):
            self._data = {}

            for name, val in scalars.items():
                self._data[name] = np.asarray(val)

            for name, val in vectors_1d.items():
                self._data[name] = np.array(val, dtype=np.float32)

            for name, val in vectors_2d.items():
                self._data[name] = np.array(val, dtype=np.float32)

            for name, val in strings.items():
                self._data[name] = val

            for i in range(5):
                self._data[f"metadata.test_{i}"] = f"meta_{i}"

        def descriptorNames(self):
            return list(self._data.keys())

        def __getitem__(self, key):
            return self._data[key]

    return MockPool()


class TestPoolToVector:
    def test_pads_short_arrays(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("rhythm.beats", 8, None),)
        )
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4]})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 8
        assert result[0] == pytest.approx(0.1)
        assert result[3] == pytest.approx(0.4)
        assert result[4] == pytest.approx(0.0)

    def test_truncates_long_arrays(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("rhythm.beats", 3, None),)
        )
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4, 0.5]})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 3

    def test_zero_fills_missing_descriptors(self, monkeypatch):
        monkeypatch.setattr(
            audio_features,
            "_DESCRIPTOR_SCHEMA",
            (("rhythm.beats", 4, None), ("lowlevel.freq", 2, None)),
        )
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4]})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 6
        assert result[4] == pytest.approx(0.0)
        assert result[5] == pytest.approx(0.0)

    def test_total_dim_matches_schema(self, monkeypatch):
        schema = (
            ("lowlevel.a", 3, None),
            ("lowlevel.b", 5, None),
            ("rhythm.c", 1, None),
        )
        monkeypatch.setattr(audio_features, "_DESCRIPTOR_SCHEMA", schema)
        pool = make_mock_pool(
            vectors_1d={"lowlevel.a": [1, 2, 3], "lowlevel.b": [1] * 5},
            scalars={"rhythm.c": 0.7},
        )
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 9

    def test_empty_schema_returns_zero_length(self, monkeypatch):
        monkeypatch.setattr(audio_features, "_DESCRIPTOR_SCHEMA", ())
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2]})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 0


class TestSummarizeStats4:
    def test_typical(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = _summarize_stats4(arr)
        assert out.shape == (4,)
        assert out[0] == pytest.approx(2.5)
        assert out[1] == pytest.approx(arr.std())
        assert out[2] == pytest.approx(1.0)
        assert out[3] == pytest.approx(4.0)

    def test_single_element(self):
        out = _summarize_stats4(np.array([7.0]))
        assert out.shape == (4,)
        assert out[0] == pytest.approx(7.0)
        assert out[1] == pytest.approx(0.0)

    def test_empty_returns_zeros(self):
        out = _summarize_stats4(np.array([]))
        assert out.shape == (4,)
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_constant_array_std_zero(self):
        out = _summarize_stats4(np.ones(10))
        assert out[1] == pytest.approx(0.0)


class TestSummarizeMatrixRowstats:
    def test_2d_shape(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        out = _summarize_matrix_rowstats(arr)
        assert out.shape == (12,)

    def test_values_correct(self):
        arr = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        out = _summarize_matrix_rowstats(arr)
        assert out[0] == pytest.approx(2.0)
        assert out[1] == pytest.approx(6.0)
        assert out[4] == pytest.approx(1.0)
        assert out[5] == pytest.approx(5.0)
        assert out[6] == pytest.approx(3.0)
        assert out[7] == pytest.approx(7.0)

    def test_empty_returns_empty(self):
        out = _summarize_matrix_rowstats(np.zeros((0, 3)))
        assert out.shape == (0,)

    def test_1d_treated_as_single_row(self):
        arr = np.array([1.0, 3.0], dtype=np.float32)
        out = _summarize_matrix_rowstats(arr)
        assert out.shape == (4,)
        assert out[0] == pytest.approx(2.0)
        assert out[2] == pytest.approx(1.0)
        assert out[3] == pytest.approx(3.0)


class TestNormalizerDispatch:
    def test_stats4_normalizer_via_schema(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("rhythm.variable", 4, "stats4"),)
        )
        pool = make_mock_pool(vectors_1d={"rhythm.variable": list(range(50))})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 4

    def test_different_lengths_same_output_length(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("rhythm.variable", 4, "stats4"),)
        )
        pool_short = make_mock_pool(vectors_1d={"rhythm.variable": [1.0, 2.0]})
        pool_long = make_mock_pool(vectors_1d={"rhythm.variable": list(range(100))})
        r1 = _essentia_pool_to_vector(pool_short)
        r2 = _essentia_pool_to_vector(pool_long)
        assert len(r1) == len(r2) == 4

    def test_matrix_rowstats_normalizer_via_schema(self, monkeypatch):
        monkeypatch.setattr(
            audio_features,
            "_DESCRIPTOR_SCHEMA",
            (("tonal.hpcp", 8, "matrix_rowstats"),),
        )
        pool = make_mock_pool(vectors_2d={"tonal.hpcp": [[1.0, 2.0], [3.0, 4.0]]})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 8

    def test_none_normalizer_passes_through(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("lowlevel.fixed", 3, None),)
        )
        pool = make_mock_pool(vectors_1d={"lowlevel.fixed": [1.0, 2.0, 3.0]})
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)

    def test_custom_normalizer_via_patch(self, monkeypatch):
        def double(arr):
            return arr.astype(np.float32).reshape(-1) * 2.0

        monkeypatch.setattr(
            audio_features,
            "_NORMALIZERS",
            {**audio_features._NORMALIZERS, "double": double},
        )
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("test.double", 3, "double"),)
        )
        pool = make_mock_pool(vectors_1d={"test.double": [1.0, 2.0, 3.0]})
        result = _essentia_pool_to_vector(pool)
        assert result[0] == pytest.approx(2.0)


class TestSchemaFingerprint:
    def test_stable_within_same_schema(self):
        f1 = schema_fingerprint()
        f2 = schema_fingerprint()
        assert f1 == f2

    def test_changes_when_schema_changes(self, monkeypatch):
        original = schema_fingerprint()
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("test.new_desc", 1, None),)
        )
        changed = schema_fingerprint()
        assert changed != original

    def test_changes_when_normalizer_key_changes(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("test.desc", 4, None),)
        )
        fp_none = schema_fingerprint()
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", (("test.desc", 4, "stats4"),)
        )
        fp_stats4 = schema_fingerprint()
        assert fp_none != fp_stats4

    def test_fingerprint_length(self):
        fp = schema_fingerprint()
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)


class TestExtractEssentiaFeaturesSegmentUnpacksTuple:
    def test_unpacks_tuple_from_extractor(self, monkeypatch):
        monkeypatch.setattr(
            audio_features,
            "_DESCRIPTOR_SCHEMA",
            (("lowlevel.val", 1, None), ("rhythm.beats", 3, None)),
        )

        mock_pool = make_mock_pool(
            scalars={"lowlevel.val": 0.5},
            vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3]},
        )

        def fake_extractor(_audio_path):
            return (mock_pool, "frames_placeholder")

        import pathlib

        fake_wav = pathlib.Path("/tmp/fake_cropped.wav")

        monkeypatch.setattr(
            audio_features,
            "_ffmpeg_crop_to_tempwav",
            lambda _path, _start, _end: fake_wav,
        )

        result = extract_essentia_features_segment(
            fake_extractor, pathlib.Path("/fake/audio.mp3"), 0.0, 10.0
        )

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.float32
        assert len(result) == 4


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not available")
class TestEssentiaExtraction:
    def test_pool_to_vector_handles_real_essentia_output(self, tmp_path):
        y = np.random.default_rng(1).standard_normal(44100 * 3).astype(np.float32)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, y, 44100)

        extractor = es.MusicExtractor()
        features, _frames = extractor(str(audio_path))

        result = _essentia_pool_to_vector(features)

        assert result.ndim == 1
        assert result.dtype == np.float32


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not available")
class TestAssertSchemaDimConsistent:
    def test_empty_schema_is_noop(self, monkeypatch):
        monkeypatch.setattr(audio_features, "_DESCRIPTOR_SCHEMA", ())
        assert_schema_dim_consistent(None)

    def test_correct_schema_passes(self, tmp_path, monkeypatch):
        val_wav = tmp_path / "dim_check.wav"
        audio_features._synthesize_wav(val_wav)
        extractor = es.MusicExtractor()
        features, _frames = extractor(str(val_wav))

        schema_entries = []
        for name in sorted(features.descriptorNames()):
            if name.startswith("metadata."):
                continue
            value = features[name]
            if isinstance(value, str):
                continue
            raw = np.asarray(value)
            arr = np.atleast_1d(raw)
            if arr.ndim >= 2:
                continue
            arr = arr.astype(np.float32).reshape(-1)
            schema_entries.append((name, len(arr), None))

        monkeypatch.setattr(audio_features, "_DESCRIPTOR_SCHEMA", tuple(schema_entries))
        assert_schema_dim_consistent(None)

    def test_wrong_length_raises_with_mismatch_info(self, monkeypatch):
        monkeypatch.setattr(
            audio_features,
            "_DESCRIPTOR_SCHEMA",
            (("lowlevel.average_loudness", 99, None),),
        )
        with pytest.raises(RuntimeError, match="lowlevel.average_loudness"):
            assert_schema_dim_consistent(None)
