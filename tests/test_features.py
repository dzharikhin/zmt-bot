import logging

import numpy as np
import pytest
import soundfile as sf

from audio import features as audio_features
from audio.features import (
    _ALIAS_KEYS,
    _essentia_pool_to_vector,
    _summarize_key_cyclic,
    _summarize_matrix_rowstats,
    _summarize_scale_binary,
    _summarize_stats4,
    assert_schema_dim_consistent,
    extract_essentia_features,
    extract_essentia_features_segment,
    extract_frame_features,
    schema_fingerprint,
)

try:
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

KEY_SCALE_SCHEMA_ENTRIES = (
    ("tonal.chords_key", 2, "key_cyclic"),
    ("tonal.chords_scale", 1, "scale_binary"),
    ("tonal.key_edma.key", 2, "key_cyclic"),
    ("tonal.key_edma.scale", 1, "scale_binary"),
    ("tonal.key_krumhansl.key", 2, "key_cyclic"),
    ("tonal.key_krumhansl.scale", 1, "scale_binary"),
    ("tonal.key_temperley.key", 2, "key_cyclic"),
    ("tonal.key_temperley.scale", 1, "scale_binary"),
)

FRAME_SCHEMA_ENTRIES = (
    ("frames.pitch", 4, "stats4"),
    ("frames.pitch_instantaneous_confidence", 4, "stats4"),
    ("frames.inharmonicity", 4, "stats4"),
    ("frames.tristimulus", 4, "stats4"),
    ("frames.oddtoevenharmonicenergyratio", 4, "stats4"),
    ("frames.dissonance", 4, "stats4"),
)


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
        monkeypatch.setattr(
            audio_features,
            "get_essentia_extractor",
            lambda _: es.MusicExtractor(),
        )
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
            "get_essentia_extractor",
            lambda _: es.MusicExtractor(),
        )
        monkeypatch.setattr(
            audio_features,
            "_DESCRIPTOR_SCHEMA",
            (("lowlevel.average_loudness", 99, None),),
        )
        with pytest.raises(RuntimeError, match="lowlevel.average_loudness"):
            assert_schema_dim_consistent(None)


class TestKeyScaleMappings:
    def test_all_twelve_keys_exact_vectors(self):
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for pitch_class, key in enumerate(keys):
            out = _summarize_key_cyclic(key)
            angle = 2 * np.pi * pitch_class / 12
            np.testing.assert_allclose(out, [np.sin(angle), np.cos(angle)])
            assert out.dtype == np.float32

    def test_flat_aliases_match_sharp_equivalents(self):
        for flat, sharp in _ALIAS_KEYS.items():
            np.testing.assert_array_equal(
                _summarize_key_cyclic(flat), _summarize_key_cyclic(sharp)
            )

    def test_b_c_cyclic_adjacency(self):
        b = _summarize_key_cyclic("B")
        c = _summarize_key_cyclic("C")
        c_sharp = _summarize_key_cyclic("C#")
        np.testing.assert_allclose(np.linalg.norm(b - c), np.linalg.norm(c - c_sharp))

    def test_unknown_key_returns_neutral_without_raising(self, caplog):
        with caplog.at_level(logging.WARNING, logger="audio.features"):
            for bad in ("X", ""):
                out = _summarize_key_cyclic(bad)
                np.testing.assert_array_equal(out, np.zeros(2))
                assert out.dtype == np.float32
        assert any("Unknown key" in r.getMessage() for r in caplog.records)

    def test_ndarray_scalar_input_handled(self):
        np.testing.assert_array_equal(
            _summarize_key_cyclic(np.asarray("C")), _summarize_key_cyclic("C")
        )

    def test_scale_major_minor(self):
        np.testing.assert_array_equal(_summarize_scale_binary("major"), [1.0])
        np.testing.assert_array_equal(_summarize_scale_binary("minor"), [0.0])
        assert _summarize_scale_binary("minor").dtype == np.float32

    def test_unknown_scale_returns_neutral_without_raising(self, caplog):
        with caplog.at_level(logging.WARNING, logger="audio.features"):
            out = _summarize_scale_binary("dorian")
            np.testing.assert_array_equal(out, [0.5])
            assert out.dtype == np.float32
        assert any("Unknown scale" in r.getMessage() for r in caplog.records)


class TestKeyScaleSchemaEntries:
    def test_entries_in_real_schema(self):
        by_name = {
            name: (length, normalizer)
            for name, length, normalizer in audio_features._DESCRIPTOR_SCHEMA
        }
        for name, length, normalizer in KEY_SCALE_SCHEMA_ENTRIES:
            assert by_name.get(name) == (length, normalizer)

    def test_entries_append_after_beats_position(self):
        names = [name for name, _, _ in audio_features._DESCRIPTOR_SCHEMA]
        idx = names.index("rhythm.beats_position")
        assert names[idx + 1 : idx + 9] == [
            name for name, _, _ in KEY_SCALE_SCHEMA_ENTRIES
        ]
        assert names[idx + 9 :] == [name for name, _, _ in FRAME_SCHEMA_ENTRIES]

    def test_family_layout_covers_new_tonal_dims(self):
        layout = audio_features.descriptor_family_layout()
        schema_total = sum(length for _, length, _ in audio_features._DESCRIPTOR_SCHEMA)
        assert sum(end - start for _, start, end in layout) == schema_total
        assert layout[-2][0] == "tonal"
        assert layout[-1] == ("frames", 4380, schema_total)

    def test_end_to_end_pool_to_vector(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", KEY_SCALE_SCHEMA_ENTRIES
        )
        pool = make_mock_pool(
            strings={
                "tonal.chords_key": "C",
                "tonal.chords_scale": "major",
                "tonal.key_edma.key": "Db",
                "tonal.key_edma.scale": "minor",
                "tonal.key_krumhansl.key": "B",
                "tonal.key_krumhansl.scale": "major",
                "tonal.key_temperley.key": "X",
                "tonal.key_temperley.scale": "phrygian",
            }
        )
        result = _essentia_pool_to_vector(pool)
        assert len(result) == 12
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0:2], [0.0, 1.0])
        np.testing.assert_allclose(result[2:3], [1.0])
        np.testing.assert_allclose(result[3:5], _summarize_key_cyclic("C#"))
        np.testing.assert_allclose(result[5:6], [0.0])
        np.testing.assert_allclose(result[6:8], _summarize_key_cyclic("B"))
        np.testing.assert_allclose(result[8:9], [1.0])
        np.testing.assert_allclose(result[9:11], [0.0, 0.0])
        np.testing.assert_allclose(result[11:12], [0.5])


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not available")
class TestKeyScaleRealEssentia:
    def test_key_scale_descriptors_present_and_normalized(self, tmp_path):
        y = np.random.default_rng(1).standard_normal(44100 * 3).astype(np.float32)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, y, 44100)

        extractor = es.MusicExtractor()
        features, _frames = extractor(str(audio_path))

        pool_names = set(features.descriptorNames())
        for name, length, normalizer in KEY_SCALE_SCHEMA_ENTRIES:
            assert name in pool_names
            arr = audio_features._NORMALIZERS[normalizer](np.asarray(features[name]))
            assert len(arr) == length


class TestFrameValuesView:
    def test_overlay_adds_frame_names_and_prefers_them(self):
        pool = make_mock_pool(scalars={"lowlevel.average_loudness": 0.5})
        frame_values = {
            "frames.pitch": np.array([1.0, 2.0], dtype=np.float32),
        }
        view = audio_features._FrameValuesView(pool, frame_values)
        names = set(view.descriptorNames())
        assert "lowlevel.average_loudness" in names
        assert "frames.pitch" in names
        assert view["frames.pitch"].tolist() == [1.0, 2.0]
        assert view["lowlevel.average_loudness"] == pytest.approx(0.5)

    def test_missing_frame_name_not_in_descriptor_names(self):
        pool = make_mock_pool(scalars={"lowlevel.average_loudness": 0.5})
        view = audio_features._FrameValuesView(pool, {})
        assert "frames.pitch" not in set(view.descriptorNames())

    def test_pool_to_vector_zero_fills_absent_frame_entries(self, monkeypatch):
        monkeypatch.setattr(
            audio_features, "_DESCRIPTOR_SCHEMA", FRAME_SCHEMA_ENTRIES[:1]
        )
        pool = make_mock_pool(scalars={"lowlevel.average_loudness": 0.5})
        view = audio_features._FrameValuesView(pool, {})
        result = _essentia_pool_to_vector(view)
        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))


class TestFrameSchemaEntries:
    def test_entries_in_real_schema(self):
        tail = audio_features._DESCRIPTOR_SCHEMA[-len(FRAME_SCHEMA_ENTRIES) :]
        assert tail == FRAME_SCHEMA_ENTRIES
        assert sum(length for _, length, _ in audio_features._DESCRIPTOR_SCHEMA) == 4404

    def test_frame_names_match_chain_output_names(self):
        assert tuple(name for name, _, _ in FRAME_SCHEMA_ENTRIES) == (
            audio_features._FRAME_DESCRIPTOR_NAMES
        )

    def test_end_to_end_stats4_summaries_via_overlay(self, monkeypatch):
        monkeypatch.setattr(audio_features, "_DESCRIPTOR_SCHEMA", FRAME_SCHEMA_ENTRIES)
        frame_values = {
            "frames.pitch": np.array([440.0, 441.0, 439.0], dtype=np.float32),
            "frames.pitch_instantaneous_confidence": np.array(
                [0.9, 0.8], dtype=np.float32
            ),
            "frames.inharmonicity": np.array([0.1, 0.3], dtype=np.float32),
            "frames.tristimulus": np.array(
                [[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]], dtype=np.float32
            ),
            "frames.oddtoevenharmonicenergyratio": np.array(
                [1.0, 2.0], dtype=np.float32
            ),
            "frames.dissonance": np.array([0.05, 0.15], dtype=np.float32),
        }
        view = audio_features._FrameValuesView(make_mock_pool(), frame_values)
        result = _essentia_pool_to_vector(view)
        assert len(result) == 24
        assert result.dtype == np.float32
        np.testing.assert_allclose(
            result[0:4], _summarize_stats4(frame_values["frames.pitch"])
        )
        np.testing.assert_allclose(
            result[12:16], _summarize_stats4(frame_values["frames.tristimulus"])
        )


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not available")
class TestFrameChain:
    def _harmonic_tone(self, seconds=2.0):
        t = np.arange(int(44100 * seconds)) / 44100
        tone = np.zeros_like(t)
        for harmonic in range(1, 8):
            tone += (1.0 / harmonic) * np.sin(2 * np.pi * 440 * harmonic * t)
        return tone.astype(np.float32)

    def test_shapes_and_names(self):
        values = extract_frame_features(self._harmonic_tone())
        assert set(values) == set(audio_features._FRAME_DESCRIPTOR_NAMES)
        n_frames = len(values["frames.pitch"])
        assert n_frames > 80
        for name in (
            "frames.pitch_instantaneous_confidence",
            "frames.inharmonicity",
            "frames.oddtoevenharmonicenergyratio",
            "frames.dissonance",
        ):
            assert values[name].shape == (n_frames,)
        assert values["frames.tristimulus"].shape == (n_frames, 3)

    def test_pitch_of_harmonic_tone(self):
        values = extract_frame_features(self._harmonic_tone())
        assert np.mean(values["frames.pitch"]) == pytest.approx(440.0, abs=2.0)
        assert np.mean(values["frames.pitch_instantaneous_confidence"]) > 0.5

    def test_tristimulus_rows_sum_to_one(self):
        values = extract_frame_features(self._harmonic_tone())
        row_sums = values["frames.tristimulus"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_all_outputs_finite(self):
        for audio in (
            self._harmonic_tone(),
            np.zeros(44100, dtype=np.float32),
            np.random.default_rng(0).standard_normal(44100).astype(np.float32),
        ):
            values = extract_frame_features(audio)
            for name, arr in values.items():
                assert np.isfinite(arr).all(), name

    def test_deterministic(self):
        audio = self._harmonic_tone()
        first = extract_frame_features(audio)
        second = extract_frame_features(audio)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)

    def test_empty_audio_returns_empty_stacks(self):
        values = extract_frame_features(np.zeros(0, dtype=np.float32))
        assert set(values) == set(audio_features._FRAME_DESCRIPTOR_NAMES)
        for name, arr in values.items():
            assert arr.size == 0, name

    def test_short_audio_does_not_raise(self):
        values = extract_frame_features(
            np.ones(audio_features._FRAME_CHAIN_FRAME_SIZE - 1, dtype=np.float32)
        )
        for name, arr in values.items():
            assert np.isfinite(arr).all(), name

    def test_non_finite_input_sanitized(self):
        audio = self._harmonic_tone()
        audio[10] = np.nan
        values = extract_frame_features(audio)
        for name, arr in values.items():
            assert np.isfinite(arr).all(), name

    def test_dc_offset_does_not_poison_pitch(self):
        audio = self._harmonic_tone() + np.float32(0.5)
        values = extract_frame_features(audio)
        assert np.mean(values["frames.pitch"]) == pytest.approx(440.0, abs=2.0)
        for name, arr in values.items():
            assert np.isfinite(arr).all(), name


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not available")
class TestFrameChainEssentiaEndToEnd:
    def test_extract_essentia_features_full_width(self, tmp_path):
        y = np.random.default_rng(1).standard_normal(44100 * 3).astype(np.float32)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, y, 44100)

        extractor = es.MusicExtractor()
        result = extract_essentia_features(extractor, audio_path)

        assert result.shape == (4404,)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        frame_values = extract_frame_features(
            es.MonoLoader(filename=str(audio_path), sampleRate=44100)()
        )
        np.testing.assert_allclose(
            result[-24:],
            np.concatenate(
                [
                    _summarize_stats4(frame_values[name])
                    for name in audio_features._FRAME_DESCRIPTOR_NAMES
                ]
            ),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_extract_essentia_features_harmonic_tail(self, tmp_path):
        t = np.arange(int(44100 * 2)) / 44100
        tone = (
            0.3 * sum((1.0 / h) * np.sin(2 * np.pi * 440 * h * t) for h in range(1, 8))
        ).astype(np.float32)
        audio_path = tmp_path / "tone.wav"
        sf.write(audio_path, tone, 44100)

        extractor = es.MusicExtractor()
        result = extract_essentia_features(extractor, audio_path)

        assert result.shape == (4404,)
        assert result[-24] == pytest.approx(440.0, abs=2.0)
        frame_values = extract_frame_features(
            es.MonoLoader(filename=str(audio_path), sampleRate=44100)()
        )
        np.testing.assert_allclose(
            result[-24:],
            np.concatenate(
                [
                    _summarize_stats4(frame_values[name])
                    for name in audio_features._FRAME_DESCRIPTOR_NAMES
                ]
            ),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_pcm16_round_trip_preserves_pitch(self, tmp_path):
        t = np.arange(int(44100 * 2)) / 44100
        tone = (
            0.3 * sum((1.0 / h) * np.sin(2 * np.pi * 440 * h * t) for h in range(1, 8))
        ).astype(np.float32)
        audio_path = tmp_path / "tone_dc.wav"
        sf.write(audio_path, tone + np.float32(0.05), 44100)

        values = extract_frame_features(
            es.MonoLoader(filename=str(audio_path), sampleRate=44100)()
        )

        assert np.mean(values["frames.pitch"]) == pytest.approx(440.0, abs=2.0)
        assert np.sum(values["frames.pitch"] == 0) == 0
        for name, arr in values.items():
            assert np.isfinite(arr).all(), name


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not available")
class TestAssertSchemaDimConsistentFrames:
    def test_missing_frame_entry_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            audio_features,
            "get_essentia_extractor",
            lambda _: es.MusicExtractor(),
        )
        monkeypatch.setattr(
            audio_features,
            "_DESCRIPTOR_SCHEMA",
            (("frames.bogus", 4, "stats4"),),
        )
        with pytest.raises(RuntimeError, match="frames.bogus"):
            assert_schema_dim_consistent(None)

    def test_real_schema_passes(self):
        import pathlib

        profile = pathlib.Path(__file__).resolve().parent.parent / (
            "benchmark/essentia_profile.yaml"
        )
        assert_schema_dim_consistent(profile)
