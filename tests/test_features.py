import pathlib

import numpy as np
import pytest
import soundfile as sf

from audio import features as audio_features
from audio.features import (
    _discover_descriptor_names,
    _essentia_pool_to_vector,
    extract_essentia_features_segment,
)

try:
    import essentia.standard as es

    ESSUNETIA_AVAILABLE = True
except ImportError:
    ESSUNETIA_AVAILABLE = False


def make_mock_pool(scalars=None, vectors_1d=None, vectors_2d=None, strings=None):
    """Create a mock pool for testing _discover_descriptor_names."""
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


class TestDiscoverDescriptorNames:
    def test_handles_0d_scalars(self):
        pool = make_mock_pool(scalars={"lowlevel.scalar_val": 0.5})
        result = _discover_descriptor_names(pool)
        assert ("lowlevel.scalar_val", 1) in result

    def test_handles_1d_vectors(self):
        pool = make_mock_pool(vectors_1d={"lowlevel.vector_val": [1.0, 2.0, 3.0]})
        result = _discover_descriptor_names(pool)
        assert ("lowlevel.vector_val", 3) in result

    def test_skips_2d_arrays(self):
        pool = make_mock_pool(vectors_2d={"lowlevel.matrix": [[1.0, 2.0], [3.0, 4.0]]})
        result = _discover_descriptor_names(pool)
        assert ("lowlevel.matrix",) not in [(name, length) for name, length in result]

    def test_skips_long_1d_arrays(self):
        pool = make_mock_pool(vectors_1d={"lowlevel.long_vec": list(range(45))})
        result = _discover_descriptor_names(pool)
        assert ("lowlevel.long_vec",) not in [(name, length) for name, length in result]

    def test_skips_strings(self):
        pool = make_mock_pool(strings={"metadata.genre": "rock"})
        result = _discover_descriptor_names(pool)
        assert ("metadata.genre",) not in [(name, length) for name, length in result]

    def test_skips_metadata_prefix(self):
        pool = make_mock_pool(strings={"metadata.artist": "Queen"})
        result = _discover_descriptor_names(pool)
        assert ("metadata.artist",) not in [(name, length) for name, length in result]


class TestPoolToVector:
    def test_pads_short_arrays(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# test profile\nprofile: test\n")
        profile_key = audio_features.compute_file_hash(profile_path)
        audio_features._DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = [("rhythm.beats", 8)]
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4]})
        result = _essentia_pool_to_vector(pool, profile_path)
        assert len(result) == 8
        assert result[0] == 0.1
        assert result[3] == 0.4
        assert result[4] == 0.0
        assert result[7] == 0.0

    def test_truncates_long_arrays(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# test profile\nprofile: test\n")
        profile_key = audio_features.compute_file_hash(profile_path)
        audio_features._DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = [("rhythm.beats", 3)]
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4, 0.5]})
        result = _essentia_pool_to_vector(pool, profile_path)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_fills_missing_descriptors(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# test profile\nprofile: test\n")
        profile_key = audio_features.compute_file_hash(profile_path)
        audio_features._DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = [
            ("rhythm.beats", 4),
            ("lowlevel.freq", 2),
        ]
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4]})
        result = _essentia_pool_to_vector(pool, profile_path)
        assert len(result) == 6
        np.testing.assert_array_almost_equal(result[:4], [0.1, 0.2, 0.3, 0.4])
        assert result[4] == 0.0
        assert result[5] == 0.0

    def test_handles_scalar_descriptors(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# test profile\nprofile: test\n")
        profile_key = audio_features.compute_file_hash(profile_path)
        audio_features._DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = [("lowlevel.val", 1)]
        pool = make_mock_pool(scalars={"lowlevel.val": 0.5})
        result = _essentia_pool_to_vector(pool, profile_path)
        assert len(result) == 1
        assert result[0] == 0.5

    def test_scalar_discovered_vector_provided_truncates(self, tmp_path):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# test profile\nprofile: test\n")
        profile_key = audio_features.compute_file_hash(profile_path)
        audio_features._DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = [("rhythm.beats", 1)]
        pool = make_mock_pool(vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3, 0.4]})
        result = _essentia_pool_to_vector(pool, profile_path)
        assert len(result) == 1
        assert result[0] == 0.1


@pytest.mark.skipif(not ESSUNETIA_AVAILABLE, reason="Essentia not available")
class TestEssentiaExtraction:
    def test_discover_descriptor_names_handles_real_essentia_output(self, tmp_path):
        y = np.random.default_rng(0).standard_normal(44100 * 3).astype(np.float32)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, y, 44100)

        extractor = es.MusicExtractor()
        features, _frames = extractor(str(audio_path))

        result = _discover_descriptor_names(features)
        assert len(result) > 0

        for name, length in result:
            assert length >= 1
            assert isinstance(length, int)

    def test_pool_to_vector_handles_real_essentia_output(self, tmp_path):
        y = np.random.default_rng(1).standard_normal(44100 * 3).astype(np.float32)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, y, 44100)

        extractor = es.MusicExtractor()
        features, _frames = extractor(str(audio_path))

        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# dummy profile\n")

        result = _essentia_pool_to_vector(features, profile_path)

        assert result.ndim == 1
        assert result.dtype == np.float32
        assert len(result) > 0


class TestExtractEssentiaFeaturesSegmentUnpacksTuple:
    def test_unpacks_tuple_from_extractor(self, tmp_path, monkeypatch):
        profile_path = tmp_path / "profile.yaml"
        profile_path.write_text("# test profile\nprofile: test\n")
        profile_key = audio_features.compute_file_hash(profile_path)
        audio_features._DESCRIPTOR_NAMES_BY_PROFILE[profile_key] = [
            ("lowlevel.val", 1),
            ("rhythm.beats", 3),
        ]

        mock_pool = make_mock_pool(
            scalars={"lowlevel.val": 0.5},
            vectors_1d={"rhythm.beats": [0.1, 0.2, 0.3]},
        )

        def fake_extractor(_audio_path):
            return (mock_pool, "frames_placeholder")

        fake_wav = tmp_path / "cropped.wav"
        fake_wav.write_bytes(b"fake")

        monkeypatch.setattr(
            audio_features,
            "_ffmpeg_crop_to_tempwav",
            lambda _path, _start, _end: fake_wav,
        )

        result = extract_essentia_features_segment(
            fake_extractor, pathlib.Path("/fake/audio.mp3"), profile_path, 0.0, 10.0
        )

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.float32
        assert len(result) == 4
