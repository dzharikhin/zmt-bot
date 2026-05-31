import pytest

from core.paths import compute_file_hash, get_embed_version


class TestComputeFileHash:
    def test_returns_sha256_hex(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = compute_file_hash(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("deterministic content")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content a")
        f2.write_text("content b")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path / "nonexistent.txt")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        h = compute_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64


class TestGetEmbedVersion:
    def test_different_profile_different_version(self, tmp_path):
        p1 = tmp_path / "profile1.yaml"
        p2 = tmp_path / "profile2.yaml"
        p1.write_text("profile: v1")
        p2.write_text("profile: v2")
        weights = tmp_path / "weights.pth"
        weights.write_bytes(b"weights")
        v1 = get_embed_version(profile_path=p1, panns_weights_path=weights)
        v2 = get_embed_version(profile_path=p2, panns_weights_path=weights)
        assert v1 != v2

    def test_same_profile_same_version(self, tmp_path):
        p = tmp_path / "profile.yaml"
        p.write_text("profile: v1")
        weights = tmp_path / "weights.pth"
        weights.write_bytes(b"weights")
        v1 = get_embed_version(profile_path=p, panns_weights_path=weights)
        v2 = get_embed_version(profile_path=p, panns_weights_path=weights)
        assert v1 == v2

    def test_different_weights_different_version(self, tmp_path):
        profile = tmp_path / "profile.yaml"
        profile.write_text("profile: v1")
        w1 = tmp_path / "weights1.pth"
        w2 = tmp_path / "weights2.pth"
        w1.write_bytes(b"weights1")
        w2.write_bytes(b"weights2")
        v1 = get_embed_version(profile_path=profile, panns_weights_path=w1)
        v2 = get_embed_version(profile_path=profile, panns_weights_path=w2)
        assert v1 != v2
