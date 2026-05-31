from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class SegmentSpec:
    """Resolved segment policy with all parameters bound."""

    type: str  # "full" | "topk_energy" | "uniform" | "topk_spectral_flux"
    window_s: float | None  # None for "full"
    k: int | None  # None for "full"; >=1 otherwise
    aggregation: str  # "mean" | "meanstd" | "max"

    def canonical(self) -> str:
        """Stable string for DuckDB segment_policy PK column.

        Examples:
            "full"
            "topk_energy:W=30.0,K=3|agg=mean"
            "uniform:W=30.0,K=3|agg=meanstd"
            "topk_spectral_flux:W=30.0,K=3|agg=max"
        """
        if self.type == "full":
            return "full"
        parts = [f"{self.type}:W={self.window_s},K={self.k}"]
        parts.append(f"agg={self.aggregation}")
        return "|".join(parts)

    @classmethod
    def parse(cls, canonical: str) -> "SegmentSpec":
        """Inverse of canonical()."""
        if canonical == "full":
            return cls(type="full", window_s=None, k=None, aggregation="mean")
        type_part, agg_part = canonical.split("|")
        agg = agg_part.removeprefix("agg=")
        type_name, params = type_part.split(":")
        param_dict = {}
        for param in params.split(","):
            key, val = param.split("=")
            param_dict[key] = val
        return cls(
            type=type_name,
            window_s=float(param_dict["W"]),
            k=int(param_dict["K"]),
            aggregation=agg,
        )


def get_segments(audio_path: Path, spec: SegmentSpec) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) intervals for the given policy.

    Edge cases:
    - duration < window_s: degrade to single full-track segment.
    - K * window_s > duration but window_s <= duration: return as many
      non-overlapping windows as fit, up to K (may be < K).
    - "full": always returns [(0, duration)].
    """
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr

    if spec.type == "full":
        return [(0.0, duration)]

    window_s = spec.window_s
    k = spec.k

    if duration < window_s:
        return [(0.0, duration)]

    if spec.type == "uniform":
        return _uniform_segments(duration, window_s, k)

    if spec.type == "topk_energy":
        return _topk_segments(y, sr, duration, window_s, k, _score_energy)

    if spec.type == "topk_spectral_flux":
        return _topk_segments(y, sr, duration, window_s, k, _score_spectral_flux)

    raise ValueError(f"Unknown segment type: {spec.type}")


def _uniform_segments(
    duration: float, window_s: float, k: int
) -> list[tuple[float, float]]:
    chunk_duration = duration / k
    segments = []
    for i in range(k):
        midpoint = (i + 0.5) * chunk_duration
        start = max(0.0, midpoint - window_s / 2)
        end = min(duration, midpoint + window_s / 2)
        segments.append((start, end))
    return segments


def _score_energy(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]


def _score_spectral_flux(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)


def _topk_segments(
    y: np.ndarray,
    sr: int,
    duration: float,
    window_s: float,
    k: int,
    score_fn,
) -> list[tuple[float, float]]:
    scores = score_fn(y, sr)
    hop_length = 512
    score_times = librosa.frames_to_time(
        np.arange(len(scores)), sr=sr, hop_length=hop_length
    )

    hop_s = window_s / 2
    candidates = []
    i = 0
    while True:
        start_s = i * hop_s
        if start_s + window_s > duration:
            break
        mask = (score_times >= start_s) & (score_times < start_s + window_s)
        if mask.any():
            mean_score = float(np.mean(scores[mask]))
            candidates.append((mean_score, start_s))
        i += 1

    return _greedy_topk(candidates, k, window_s)


def _greedy_topk(
    candidates: list[tuple[float, float]], k: int, window_s: float
) -> list[tuple[float, float]]:
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: list[tuple[float, float]] = []
    for _score, start_s in candidates:
        end_s = start_s + window_s
        if any(
            start_s < sel_end and end_s > sel_start for sel_start, sel_end in selected
        ):
            continue
        selected.append((start_s, end_s))
        if len(selected) == k:
            break
    selected.sort(key=lambda x: x[0])
    return selected
