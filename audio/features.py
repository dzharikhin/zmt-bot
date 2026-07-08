import hashlib
import logging
import pathlib
import subprocess
import tempfile
import wave
from typing import Callable

import librosa
import numpy as np
from panns_inference import AudioTagging

import config
import essentia
import essentia.standard as es
from audio.extractor import CombinedExtractor

essentia.EssentiaLogger().warningActive = False

logger = logging.getLogger(__name__)


def _summarize_stats4(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(4, dtype=np.float32)
    a = arr.astype(np.float32).reshape(-1)
    return np.array([a.mean(), a.std(), a.min(), a.max()], dtype=np.float32)


def _summarize_matrix_rowstats(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    m = np.atleast_2d(arr).astype(np.float32)
    return np.concatenate([m.mean(axis=1), m.std(axis=1), m.min(axis=1), m.max(axis=1)])


_NORMALIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "stats4": _summarize_stats4,
    "matrix_rowstats": _summarize_matrix_rowstats,
}

# Generated from audit of 10 tracks with audit/descriptor_shapes.py
# Set A (deterministic shape): 551 descriptors
# Set B (variable shape, normalized): 1 descriptors
_DESCRIPTOR_SCHEMA: tuple[tuple[str, int, str | None], ...] = (
    ("lowlevel.average_loudness", 1, None),
    ("lowlevel.barkbands.dmean", 27, None),
    ("lowlevel.barkbands.dmean2", 27, None),
    ("lowlevel.barkbands.dvar", 27, None),
    ("lowlevel.barkbands.dvar2", 27, None),
    ("lowlevel.barkbands.max", 27, None),
    ("lowlevel.barkbands.mean", 27, None),
    ("lowlevel.barkbands.median", 27, None),
    ("lowlevel.barkbands.min", 27, None),
    ("lowlevel.barkbands.stdev", 27, None),
    ("lowlevel.barkbands.var", 27, None),
    ("lowlevel.barkbands_crest.dmean", 1, None),
    ("lowlevel.barkbands_crest.dmean2", 1, None),
    ("lowlevel.barkbands_crest.dvar", 1, None),
    ("lowlevel.barkbands_crest.dvar2", 1, None),
    ("lowlevel.barkbands_crest.max", 1, None),
    ("lowlevel.barkbands_crest.mean", 1, None),
    ("lowlevel.barkbands_crest.median", 1, None),
    ("lowlevel.barkbands_crest.min", 1, None),
    ("lowlevel.barkbands_crest.stdev", 1, None),
    ("lowlevel.barkbands_crest.var", 1, None),
    ("lowlevel.barkbands_flatness_db.dmean", 1, None),
    ("lowlevel.barkbands_flatness_db.dmean2", 1, None),
    ("lowlevel.barkbands_flatness_db.dvar", 1, None),
    ("lowlevel.barkbands_flatness_db.dvar2", 1, None),
    ("lowlevel.barkbands_flatness_db.max", 1, None),
    ("lowlevel.barkbands_flatness_db.mean", 1, None),
    ("lowlevel.barkbands_flatness_db.median", 1, None),
    ("lowlevel.barkbands_flatness_db.min", 1, None),
    ("lowlevel.barkbands_flatness_db.stdev", 1, None),
    ("lowlevel.barkbands_flatness_db.var", 1, None),
    ("lowlevel.barkbands_kurtosis.dmean", 1, None),
    ("lowlevel.barkbands_kurtosis.dmean2", 1, None),
    ("lowlevel.barkbands_kurtosis.dvar", 1, None),
    ("lowlevel.barkbands_kurtosis.dvar2", 1, None),
    ("lowlevel.barkbands_kurtosis.max", 1, None),
    ("lowlevel.barkbands_kurtosis.mean", 1, None),
    ("lowlevel.barkbands_kurtosis.median", 1, None),
    ("lowlevel.barkbands_kurtosis.min", 1, None),
    ("lowlevel.barkbands_kurtosis.stdev", 1, None),
    ("lowlevel.barkbands_kurtosis.var", 1, None),
    ("lowlevel.barkbands_skewness.dmean", 1, None),
    ("lowlevel.barkbands_skewness.dmean2", 1, None),
    ("lowlevel.barkbands_skewness.dvar", 1, None),
    ("lowlevel.barkbands_skewness.dvar2", 1, None),
    ("lowlevel.barkbands_skewness.max", 1, None),
    ("lowlevel.barkbands_skewness.mean", 1, None),
    ("lowlevel.barkbands_skewness.median", 1, None),
    ("lowlevel.barkbands_skewness.min", 1, None),
    ("lowlevel.barkbands_skewness.stdev", 1, None),
    ("lowlevel.barkbands_skewness.var", 1, None),
    ("lowlevel.barkbands_spread.dmean", 1, None),
    ("lowlevel.barkbands_spread.dmean2", 1, None),
    ("lowlevel.barkbands_spread.dvar", 1, None),
    ("lowlevel.barkbands_spread.dvar2", 1, None),
    ("lowlevel.barkbands_spread.max", 1, None),
    ("lowlevel.barkbands_spread.mean", 1, None),
    ("lowlevel.barkbands_spread.median", 1, None),
    ("lowlevel.barkbands_spread.min", 1, None),
    ("lowlevel.barkbands_spread.stdev", 1, None),
    ("lowlevel.barkbands_spread.var", 1, None),
    ("lowlevel.dissonance.dmean", 1, None),
    ("lowlevel.dissonance.dmean2", 1, None),
    ("lowlevel.dissonance.dvar", 1, None),
    ("lowlevel.dissonance.dvar2", 1, None),
    ("lowlevel.dissonance.max", 1, None),
    ("lowlevel.dissonance.mean", 1, None),
    ("lowlevel.dissonance.median", 1, None),
    ("lowlevel.dissonance.min", 1, None),
    ("lowlevel.dissonance.stdev", 1, None),
    ("lowlevel.dissonance.var", 1, None),
    ("lowlevel.dynamic_complexity", 1, None),
    ("lowlevel.erbbands.dmean", 40, None),
    ("lowlevel.erbbands.dmean2", 40, None),
    ("lowlevel.erbbands.dvar", 40, None),
    ("lowlevel.erbbands.dvar2", 40, None),
    ("lowlevel.erbbands.max", 40, None),
    ("lowlevel.erbbands.mean", 40, None),
    ("lowlevel.erbbands.median", 40, None),
    ("lowlevel.erbbands.min", 40, None),
    ("lowlevel.erbbands.stdev", 40, None),
    ("lowlevel.erbbands.var", 40, None),
    ("lowlevel.erbbands_crest.dmean", 1, None),
    ("lowlevel.erbbands_crest.dmean2", 1, None),
    ("lowlevel.erbbands_crest.dvar", 1, None),
    ("lowlevel.erbbands_crest.dvar2", 1, None),
    ("lowlevel.erbbands_crest.max", 1, None),
    ("lowlevel.erbbands_crest.mean", 1, None),
    ("lowlevel.erbbands_crest.median", 1, None),
    ("lowlevel.erbbands_crest.min", 1, None),
    ("lowlevel.erbbands_crest.stdev", 1, None),
    ("lowlevel.erbbands_crest.var", 1, None),
    ("lowlevel.erbbands_flatness_db.dmean", 1, None),
    ("lowlevel.erbbands_flatness_db.dmean2", 1, None),
    ("lowlevel.erbbands_flatness_db.dvar", 1, None),
    ("lowlevel.erbbands_flatness_db.dvar2", 1, None),
    ("lowlevel.erbbands_flatness_db.max", 1, None),
    ("lowlevel.erbbands_flatness_db.mean", 1, None),
    ("lowlevel.erbbands_flatness_db.median", 1, None),
    ("lowlevel.erbbands_flatness_db.min", 1, None),
    ("lowlevel.erbbands_flatness_db.stdev", 1, None),
    ("lowlevel.erbbands_flatness_db.var", 1, None),
    ("lowlevel.erbbands_kurtosis.dmean", 1, None),
    ("lowlevel.erbbands_kurtosis.dmean2", 1, None),
    ("lowlevel.erbbands_kurtosis.dvar", 1, None),
    ("lowlevel.erbbands_kurtosis.dvar2", 1, None),
    ("lowlevel.erbbands_kurtosis.max", 1, None),
    ("lowlevel.erbbands_kurtosis.mean", 1, None),
    ("lowlevel.erbbands_kurtosis.median", 1, None),
    ("lowlevel.erbbands_kurtosis.min", 1, None),
    ("lowlevel.erbbands_kurtosis.stdev", 1, None),
    ("lowlevel.erbbands_kurtosis.var", 1, None),
    ("lowlevel.erbbands_skewness.dmean", 1, None),
    ("lowlevel.erbbands_skewness.dmean2", 1, None),
    ("lowlevel.erbbands_skewness.dvar", 1, None),
    ("lowlevel.erbbands_skewness.dvar2", 1, None),
    ("lowlevel.erbbands_skewness.max", 1, None),
    ("lowlevel.erbbands_skewness.mean", 1, None),
    ("lowlevel.erbbands_skewness.median", 1, None),
    ("lowlevel.erbbands_skewness.min", 1, None),
    ("lowlevel.erbbands_skewness.stdev", 1, None),
    ("lowlevel.erbbands_skewness.var", 1, None),
    ("lowlevel.erbbands_spread.dmean", 1, None),
    ("lowlevel.erbbands_spread.dmean2", 1, None),
    ("lowlevel.erbbands_spread.dvar", 1, None),
    ("lowlevel.erbbands_spread.dvar2", 1, None),
    ("lowlevel.erbbands_spread.max", 1, None),
    ("lowlevel.erbbands_spread.mean", 1, None),
    ("lowlevel.erbbands_spread.median", 1, None),
    ("lowlevel.erbbands_spread.min", 1, None),
    ("lowlevel.erbbands_spread.stdev", 1, None),
    ("lowlevel.erbbands_spread.var", 1, None),
    ("lowlevel.gfcc.cov", 169, None),
    ("lowlevel.gfcc.icov", 169, None),
    ("lowlevel.gfcc.mean", 13, None),
    ("lowlevel.hfc.dmean", 1, None),
    ("lowlevel.hfc.dmean2", 1, None),
    ("lowlevel.hfc.dvar", 1, None),
    ("lowlevel.hfc.dvar2", 1, None),
    ("lowlevel.hfc.max", 1, None),
    ("lowlevel.hfc.mean", 1, None),
    ("lowlevel.hfc.median", 1, None),
    ("lowlevel.hfc.min", 1, None),
    ("lowlevel.hfc.stdev", 1, None),
    ("lowlevel.hfc.var", 1, None),
    ("lowlevel.loudness_ebu128.integrated", 1, None),
    ("lowlevel.loudness_ebu128.loudness_range", 1, None),
    ("lowlevel.loudness_ebu128.momentary.dmean", 1, None),
    ("lowlevel.loudness_ebu128.momentary.dmean2", 1, None),
    ("lowlevel.loudness_ebu128.momentary.dvar", 1, None),
    ("lowlevel.loudness_ebu128.momentary.dvar2", 1, None),
    ("lowlevel.loudness_ebu128.momentary.max", 1, None),
    ("lowlevel.loudness_ebu128.momentary.mean", 1, None),
    ("lowlevel.loudness_ebu128.momentary.median", 1, None),
    ("lowlevel.loudness_ebu128.momentary.min", 1, None),
    ("lowlevel.loudness_ebu128.momentary.stdev", 1, None),
    ("lowlevel.loudness_ebu128.momentary.var", 1, None),
    ("lowlevel.loudness_ebu128.short_term.dmean", 1, None),
    ("lowlevel.loudness_ebu128.short_term.dmean2", 1, None),
    ("lowlevel.loudness_ebu128.short_term.dvar", 1, None),
    ("lowlevel.loudness_ebu128.short_term.dvar2", 1, None),
    ("lowlevel.loudness_ebu128.short_term.max", 1, None),
    ("lowlevel.loudness_ebu128.short_term.mean", 1, None),
    ("lowlevel.loudness_ebu128.short_term.median", 1, None),
    ("lowlevel.loudness_ebu128.short_term.min", 1, None),
    ("lowlevel.loudness_ebu128.short_term.stdev", 1, None),
    ("lowlevel.loudness_ebu128.short_term.var", 1, None),
    ("lowlevel.melbands.dmean", 40, None),
    ("lowlevel.melbands.dmean2", 40, None),
    ("lowlevel.melbands.dvar", 40, None),
    ("lowlevel.melbands.dvar2", 40, None),
    ("lowlevel.melbands.max", 40, None),
    ("lowlevel.melbands.mean", 40, None),
    ("lowlevel.melbands.median", 40, None),
    ("lowlevel.melbands.min", 40, None),
    ("lowlevel.melbands.stdev", 40, None),
    ("lowlevel.melbands.var", 40, None),
    ("lowlevel.melbands128.dmean", 128, None),
    ("lowlevel.melbands128.dmean2", 128, None),
    ("lowlevel.melbands128.dvar", 128, None),
    ("lowlevel.melbands128.dvar2", 128, None),
    ("lowlevel.melbands128.max", 128, None),
    ("lowlevel.melbands128.mean", 128, None),
    ("lowlevel.melbands128.median", 128, None),
    ("lowlevel.melbands128.min", 128, None),
    ("lowlevel.melbands128.stdev", 128, None),
    ("lowlevel.melbands128.var", 128, None),
    ("lowlevel.melbands_crest.dmean", 1, None),
    ("lowlevel.melbands_crest.dmean2", 1, None),
    ("lowlevel.melbands_crest.dvar", 1, None),
    ("lowlevel.melbands_crest.dvar2", 1, None),
    ("lowlevel.melbands_crest.max", 1, None),
    ("lowlevel.melbands_crest.mean", 1, None),
    ("lowlevel.melbands_crest.median", 1, None),
    ("lowlevel.melbands_crest.min", 1, None),
    ("lowlevel.melbands_crest.stdev", 1, None),
    ("lowlevel.melbands_crest.var", 1, None),
    ("lowlevel.melbands_flatness_db.dmean", 1, None),
    ("lowlevel.melbands_flatness_db.dmean2", 1, None),
    ("lowlevel.melbands_flatness_db.dvar", 1, None),
    ("lowlevel.melbands_flatness_db.dvar2", 1, None),
    ("lowlevel.melbands_flatness_db.max", 1, None),
    ("lowlevel.melbands_flatness_db.mean", 1, None),
    ("lowlevel.melbands_flatness_db.median", 1, None),
    ("lowlevel.melbands_flatness_db.min", 1, None),
    ("lowlevel.melbands_flatness_db.stdev", 1, None),
    ("lowlevel.melbands_flatness_db.var", 1, None),
    ("lowlevel.melbands_kurtosis.dmean", 1, None),
    ("lowlevel.melbands_kurtosis.dmean2", 1, None),
    ("lowlevel.melbands_kurtosis.dvar", 1, None),
    ("lowlevel.melbands_kurtosis.dvar2", 1, None),
    ("lowlevel.melbands_kurtosis.max", 1, None),
    ("lowlevel.melbands_kurtosis.mean", 1, None),
    ("lowlevel.melbands_kurtosis.median", 1, None),
    ("lowlevel.melbands_kurtosis.min", 1, None),
    ("lowlevel.melbands_kurtosis.stdev", 1, None),
    ("lowlevel.melbands_kurtosis.var", 1, None),
    ("lowlevel.melbands_skewness.dmean", 1, None),
    ("lowlevel.melbands_skewness.dmean2", 1, None),
    ("lowlevel.melbands_skewness.dvar", 1, None),
    ("lowlevel.melbands_skewness.dvar2", 1, None),
    ("lowlevel.melbands_skewness.max", 1, None),
    ("lowlevel.melbands_skewness.mean", 1, None),
    ("lowlevel.melbands_skewness.median", 1, None),
    ("lowlevel.melbands_skewness.min", 1, None),
    ("lowlevel.melbands_skewness.stdev", 1, None),
    ("lowlevel.melbands_skewness.var", 1, None),
    ("lowlevel.melbands_spread.dmean", 1, None),
    ("lowlevel.melbands_spread.dmean2", 1, None),
    ("lowlevel.melbands_spread.dvar", 1, None),
    ("lowlevel.melbands_spread.dvar2", 1, None),
    ("lowlevel.melbands_spread.max", 1, None),
    ("lowlevel.melbands_spread.mean", 1, None),
    ("lowlevel.melbands_spread.median", 1, None),
    ("lowlevel.melbands_spread.min", 1, None),
    ("lowlevel.melbands_spread.stdev", 1, None),
    ("lowlevel.melbands_spread.var", 1, None),
    ("lowlevel.mfcc.cov", 169, None),
    ("lowlevel.mfcc.icov", 169, None),
    ("lowlevel.mfcc.mean", 13, None),
    ("lowlevel.pitch_salience.dmean", 1, None),
    ("lowlevel.pitch_salience.dmean2", 1, None),
    ("lowlevel.pitch_salience.dvar", 1, None),
    ("lowlevel.pitch_salience.dvar2", 1, None),
    ("lowlevel.pitch_salience.max", 1, None),
    ("lowlevel.pitch_salience.mean", 1, None),
    ("lowlevel.pitch_salience.median", 1, None),
    ("lowlevel.pitch_salience.min", 1, None),
    ("lowlevel.pitch_salience.stdev", 1, None),
    ("lowlevel.pitch_salience.var", 1, None),
    ("lowlevel.silence_rate_20dB.dmean", 1, None),
    ("lowlevel.silence_rate_20dB.dmean2", 1, None),
    ("lowlevel.silence_rate_20dB.dvar", 1, None),
    ("lowlevel.silence_rate_20dB.dvar2", 1, None),
    ("lowlevel.silence_rate_20dB.max", 1, None),
    ("lowlevel.silence_rate_20dB.mean", 1, None),
    ("lowlevel.silence_rate_20dB.median", 1, None),
    ("lowlevel.silence_rate_20dB.min", 1, None),
    ("lowlevel.silence_rate_20dB.stdev", 1, None),
    ("lowlevel.silence_rate_20dB.var", 1, None),
    ("lowlevel.silence_rate_30dB.dmean", 1, None),
    ("lowlevel.silence_rate_30dB.dmean2", 1, None),
    ("lowlevel.silence_rate_30dB.dvar", 1, None),
    ("lowlevel.silence_rate_30dB.dvar2", 1, None),
    ("lowlevel.silence_rate_30dB.max", 1, None),
    ("lowlevel.silence_rate_30dB.mean", 1, None),
    ("lowlevel.silence_rate_30dB.median", 1, None),
    ("lowlevel.silence_rate_30dB.min", 1, None),
    ("lowlevel.silence_rate_30dB.stdev", 1, None),
    ("lowlevel.silence_rate_30dB.var", 1, None),
    ("lowlevel.silence_rate_60dB.dmean", 1, None),
    ("lowlevel.silence_rate_60dB.dmean2", 1, None),
    ("lowlevel.silence_rate_60dB.dvar", 1, None),
    ("lowlevel.silence_rate_60dB.dvar2", 1, None),
    ("lowlevel.silence_rate_60dB.max", 1, None),
    ("lowlevel.silence_rate_60dB.mean", 1, None),
    ("lowlevel.silence_rate_60dB.median", 1, None),
    ("lowlevel.silence_rate_60dB.min", 1, None),
    ("lowlevel.silence_rate_60dB.stdev", 1, None),
    ("lowlevel.silence_rate_60dB.var", 1, None),
    ("lowlevel.spectral_centroid.dmean", 1, None),
    ("lowlevel.spectral_centroid.dmean2", 1, None),
    ("lowlevel.spectral_centroid.dvar", 1, None),
    ("lowlevel.spectral_centroid.dvar2", 1, None),
    ("lowlevel.spectral_centroid.max", 1, None),
    ("lowlevel.spectral_centroid.mean", 1, None),
    ("lowlevel.spectral_centroid.median", 1, None),
    ("lowlevel.spectral_centroid.min", 1, None),
    ("lowlevel.spectral_centroid.stdev", 1, None),
    ("lowlevel.spectral_centroid.var", 1, None),
    ("lowlevel.spectral_complexity.dmean", 1, None),
    ("lowlevel.spectral_complexity.dmean2", 1, None),
    ("lowlevel.spectral_complexity.dvar", 1, None),
    ("lowlevel.spectral_complexity.dvar2", 1, None),
    ("lowlevel.spectral_complexity.max", 1, None),
    ("lowlevel.spectral_complexity.mean", 1, None),
    ("lowlevel.spectral_complexity.median", 1, None),
    ("lowlevel.spectral_complexity.min", 1, None),
    ("lowlevel.spectral_complexity.stdev", 1, None),
    ("lowlevel.spectral_complexity.var", 1, None),
    ("lowlevel.spectral_contrast_coeffs.dmean", 6, None),
    ("lowlevel.spectral_contrast_coeffs.dmean2", 6, None),
    ("lowlevel.spectral_contrast_coeffs.dvar", 6, None),
    ("lowlevel.spectral_contrast_coeffs.dvar2", 6, None),
    ("lowlevel.spectral_contrast_coeffs.max", 6, None),
    ("lowlevel.spectral_contrast_coeffs.mean", 6, None),
    ("lowlevel.spectral_contrast_coeffs.median", 6, None),
    ("lowlevel.spectral_contrast_coeffs.min", 6, None),
    ("lowlevel.spectral_contrast_coeffs.stdev", 6, None),
    ("lowlevel.spectral_contrast_coeffs.var", 6, None),
    ("lowlevel.spectral_contrast_valleys.dmean", 6, None),
    ("lowlevel.spectral_contrast_valleys.dmean2", 6, None),
    ("lowlevel.spectral_contrast_valleys.dvar", 6, None),
    ("lowlevel.spectral_contrast_valleys.dvar2", 6, None),
    ("lowlevel.spectral_contrast_valleys.max", 6, None),
    ("lowlevel.spectral_contrast_valleys.mean", 6, None),
    ("lowlevel.spectral_contrast_valleys.median", 6, None),
    ("lowlevel.spectral_contrast_valleys.min", 6, None),
    ("lowlevel.spectral_contrast_valleys.stdev", 6, None),
    ("lowlevel.spectral_contrast_valleys.var", 6, None),
    ("lowlevel.spectral_decrease.dmean", 1, None),
    ("lowlevel.spectral_decrease.dmean2", 1, None),
    ("lowlevel.spectral_decrease.dvar", 1, None),
    ("lowlevel.spectral_decrease.dvar2", 1, None),
    ("lowlevel.spectral_decrease.max", 1, None),
    ("lowlevel.spectral_decrease.mean", 1, None),
    ("lowlevel.spectral_decrease.median", 1, None),
    ("lowlevel.spectral_decrease.min", 1, None),
    ("lowlevel.spectral_decrease.stdev", 1, None),
    ("lowlevel.spectral_decrease.var", 1, None),
    ("lowlevel.spectral_energy.dmean", 1, None),
    ("lowlevel.spectral_energy.dmean2", 1, None),
    ("lowlevel.spectral_energy.dvar", 1, None),
    ("lowlevel.spectral_energy.dvar2", 1, None),
    ("lowlevel.spectral_energy.max", 1, None),
    ("lowlevel.spectral_energy.mean", 1, None),
    ("lowlevel.spectral_energy.median", 1, None),
    ("lowlevel.spectral_energy.min", 1, None),
    ("lowlevel.spectral_energy.stdev", 1, None),
    ("lowlevel.spectral_energy.var", 1, None),
    ("lowlevel.spectral_energyband_high.dmean", 1, None),
    ("lowlevel.spectral_energyband_high.dmean2", 1, None),
    ("lowlevel.spectral_energyband_high.dvar", 1, None),
    ("lowlevel.spectral_energyband_high.dvar2", 1, None),
    ("lowlevel.spectral_energyband_high.max", 1, None),
    ("lowlevel.spectral_energyband_high.mean", 1, None),
    ("lowlevel.spectral_energyband_high.median", 1, None),
    ("lowlevel.spectral_energyband_high.min", 1, None),
    ("lowlevel.spectral_energyband_high.stdev", 1, None),
    ("lowlevel.spectral_energyband_high.var", 1, None),
    ("lowlevel.spectral_energyband_low.dmean", 1, None),
    ("lowlevel.spectral_energyband_low.dmean2", 1, None),
    ("lowlevel.spectral_energyband_low.dvar", 1, None),
    ("lowlevel.spectral_energyband_low.dvar2", 1, None),
    ("lowlevel.spectral_energyband_low.max", 1, None),
    ("lowlevel.spectral_energyband_low.mean", 1, None),
    ("lowlevel.spectral_energyband_low.median", 1, None),
    ("lowlevel.spectral_energyband_low.min", 1, None),
    ("lowlevel.spectral_energyband_low.stdev", 1, None),
    ("lowlevel.spectral_energyband_low.var", 1, None),
    ("lowlevel.spectral_energyband_middle_high.dmean", 1, None),
    ("lowlevel.spectral_energyband_middle_high.dmean2", 1, None),
    ("lowlevel.spectral_energyband_middle_high.dvar", 1, None),
    ("lowlevel.spectral_energyband_middle_high.dvar2", 1, None),
    ("lowlevel.spectral_energyband_middle_high.max", 1, None),
    ("lowlevel.spectral_energyband_middle_high.mean", 1, None),
    ("lowlevel.spectral_energyband_middle_high.median", 1, None),
    ("lowlevel.spectral_energyband_middle_high.min", 1, None),
    ("lowlevel.spectral_energyband_middle_high.stdev", 1, None),
    ("lowlevel.spectral_energyband_middle_high.var", 1, None),
    ("lowlevel.spectral_energyband_middle_low.dmean", 1, None),
    ("lowlevel.spectral_energyband_middle_low.dmean2", 1, None),
    ("lowlevel.spectral_energyband_middle_low.dvar", 1, None),
    ("lowlevel.spectral_energyband_middle_low.dvar2", 1, None),
    ("lowlevel.spectral_energyband_middle_low.max", 1, None),
    ("lowlevel.spectral_energyband_middle_low.mean", 1, None),
    ("lowlevel.spectral_energyband_middle_low.median", 1, None),
    ("lowlevel.spectral_energyband_middle_low.min", 1, None),
    ("lowlevel.spectral_energyband_middle_low.stdev", 1, None),
    ("lowlevel.spectral_energyband_middle_low.var", 1, None),
    ("lowlevel.spectral_entropy.dmean", 1, None),
    ("lowlevel.spectral_entropy.dmean2", 1, None),
    ("lowlevel.spectral_entropy.dvar", 1, None),
    ("lowlevel.spectral_entropy.dvar2", 1, None),
    ("lowlevel.spectral_entropy.max", 1, None),
    ("lowlevel.spectral_entropy.mean", 1, None),
    ("lowlevel.spectral_entropy.median", 1, None),
    ("lowlevel.spectral_entropy.min", 1, None),
    ("lowlevel.spectral_entropy.stdev", 1, None),
    ("lowlevel.spectral_entropy.var", 1, None),
    ("lowlevel.spectral_flux.dmean", 1, None),
    ("lowlevel.spectral_flux.dmean2", 1, None),
    ("lowlevel.spectral_flux.dvar", 1, None),
    ("lowlevel.spectral_flux.dvar2", 1, None),
    ("lowlevel.spectral_flux.max", 1, None),
    ("lowlevel.spectral_flux.mean", 1, None),
    ("lowlevel.spectral_flux.median", 1, None),
    ("lowlevel.spectral_flux.min", 1, None),
    ("lowlevel.spectral_flux.stdev", 1, None),
    ("lowlevel.spectral_flux.var", 1, None),
    ("lowlevel.spectral_kurtosis.dmean", 1, None),
    ("lowlevel.spectral_kurtosis.dmean2", 1, None),
    ("lowlevel.spectral_kurtosis.dvar", 1, None),
    ("lowlevel.spectral_kurtosis.dvar2", 1, None),
    ("lowlevel.spectral_kurtosis.max", 1, None),
    ("lowlevel.spectral_kurtosis.mean", 1, None),
    ("lowlevel.spectral_kurtosis.median", 1, None),
    ("lowlevel.spectral_kurtosis.min", 1, None),
    ("lowlevel.spectral_kurtosis.stdev", 1, None),
    ("lowlevel.spectral_kurtosis.var", 1, None),
    ("lowlevel.spectral_rms.dmean", 1, None),
    ("lowlevel.spectral_rms.dmean2", 1, None),
    ("lowlevel.spectral_rms.dvar", 1, None),
    ("lowlevel.spectral_rms.dvar2", 1, None),
    ("lowlevel.spectral_rms.max", 1, None),
    ("lowlevel.spectral_rms.mean", 1, None),
    ("lowlevel.spectral_rms.median", 1, None),
    ("lowlevel.spectral_rms.min", 1, None),
    ("lowlevel.spectral_rms.stdev", 1, None),
    ("lowlevel.spectral_rms.var", 1, None),
    ("lowlevel.spectral_rolloff.dmean", 1, None),
    ("lowlevel.spectral_rolloff.dmean2", 1, None),
    ("lowlevel.spectral_rolloff.dvar", 1, None),
    ("lowlevel.spectral_rolloff.dvar2", 1, None),
    ("lowlevel.spectral_rolloff.max", 1, None),
    ("lowlevel.spectral_rolloff.mean", 1, None),
    ("lowlevel.spectral_rolloff.median", 1, None),
    ("lowlevel.spectral_rolloff.min", 1, None),
    ("lowlevel.spectral_rolloff.stdev", 1, None),
    ("lowlevel.spectral_rolloff.var", 1, None),
    ("lowlevel.spectral_skewness.dmean", 1, None),
    ("lowlevel.spectral_skewness.dmean2", 1, None),
    ("lowlevel.spectral_skewness.dvar", 1, None),
    ("lowlevel.spectral_skewness.dvar2", 1, None),
    ("lowlevel.spectral_skewness.max", 1, None),
    ("lowlevel.spectral_skewness.mean", 1, None),
    ("lowlevel.spectral_skewness.median", 1, None),
    ("lowlevel.spectral_skewness.min", 1, None),
    ("lowlevel.spectral_skewness.stdev", 1, None),
    ("lowlevel.spectral_skewness.var", 1, None),
    ("lowlevel.spectral_spread.dmean", 1, None),
    ("lowlevel.spectral_spread.dmean2", 1, None),
    ("lowlevel.spectral_spread.dvar", 1, None),
    ("lowlevel.spectral_spread.dvar2", 1, None),
    ("lowlevel.spectral_spread.max", 1, None),
    ("lowlevel.spectral_spread.mean", 1, None),
    ("lowlevel.spectral_spread.median", 1, None),
    ("lowlevel.spectral_spread.min", 1, None),
    ("lowlevel.spectral_spread.stdev", 1, None),
    ("lowlevel.spectral_spread.var", 1, None),
    ("lowlevel.spectral_strongpeak.dmean", 1, None),
    ("lowlevel.spectral_strongpeak.dmean2", 1, None),
    ("lowlevel.spectral_strongpeak.dvar", 1, None),
    ("lowlevel.spectral_strongpeak.dvar2", 1, None),
    ("lowlevel.spectral_strongpeak.max", 1, None),
    ("lowlevel.spectral_strongpeak.mean", 1, None),
    ("lowlevel.spectral_strongpeak.median", 1, None),
    ("lowlevel.spectral_strongpeak.min", 1, None),
    ("lowlevel.spectral_strongpeak.stdev", 1, None),
    ("lowlevel.spectral_strongpeak.var", 1, None),
    ("lowlevel.zerocrossingrate.dmean", 1, None),
    ("lowlevel.zerocrossingrate.dmean2", 1, None),
    ("lowlevel.zerocrossingrate.dvar", 1, None),
    ("lowlevel.zerocrossingrate.dvar2", 1, None),
    ("lowlevel.zerocrossingrate.max", 1, None),
    ("lowlevel.zerocrossingrate.mean", 1, None),
    ("lowlevel.zerocrossingrate.median", 1, None),
    ("lowlevel.zerocrossingrate.min", 1, None),
    ("lowlevel.zerocrossingrate.stdev", 1, None),
    ("lowlevel.zerocrossingrate.var", 1, None),
    ("rhythm.beats_count", 1, None),
    ("rhythm.beats_loudness.dmean", 1, None),
    ("rhythm.beats_loudness.dmean2", 1, None),
    ("rhythm.beats_loudness.dvar", 1, None),
    ("rhythm.beats_loudness.dvar2", 1, None),
    ("rhythm.beats_loudness.max", 1, None),
    ("rhythm.beats_loudness.mean", 1, None),
    ("rhythm.beats_loudness.median", 1, None),
    ("rhythm.beats_loudness.min", 1, None),
    ("rhythm.beats_loudness.stdev", 1, None),
    ("rhythm.beats_loudness.var", 1, None),
    ("rhythm.beats_loudness_band_ratio.dmean", 6, None),
    ("rhythm.beats_loudness_band_ratio.dmean2", 6, None),
    ("rhythm.beats_loudness_band_ratio.dvar", 6, None),
    ("rhythm.beats_loudness_band_ratio.dvar2", 6, None),
    ("rhythm.beats_loudness_band_ratio.max", 6, None),
    ("rhythm.beats_loudness_band_ratio.mean", 6, None),
    ("rhythm.beats_loudness_band_ratio.median", 6, None),
    ("rhythm.beats_loudness_band_ratio.min", 6, None),
    ("rhythm.beats_loudness_band_ratio.stdev", 6, None),
    ("rhythm.beats_loudness_band_ratio.var", 6, None),
    ("rhythm.bpm", 1, None),
    ("rhythm.bpm_histogram", 250, None),
    ("rhythm.bpm_histogram_first_peak_bpm", 1, None),
    ("rhythm.bpm_histogram_first_peak_weight", 1, None),
    ("rhythm.bpm_histogram_second_peak_bpm", 1, None),
    ("rhythm.bpm_histogram_second_peak_spread", 1, None),
    ("rhythm.bpm_histogram_second_peak_weight", 1, None),
    ("rhythm.danceability", 1, None),
    ("rhythm.onset_rate", 1, None),
    ("tonal.chords_changes_rate", 1, None),
    ("tonal.chords_histogram", 24, None),
    ("tonal.chords_number_rate", 1, None),
    ("tonal.chords_strength.dmean", 1, None),
    ("tonal.chords_strength.dmean2", 1, None),
    ("tonal.chords_strength.dvar", 1, None),
    ("tonal.chords_strength.dvar2", 1, None),
    ("tonal.chords_strength.max", 1, None),
    ("tonal.chords_strength.mean", 1, None),
    ("tonal.chords_strength.median", 1, None),
    ("tonal.chords_strength.min", 1, None),
    ("tonal.chords_strength.stdev", 1, None),
    ("tonal.chords_strength.var", 1, None),
    ("tonal.hpcp.dmean", 36, None),
    ("tonal.hpcp.dmean2", 36, None),
    ("tonal.hpcp.dvar", 36, None),
    ("tonal.hpcp.dvar2", 36, None),
    ("tonal.hpcp.max", 36, None),
    ("tonal.hpcp.mean", 36, None),
    ("tonal.hpcp.median", 36, None),
    ("tonal.hpcp.min", 36, None),
    ("tonal.hpcp.stdev", 36, None),
    ("tonal.hpcp.var", 36, None),
    ("tonal.hpcp_crest.dmean", 1, None),
    ("tonal.hpcp_crest.dmean2", 1, None),
    ("tonal.hpcp_crest.dvar", 1, None),
    ("tonal.hpcp_crest.dvar2", 1, None),
    ("tonal.hpcp_crest.max", 1, None),
    ("tonal.hpcp_crest.mean", 1, None),
    ("tonal.hpcp_crest.median", 1, None),
    ("tonal.hpcp_crest.min", 1, None),
    ("tonal.hpcp_crest.stdev", 1, None),
    ("tonal.hpcp_crest.var", 1, None),
    ("tonal.hpcp_entropy.dmean", 1, None),
    ("tonal.hpcp_entropy.dmean2", 1, None),
    ("tonal.hpcp_entropy.dvar", 1, None),
    ("tonal.hpcp_entropy.dvar2", 1, None),
    ("tonal.hpcp_entropy.max", 1, None),
    ("tonal.hpcp_entropy.mean", 1, None),
    ("tonal.hpcp_entropy.median", 1, None),
    ("tonal.hpcp_entropy.min", 1, None),
    ("tonal.hpcp_entropy.stdev", 1, None),
    ("tonal.hpcp_entropy.var", 1, None),
    ("tonal.key_edma.strength", 1, None),
    ("tonal.key_krumhansl.strength", 1, None),
    ("tonal.key_temperley.strength", 1, None),
    ("tonal.thpcp", 36, None),
    ("tonal.tuning_diatonic_strength", 1, None),
    ("tonal.tuning_equal_tempered_deviation", 1, None),
    ("tonal.tuning_frequency", 1, None),
    ("tonal.tuning_nontempered_energy_ratio", 1, None),
    ("rhythm.beats_position", 4, "stats4"),
)


def schema_fingerprint() -> str:
    return hashlib.sha256(repr(_DESCRIPTOR_SCHEMA).encode()).hexdigest()[:16]


def _synthesize_wav(
    path: pathlib.Path, duration_s: float = 3.0, sr: int = 44100
) -> None:
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(sr * duration_s)) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def assert_schema_dim_consistent(profile_path: pathlib.Path | None = None) -> None:
    if not _DESCRIPTOR_SCHEMA:
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = pathlib.Path(tmp_dir) / "dim_check_noise.wav"
        _synthesize_wav(wav_path)
        extractor = get_essentia_extractor(profile_path)
        features, _frames = extractor(str(wav_path))
    pool_names = set(features.descriptorNames())
    mismatches = []
    for name, expected_length, normalizer_key in _DESCRIPTOR_SCHEMA:
        if name not in pool_names:
            continue
        raw = np.asarray(features[name])
        if normalizer_key is not None:
            arr = _NORMALIZERS[normalizer_key](raw)
        else:
            arr = raw.astype(np.float32).reshape(-1)
        if len(arr) != expected_length:
            mismatches.append(
                f"  {name}: schema declares length {expected_length}, "
                f"got {len(arr)} (raw shape {raw.shape})"
            )
    if mismatches:
        bullet_list = "\n".join(mismatches)
        raise RuntimeError(
            f"Schema dimension mismatch:\n{bullet_list}\n"
            f"Update _DESCRIPTOR_SCHEMA or re-run: "
            f"poetry run python -m audit.descriptor_shapes discover ..."
        )


def decode_audio(audio_path: pathlib.Path, sample_rate: int = 16000) -> bytes:
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout[44:]


def get_essentia_extractor(profile_path: pathlib.Path | None = None):
    if profile_path is None:
        profile_path = config.data_path / "essentia_extractor_profile.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Essentia profile not found: {profile_path}")
    return es.MusicExtractor(profile=str(profile_path))


def _essentia_pool_to_vector(pool) -> np.ndarray:
    pool_names = set(pool.descriptorNames())
    parts = []
    for name, expected_length, normalizer_key in _DESCRIPTOR_SCHEMA:
        if name not in pool_names:
            parts.append(np.zeros(expected_length, dtype=np.float32))
            continue
        raw = np.asarray(pool[name])
        if normalizer_key is not None:
            arr = _NORMALIZERS[normalizer_key](raw)
        else:
            arr = raw.astype(np.float32).reshape(-1)
        if len(arr) < expected_length:
            arr = np.concatenate(
                [arr, np.zeros(expected_length - len(arr), dtype=np.float32)]
            )
        elif len(arr) > expected_length:
            arr = arr[:expected_length]
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def extract_essentia_features(extractor, audio_path) -> np.ndarray:
    features, _frames = extractor(str(audio_path))
    return _essentia_pool_to_vector(features)


def extract_essentia_features_segment(
    extractor,
    audio_path,
    start: float,
    end: float,
) -> np.ndarray:
    cropped_path = _ffmpeg_crop_to_tempwav(audio_path, start, end)
    try:
        features, _frames = extractor(str(cropped_path))
        return _essentia_pool_to_vector(features)
    finally:
        cropped_path.unlink(missing_ok=True)


class PANNsCNN14:
    def __init__(self, weights_path: pathlib.Path):
        self.tagger = AudioTagging(
            checkpoint_path=str(weights_path),
            device="cpu",
        )

    def extract(self, audio_path: pathlib.Path) -> np.ndarray:
        waveform, _sr = librosa.load(str(audio_path), sr=32000, mono=True)
        _clipwise_output, embedding = self.tagger.inference(waveform[None, :])
        return embedding.reshape(-1)

    def extract_segment(
        self, audio_path: pathlib.Path, start_s: float, end_s: float
    ) -> np.ndarray:
        waveform, _sr = librosa.load(
            str(audio_path),
            sr=32000,
            mono=True,
            offset=start_s,
            duration=end_s - start_s,
        )
        if len(waveform) == 0:
            return np.zeros(2048, dtype=np.float32)
        _clipwise_output, embedding = self.tagger.inference(waveform[None, :])
        return embedding.reshape(-1)


def _ffmpeg_crop_to_tempwav(
    audio_path: pathlib.Path, start_s: float, end_s: float
) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        str(start_s),
        "-to",
        str(end_s),
        "-ac",
        "1",
        "-ar",
        "16000",
        tmp.name,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return pathlib.Path(tmp.name)


def prepare_extractor(
    profile_path: pathlib.Path | None = None,
    panns_weights_path: pathlib.Path | None = None,
) -> CombinedExtractor:
    if panns_weights_path is None:
        panns_weights_path = config.panns_weights_path
    essentia_extractor = get_essentia_extractor(profile_path)
    panns_model = PANNsCNN14(panns_weights_path)
    return CombinedExtractor(
        essentia_extractor=essentia_extractor,
        panns_model=panns_model,
        essentia_extract_fn=extract_essentia_features,
        essentia_extract_segment_fn=extract_essentia_features_segment,
    )
