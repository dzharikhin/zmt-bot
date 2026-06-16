import numpy as np

from audio.aggregation import aggregate
from audio.segments import SegmentSpec, get_segments


class CombinedExtractor:
    def __init__(
        self,
        essentia_extractor,
        panns_model,
        essentia_extract_fn,
        essentia_extract_segment_fn,
        profile_path=None,
    ):
        self.profile_path = profile_path
        self.essentia_extractor = essentia_extractor
        self.panns_model = panns_model
        self.essentia_extract_fn = essentia_extract_fn
        self.essentia_extract_segment_fn = essentia_extract_segment_fn

    def __call__(
        self,
        audio_path,
        segment_spec=None,
    ):
        spec = segment_spec or SegmentSpec(
            type="full", window_s=None, k=None, aggregation="mean"
        )
        segments = get_segments(audio_path, spec)

        essentia_vectors = []
        panns_vectors = []

        for start, end in segments:
            if spec.type == "full":
                essentia_vec = self.essentia_extract_fn(
                    self.essentia_extractor, audio_path, self.profile_path
                )
                panns_vec = self.panns_model.extract(audio_path)
            else:
                essentia_vec = self.essentia_extract_segment_fn(
                    self.essentia_extractor, audio_path, self.profile_path, start, end
                )
                panns_vec = self.panns_model.extract_segment(audio_path, start, end)

            essentia_vectors.append(essentia_vec)
            panns_vectors.append(panns_vec)

        essentia_agg = aggregate(essentia_vectors, spec.aggregation)
        panns_agg = aggregate(panns_vectors, spec.aggregation)
        return np.concatenate([essentia_agg, panns_agg])


def extract_features_for_mp3(audio_path, extractor):
    return extractor(audio_path)
