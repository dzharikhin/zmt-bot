import dataclasses
import logging
import pathlib
import sys

import polars as pl

from audio.features import extract_features_for_mp3, AudioFeatures
from dataset.persistent_dataset_processor import DataFrameBuilder

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclasses.dataclass
class MappedAudioFeatures(AudioFeatures):
    liked: int


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    def stats(progress_state: DataFrameBuilder.ProgressStat):
        logger.error(
            f"{progress_state.succeed=}/{progress_state.failed=}/{progress_state.data_size=}"
        )

    liked_audio_path = pathlib.Path("data/118517468/liked")
    disliked_audio_path = pathlib.Path("data/118517468/disliked")

    def liked_feature_extractor(track_id: str) -> MappedAudioFeatures:
        features = extract_features_for_mp3(liked_audio_path.joinpath(track_id))
        logger.info(f"extracted features for {track_id=}")
        return MappedAudioFeatures(**dataclasses.asdict(features), liked=1)

    def disliked_feature_extractor(track_id: str) -> MappedAudioFeatures:
        features = extract_features_for_mp3(disliked_audio_path.joinpath(track_id))
        return MappedAudioFeatures(liked=0, **dataclasses.asdict(features))

    with DataFrameBuilder(
        working_dir=pathlib.Path("data/tmp/liked_dir"),
        index_generator=DataFrameBuilder.GeneratingParams(
            generator=(f.name for f in liked_audio_path.iterdir() if f.is_file()),
            result_schema=("row_id", pl.String),
        ),
        mappers=[liked_feature_extractor],
        batch_size=1,
        cleanup_on_exit=False,
        progress_tracker=stats,
    ) as liked_frame, DataFrameBuilder(
        working_dir=pathlib.Path("data/tmp/disliked_dir"),
        index_generator=DataFrameBuilder.GeneratingParams(
            generator=(f.name for f in disliked_audio_path.iterdir() if f.is_file()),
            result_schema=("row_id", pl.String),
        ),
        mappers=[disliked_feature_extractor],
        batch_size=1,
        cleanup_on_exit=False,
        progress_tracker=stats,
    ) as disliked_frame:
        print(liked_frame)
        print(disliked_frame)
