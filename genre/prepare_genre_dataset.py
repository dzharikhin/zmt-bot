import csv
import logging
import pathlib
import tempfile
from typing import cast

import atomics
from soundfile import LibsndfileError

from audio.features import extract_features_for_mp3, AudioFeaturesType
from dataset.persistent_dataset_processor import DataSetFromDataManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def prepare_audio_features_dataset(results_dir: pathlib.Path, audio_dir: pathlib.Path) -> pathlib.Path:
    counter = atomics.atomic(width=4, atype=atomics.INT)
    dataset_path = results_dir.joinpath("audio_features_dataset.csv")
    fails_path = results_dir.joinpath(f"{dataset_path.stem}-processing_failed.csv")
    with tempfile.TemporaryDirectory() as tmp:
        dataset_manager = DataSetFromDataManager(
            dataset_path,
            row_schema=(("track_id", str), *[(f"f{i}", float) for i in range(1, 95)]),
            index_generator=(f.stem for f in audio_dir.iterdir() if f.is_file()),
            intermediate_results_dir=pathlib.Path(tmp),
            batch_size=1000,
        )
        with dataset_manager as ds:

            def generate_features(row_id: str) -> AudioFeaturesType:
                try:
                    row = extract_features_for_mp3(
                        track_id=row_id,
                        mp3_path=audio_dir.joinpath(f"{row_id}.mp3"),
                    )
                except LibsndfileError as e:
                    with fails_path.open(mode="a") as fail_file:
                        csv.writer(fail_file).writerow([row_id, str(e)])
                    logging.warning(
                        f"failed to get features for {row_id},added to fail log, returning stub: {e}"
                    )
                    return cast(AudioFeaturesType, tuple([row_id] + [None] * 94))

                done = counter.fetch_inc() + 1
                if done % 100 == 0:
                    logging.info(f"feature generation calls/dataset_size stat: {done}/{ds.size}")
                return row

            ds.fill(generate_features)
            logging.info(f"total feature generation calls/dataset_size stat: {counter.load()}/{ds.size}")
    with fails_path.open(mode="rt") as fails:
        failed_row_ids = {row[0] for row in csv.reader(fails)}
    dataset_manager.remove_failures_in_place(failed_row_ids)
    return dataset_path


if __name__ == "__main__":
    prepare_audio_features_dataset(
        pathlib.Path("."), pathlib.Path("/home/jrx/snippets")
    )
