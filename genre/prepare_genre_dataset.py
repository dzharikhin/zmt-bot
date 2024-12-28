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

dataset_path = pathlib.Path("features_to_genres_dataset.csv")
fails_path = pathlib.Path(f"{dataset_path.stem}-Processing_failed.csv")
counter = atomics.atomic(width=4, atype=atomics.INT)

snippets_path = pathlib.Path("/home/jrx/snippets")

if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as tmp:
        dataset_manager = DataSetFromDataManager(
            dataset_path,
            row_schema=(("track_id", str), *[(f"f{i}", float) for i in range(1, 95)]),
            index_generator=(f.stem for f in snippets_path.iterdir() if f.is_file()),
            intermediate_results_dir=pathlib.Path(tmp),
            batch_size=1000,
        )
        with dataset_manager as ds:

            def generate_features(row_id: str) -> AudioFeaturesType:
                try:
                    row = extract_features_for_mp3(
                        track_id=row_id,
                        mp3_path=snippets_path.joinpath(f"{row_id}.mp3"),
                    )
                except LibsndfileError as e:
                    with fails_path.open(mode="a") as fails:
                        csv.writer(fails).writerow([row_id, str(e)])
                    logging.warning(
                        f"failed to get features for {row_id},added to fail log, returning stub: {e}"
                    )
                    return cast(AudioFeaturesType, tuple([row_id] + [None] * 94))

                done = counter.fetch_inc() + 1
                if done % 100 == 0:
                    logging.info(f"{done}/{ds.size}")
                return row

            ds.fill(generate_features)
            logging.info(f"totally processed: {counter.load()}/{ds.size}")
    with fails_path.open(mode="rt") as fails:
        failed_row_ids = {row[0] for row in csv.reader(fails)}
    dataset_manager.remove_failures_in_place(failed_row_ids)
