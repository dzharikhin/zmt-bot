import pathlib
import tempfile

from train import _prepare_audio_features_dataset

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=pathlib.Path("data/tmp"), delete=False) as tmp:
        liked_data = _prepare_audio_features_dataset(
            audio_dir=pathlib.Path("data/118517468/liked"),
            tmp_dir=pathlib.Path(tmp),
            results_dir=pathlib.Path("data"),
            is_liked=True,
        )
        disliked_data = _prepare_audio_features_dataset(
            audio_dir=pathlib.Path("data/118517468/disliked"),
            tmp_dir=pathlib.Path(tmp),
            results_dir=pathlib.Path("data"),
            is_liked=False,
        )
