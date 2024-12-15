import pathlib

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from audio.features import extract_features_for_mp3
from genre.genre_dataset import Dataset


def main(*, genre_dataset_path: pathlib.Path, snippets_dir: pathlib.Path):
    genre_dataframe = Dataset(genre_dataset_path).dataframe

    rows = {}
    for snippet in pathlib.Path(snippets_dir).iterdir():
        track_id, data_row = extract_features_for_mp3(
            track_id=snippet.stem, mp3_path=snippet
        )
        rows[track_id] = data_row

    audio_features_dataframe = pd.DataFrame.from_dict(rows, orient="index")
    audio_features_dataframe.index.name = "track_id"

    genre_with_features = pd.merge(
        genre_dataframe,
        audio_features_dataframe,
        left_index=True,
        right_index=True,
    )

    data = genre_with_features.drop(columns=["name", "artist", "position"])
    data["genre_name"].astype("category")
    data = data.values
    X = data[:, 1:]
    y = data[:, 0]
    model = xgb.XGBClassifier(tree_method="hist")
    label_encoder = LabelEncoder().fit(y)
    label_encoded_y = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, label_encoded_y, test_size=0.3, random_state=42
    )

    model.fit(X, label_encoded_y)
    predicted = model.predict(X_test)

    print(accuracy_score(y_test, predicted))



if __name__ == "__main__":
    main(
        genre_dataset_path=pathlib.Path("csv/songs.csv"),
        snippets_dir=pathlib.Path("snippets"),
    )
