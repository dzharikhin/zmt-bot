import polars as pl

from train import train_dissimilar_model, LIKED_COLUMN_NAME, ID_COLUMN_NAME

if __name__ == "__main__":
    frames = 0
    liked_dataframe = pl.scan_csv(f"data/audio_features_dataset_f{frames}-1.csv")
    disliked_dataframe = pl.scan_csv(f"data/audio_features_dataset_f{frames}-0.csv")
    data = (
        pl.concat([liked_dataframe, disliked_dataframe])
        .collect(engine="streaming")
        .drop_nulls()
        .sample(fraction=1, shuffle=True)
    )
    data_stats = data.group_by(by=pl.col(LIKED_COLUMN_NAME)).agg(
        pl.col(LIKED_COLUMN_NAME).count()
    )
    test_track_ids = (
        data.group_by(pl.col(LIKED_COLUMN_NAME))
        .agg(
            pl.col(ID_COLUMN_NAME).sample(
                fraction=0.3, with_replacement=False, shuffle=True
            )
        )
        .explode(pl.col(ID_COLUMN_NAME))
        .select(ID_COLUMN_NAME)
    )

    test_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME))
    )
    train_data = data.filter(
        pl.col(ID_COLUMN_NAME).is_in(test_track_ids.get_column(ID_COLUMN_NAME)).not_()
    )
    model, accuracy = train_dissimilar_model(
        train_data=train_data, test_data=test_data, contamination_fraction=0.2
    )
    print(f"for {frames=} {accuracy=}")
