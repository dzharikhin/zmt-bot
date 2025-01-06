import pathlib
import polars as pl

GENRE_COLUMN_NAME = "genre_name"
ID_COLUMN_NAME = "spotify_id"


def main(*, genre_dataset_path: pathlib.Path, mapping_path: pathlib.Path):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    genre_dataset.select(pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)).group_by(
        pl.col(GENRE_COLUMN_NAME)
    ).agg(pl.col(f"{ID_COLUMN_NAME}")).with_columns(
        pl.col(ID_COLUMN_NAME).list.join(";")
    ).collect().sort(
        by=pl.col(GENRE_COLUMN_NAME)
    ).write_csv(
        mapping_path
    )


if __name__ == "__main__":
    main(
        genre_dataset_path=pathlib.Path("csv/songs.csv.prepared"),
        mapping_path=pathlib.Path("csv/songs.grouped_by_genre.csv"),
    )
