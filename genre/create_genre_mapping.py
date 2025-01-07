import pathlib
import polars as pl

GENRE_COLUMN_NAME = "genre_name"
ID_COLUMN_NAME = "spotify_id"

MANUAL_GENRE_CLUSTERING = {
    "a cappella": [
        "a cappella",
    ],
    "abstract": ["abstract", "abstractro"],
    "hip hop": [
        "hip hop",
        "abstract beats",
        "abstract hip hop",
        "alternative hip hop",
        "australian hip hop",
        "austrian hip hop",
        "bay area hip hop",
        "brazilian hip hop",
        "canadian hip hop",
        "chip hop",
        "christian hip hop",
        "czech hip hop",
        "danish hip hop",
        "deep dutch hip hop",
        "deep east coast hip hop",
        "deep german hip hop",
        "deep latin hip hop",
        "deep swedish hip hop",
        "deep underground hip hop",
        "detroit hip hop",
        "dutch hip hop",
        "east coast hip hop",
        "finnish hip hop",
        "french hip hop",
        "german hip hop",
        "greek hip hop",
        "hardcore hip hop",
        "hip hop quebecois",
        "hip hop tuga",
        "hungarian hip hop",
        "italian hip hop",
        "latin hip hop",
        "memphis hip hop",
        "norwegian hip hop",
        "old school hip hop",
        "outer hip hop",
        "polish hip hop",
        "russian hip hop",
        "slovak hip hop",
        "southern hip hop",
        "spanish hip hop",
        "swedish hip hop",
        "swiss hip hop",
        "turkish hip hop",
        "uk hip hop",
        "underground hip hop",
        "underground latin hip hop",
    ],
    "accordeon": [
        "accordeon",
        "accordion",
    ],
    "house": [
        "acid house",
        "chicago house",
        "deep deep house",
        "deep deep tech house",
        "deep disco house",
        "deep euro house",
        "deep funk house",
        "deep groove house",
        "deep house",
        "deep melodic euro house",
        "deep progressive house",
        "deep soul house",
        "deep tech house",
        "deep tropical house",
        "deep vocal house",
        "disco house",
        "dutch house",
        "electro house",
        "fidget house",
        "filter house",
        "float house",
        "funky tech house",
        "hip house",
        "kwaito house",
        "minimal tech house",
        "nordic house",
        "pop house",
        "progressive electro house",
        "tech house",
        "tribal house",
        "tropical house",
        "vocal house",
        "",
    ],
    "progressive": [
        "dark progressive house",
        "greek house",
        "microhouse",
        "outsider house",
        "progressive trance house",
    ],
    "hardstyle": ["bouncy house", "hard house", ""],
    "lounge": ["lounge house", ""],
    "trance": ["progressive house", ""],
    # "jazzy": ["acid jazz",],
    "techno": ["techno", "acid techno", ""],
    # "instrumental": ["acousmatic",],
}


def main(*, genre_dataset_path: pathlib.Path, mapping_path: pathlib.Path):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    inverse_mapping = {
        specific_name: common_name
        for common_name, specific_names in MANUAL_GENRE_CLUSTERING.items()
        for specific_name in specific_names
    }

    def mapper(genre_name: str) -> str:
        return inverse_mapping.get(genre_name, genre_name)

    meaningful_genres_data = (
        genre_dataset.group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(pl.len().gt(100).alias("examples_cnt"))
        .filter(pl.col("examples_cnt"))
        .select(pl.col(GENRE_COLUMN_NAME))
        .collect()
    )
    genre_dataset = genre_dataset.select(
        pl.col("spotify_id"), pl.col(GENRE_COLUMN_NAME)
    ).filter(pl.col(GENRE_COLUMN_NAME).is_in(meaningful_genres_data))

    genre_dataset.with_columns(
        pl.col(GENRE_COLUMN_NAME)
        .map_elements(mapper, pl.String)
        .alias(f"{GENRE_COLUMN_NAME}_mapped")
    ).with_columns(
        pl.col(f"{GENRE_COLUMN_NAME}_mapped").alias(GENRE_COLUMN_NAME)
    ).select(
        pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME)
    ).group_by(
        pl.col(GENRE_COLUMN_NAME)
    ).agg(
        pl.col(f"{ID_COLUMN_NAME}")
    ).with_columns(
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
