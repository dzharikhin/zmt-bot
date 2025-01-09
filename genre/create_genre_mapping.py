import pathlib
from collections import defaultdict

import polars as pl

GENRE_COLUMN_NAME = "genre_name"
ID_COLUMN_NAME = "track_id"

# genres grouped by not sound criteria
MIXED_GENRES = {
    "abstract",
    "abstractro",
    "dance",
    "adult standards",
    "acousmatic",
    "ambient",
    "anime score",
    "anime cv",
    "anime",
    "afrikaans",
    "accordeon",
    "a cappella",
    "aussietronica",
    "avant-garde",
}

# fixing and enhancing automapping
MANUAL_GENRE_MAPPING = {
    "hip hop": {
        # "abstract beats",
        "chip hop",
        "deep hip hop",
        "deep underground hip hop",
        "hardcore hip hop",
        "hip hop quebecois",
        "hip hop tuga",
        "outer hip hop",
        "underground latin hip hop",
        "rap",
        "chicano rap",
        "dirty south rap",
        "dirty texas rap",
        "gangster rap",
    },
    "accordeon": {
        "accordion",
    },
    "house": {
        "acid house",
        "chicago house",
        "deep house",
        "deep tech house",
        "disco house",
        "euro house",
        "funk house",
        "groove house",
        "melodic euro house",
        "progressive house",
        "soul house",
        "tech house",
        "tropical house",
        "vocal house",
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
    },
    "progressive": {
        "dark progressive house",
        "greek house",
        "microhouse",
        "outsider house",
        "progressive trance house",
    },
    "hardstyle": {
        "bouncy house",
        "hard house",
    },
    "trance": {
        "progressive house",
    },
    "techno": {
        "acid techno",
        "aggrotech",
    },
    "pop": {
        "acoustic pop",
        "alternative pop",
        "antiviral pop",
        "arena pop",
        "britpop",
        "folk-pop",
        "anthem worship",
        "austropop",
        "axe",
    },
    "alternative rock": {
        "alternative pop rock",
        "alternative rock",
        "rap rock",
        "nu metal",
        "alternative metal",
    },
    "rock": {
        "alt-indie rock",
        "alternative",
        "anti-folk",
        "art rock",
        "australian alternative rock",
    },
    "indie": {
        "irish indie",
        "alternative ccm",
        "austindie",
    },
    "afrobeat": {
        "afrobeats",
    },
    "country": {
        "americana",
    },
    "blues": {
        "acoustic blues",
    },
    "beats": {
        "ambeat",
    },
    "punk": {
        "anarcho-punk",
    },
    "andean folk": {"andean"},
    "emo": {"anthem emo"},
    "latin": {
        "azonto",
        "azontobeats",
    },
}

GENRE_VARIATIONS = {
    "album": False,
    "alternative": False,
    "alt-": False,
    "deep": True,
    "abstract": False,
    "underground": False,
    "old school": False,
    "lounge": False,
    "traditional": False,
    "modern": False,
    "hardcore": True,
    "classic": False,
    "atmospheric": False,
    "ambient": False,
    "vocal": False,
    "vintage": False,
    "retro": False,
    "regional": False,
    "melodic": False,
    "hard": True,
    "geek": False,
    "garage": False,
    "experimental": True,
    "avant-garde": False,
}

GEOGRAPHY_LABELS = {
    "african",
    "albanian",
    "albuquerque",
    "appalachian",
    "arab",
    "argentine",
    "armenian",
    "athens",
    "australian",
    "austrian",
    "bay area",
    "belgian",
    "brazilian",
    "british",
    "brooklyn",
    "bulgarian",
    "canadian",
    "catalan",
    "caucasian",
    "celtic",
    "chicago",
    "chinese",
    "colombian",
    "columbus ohio",
    "corsican",
    "croatian",
    "cuban",
    "czech",
    "dallas",
    "danish",
    "denver",
    "dominican",
    "dutch",
    "east coast",
    "ethiopian",
    "estonian",
    "german",
    "hebrew",
    "indian",
    "indonesian",
    "irish",
    "icelandic",
    "israeli",
    "j-",
    "japanese",
    "k-",
    "kiwi",
    "korean",
    "kurdish",
    "la",
    "latin",
    "latvian",
    "swedish",
    "detroit",
    "finnish",
    "french",
    "greek",
    "hungarian",
    "italian",
    "malaysian",
    "memphis",
    "mexican",
    "michigan",
    "nordic",
    "northern",
    "norwegian",
    "pakistani",
    "persian",
    "perth",
    "peruvian",
    "polish",
    "polynesian",
    "portland",
    "portuguese",
    "romanian",
    "russian",
    "scottish",
    "seattle",
    "slavic",
    "slovak",
    "slovenian",
    "southern",
    "spanish",
    "suomi",
    "swiss",
    "taiwanese",
    "texas",
    "thai",
    "turkish",
    "uk",
    "ukrainian",
    "vancouver",
    "venezuelan",
    "vegas",
    "vienna",
    "welsh",
    "west african",
    "west coast",
    "yugoslav",
    # not geo, but
    "christian",
    "viking",
}


def _collapse_genre_geography(unique_genres: list[str]) -> dict[str, set[str]]:

    mapping = defaultdict(set)
    for genre in unique_genres:
        if genre.endswith("folk"):
            continue
        raw_genre = genre
        variation, distinct = next(
            (
                (var, distinct)
                for var, distinct in GENRE_VARIATIONS.items()
                if genre.startswith(var if var.endswith("-") else f"{var} ")
            ),
            (None, None),
        )
        if variation:
            genre = genre.replace(variation, "").strip()

        geo = next(
            (
                geo
                for geo in GEOGRAPHY_LABELS
                if genre.startswith(geo if geo.endswith("-") else f"{geo} ")
            ),
            None,
        )
        if geo:
            genre = genre.replace(geo, "").strip()
        if (
            genre
            and genre != raw_genre
            and (key := f"{variation} {genre}" if distinct else genre) != raw_genre
        ):
            mapping[key].add(raw_genre)
    return mapping


def _merge_mapping(
    mapping: dict[str, set[str]], overriding_mapping: dict[str, set[str]]
) -> dict[str, set[str]]:
    merged = {}
    for genre, raw_genres in mapping.items():
        if overriding_mappings := overriding_mapping.get(genre):
            for override_mapping in overriding_mappings:
                if override_mapping.startswith("-"):
                    raw_genres.remove(override_mapping.removeprefix("-"))
                else:
                    raw_genres.add(override_mapping)
        merged[genre] = raw_genres
    for genre, override_mappings in overriding_mapping.items():
        if genre not in merged:
            merged[genre] = override_mappings
        for override_mapping in override_mappings:
            if override_mapping in merged:
                merged[genre].update(merged[override_mapping])
                del merged[override_mapping]
    return merged


def group_genres(
    *,
    genre_dataset_path: pathlib.Path,
    grouped_by_genre_path: pathlib.Path,
    mapped_genre_dataset_path: pathlib.Path,
):
    genre_dataset = pl.scan_csv(genre_dataset_path)
    # meaningful_genres_data = (
    #     genre_dataset.group_by(pl.col(GENRE_COLUMN_NAME))
    #     .agg(pl.len().gt(100).alias("examples_cnt"))
    #     .filter(pl.col("examples_cnt"))
    #     .select(pl.col(GENRE_COLUMN_NAME))
    #     .collect()
    # )
    # genre_dataset = genre_dataset.filter(pl.col(GENRE_COLUMN_NAME).is_in(meaningful_genres_data))
    unique_genres: list[str] = (
        genre_dataset.select(pl.col(GENRE_COLUMN_NAME))
        .unique()
        .collect()
        .get_column(GENRE_COLUMN_NAME)
        .to_list()
    )
    merged_mapping = _merge_mapping(
        _collapse_genre_geography(unique_genres), MANUAL_GENRE_MAPPING
    )
    print(f"{merged_mapping=}")
    inverse_mapping = {
        specific_name: common_name
        for common_name, specific_names in merged_mapping.items()
        for specific_name in specific_names
    }

    def mapper(genre_name: str) -> str:
        return inverse_mapping.get(genre_name, genre_name)

    genre_dataset.with_columns(
        pl.col(GENRE_COLUMN_NAME)
        .map_elements(mapper, pl.String)
        .alias(f"{GENRE_COLUMN_NAME}_mapped")
    ).with_columns(
        pl.col(f"{GENRE_COLUMN_NAME}_mapped").alias(GENRE_COLUMN_NAME)
    ).filter(
        pl.col(GENRE_COLUMN_NAME).is_in(list(MIXED_GENRES)).not_()
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
        grouped_by_genre_path
    )


if __name__ == "__main__":
    group_genres(
        genre_dataset_path=pathlib.Path("csv/songs-genre_filtered.csv"),
        grouped_by_genre_path=pathlib.Path(
            "csv/songs-genre_filtered-grouped_by_genre.csv"
        ),
        mapped_genre_dataset_path=pathlib.Path(
            "csv/songs-genre_filtered-mapped_genres.csv"
        ),
    )
