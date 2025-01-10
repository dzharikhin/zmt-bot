import pathlib
import sys
from collections import defaultdict
from functools import reduce

import polars as pl

GENRE_COLUMN_NAME = "genre_name"
ID_COLUMN_NAME = "track_id"

# genres grouped by not sound criteria or sounding very non-uniform
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
    "barnemusikk",
    "bass music",
    "bass trip",
    "bassline",
    "beach music",
    "beats",
    "bemani",
    "big beat",
    "bmore",
    "brill building pop",
    "broken beat",
    "cantautor",
    "chamber psych",
    "children's music",
    "classical",
    "college a cappella",
    "comedy",
    "comic",
    "complextro",
    "composition",
    "compositional ambient",
    "core",
    "coverchill",
    "dansband",
    "danseband",
    "dansktop",
    "dance band",
    "demoscene",
    "destroy techno",
    "digital hardcore",
    "doujin",
    "drama",
    "drill and bass",
    "drone",
    "e6fi",
    "early music ensemble",
    "easy listening",
    "edm",
    "electro dub",
    "electro jazz",
    "electro trash",
    "electroacoustic improvisation",
    "electroclash",
    "electrofox",
    "electronic",
    "electronica",
    "environmental",
    "epicore",
    "escape room",
    "eurovision",
    "experimental",
    "fake",
    "fingerstyle",
    "free improvisation",
    "funky",
    "fusion",
    "future ambient",
    "gamecore",
    "garage",
    "girl group",
    "glitch",
    "glitch beats",
    "glitch hop",
    "industrial",
    "guidance",
    "halloween",
    "hauntology",
    "hoerspiel",
    "hollywood",
    "hop",
}

# fixing and enhancing automapping
MANUAL_GENRE_MAPPING = {
    "hip hop": {
        "bounce",
        "chicano rap",
        "chip hop",
        "deep hip hop",
        "underground hip hop",
        "dirty south rap",
        "dirty texas rap",
        "gangster rap",
        "hardcore hip hop",
        "hip hop quebecois",
        "hip hop tuga",
        "outer hip hop",
        "rap",
        "underground latin hip hop",
        "crunk",
        "flick hop",
        "flow",
        "fluxwork",
        "football",
        "fourth world",
        "francoton",
        "g funk",
        "grime",
        "hip pop",
        "horrorcore",
        "hyphy",
    },
    "trap": {
        "bass trap",
        "dwn trap",
    },
    "accordeon": {
        "accordion",
    },
    "house": {
        "acid house",
        "balearic",
        "ballroom",
        "chicago house",
        "deep house",
        "disco house",
        "disco house",
        "dutch house",
        "electro house",
        "euro house",
        "fidget house",
        "filter house",
        "float house",
        "funk house",
        "funky tech house",
        "groove house",
        "hip house",
        "kwaito house",
        "melodic euro house",
        "minimal tech house",
        "nordic house",
        "pop house",
        "progressive electro house",
        "soul house",
        "tech house",
        "tribal house",
        "tropical house",
        "tropical house",
        "vocal house",
        "vocal house",
        "-greek house",
        "groove room",
    },
    "progressive": {
        "dark progressive house",
        "greek house",
        "microhouse",
        "outsider house",
        "progressive trance house",
    },
    "progressive house": {
        "-dark progressive house",
    },
    "hardstyle": {
        "bouncy house",
        "hard house",
        "dark hardcore",
        "doomcore",
        "gabba",
        "hands up",
        "happy hardcore",
        "hardcore techno",
    },
    "hardcore": {
        "-dark hardcore",
        "hardcore rock",
        "hardcore punk",
    },
    "trance": {
        "bubble trance",
        "progressive house",
        "chill-out trance",
        "glitter trance",
    },
    "psytrance": {
        "full on",
        "goa trance",
    },
    "techno": {
        "acid techno",
        "dub techno",
    },
    "industrial": {
        "aggrotech",
        "ebm",
        "electro-industrial",
        "grave wave",
    },
    "pop": {
        "acoustic pop",
        "alternative pop",
        "anthem worship",
        "antiviral pop",
        "arena pop",
        "austropop",
        "axe",
        "bow pop",
        "boy band",
        "britpop",
        "bubblegum dance",
        "bubblegum pop",
        "folk-pop",
        "candy pop",
        "cantopop",
        "canzone napoletana",
        "carnaval",
        "ccm",
        "chamber pop",
        "channel pop",
        "chill groove",
        "chill lounge",
        "christelijk",
        "christian",
        "city pop",
        "country dawn",
        "country gospel",
        "country rock",
        "covertrance",
        "dance pop",
        "desi",
        "disco polo",
        "discofox",
        "electropowerpop",
        "etherpop",
        "europop",
        "folk pop",
        "gauze pop",
        "gospel",
    },
    "ccm": {
        "-alternative ccm",
    },
    "alternative rock": {
        "-australian alternative rock",
        "alternative metal",
        "alternative pop rock",
        "rap rock",
        "funk metal",
        "heavy alternative",
    },
    "metal": {
        "-alternative metal",
        "cyber metal",
        "fallen angel",
        "folk metal",
        "funeral doom",
        "gothic doom",
        "gothic metal",
        "gothic symphonic metal",
        "groove metal",
        "hatecore",
    },
    "pop rock": {
        "-alternative pop rock",
    },
    "rock": {
        "alternative rock",
        "anti-folk",
        "art rock",
        "australian alternative rock",
        "canterbury scene",
        "comedy rock",
        "corrosion",
        "dance rock",
        "downshift",
        "folk punk",
        "folk rock",
        "gbvfi",
        "glam rock",
        "gothic alternative",
        "gothic rock",
        "grunge pop",
        "hard glam",
        "hard stoner rock",
        "heavy gothic rock",
    },
    "post rock": {
        "crossover prog",
        "djent",
        "dream pop",
        "dreamo",
        "ethereal wave",
        "experimental psych",
        "experimental rock",
        "freakbeat",
        "gothic post-punk",
    },
    "hardcore rock": {
        "chaotic hardcore",
        "electronicore",
    },
    "black metal": {
        "black death",
        "black sludge",
        "black thrash",
        "chaotic black metal",
        "cryptic black metal",
        "depressive black metal",
        "doom metal",
    },
    "post black metal": {
        "blackgaze",
    },
    "death metal": {
        "brutal death metal",
        "brutal deathcore",
        "charred death",
        "death core",
        "deathgrind",
        "grindcore",
        "grisly death metal",
    },
    "ska": {
        "euroska",
    },
    "indie": {
        "alternative ccm",
        "austindie",
        "irish indie",
        "chillwave",
    },
    "afrobeat": {
        "afrobeats",
    },
    "country": {
        "americana",
        "commons",
        "country road",
        "cowboy western",
        "gothic americana",
    },
    "blues": {
        "acoustic blues",
        "bluegrass",
        "blues-rock guitar",
        "blues-rock",
        "country blues",
        "electric blues",
        "gospel blues",
        "harmonica blues",
    },
    "punk rock": {
        "punk",
        "anarcho-punk",
        "crack rock steady",
        "crust punk",
        "fast melodic punk",
        "horror punk",
    },
    "post punk rock": {
        "dance punk rock",
        "wave",
        "electropunk",
    },
    "andean folk": {
        "andean",
    },
    "emo": {
        "anthem emo",
        "emo punk",
    },
    "latin": {
        "azonto",
        "azontobeats",
        "banda",
        "choro",
        "cubaton",
        "hiplife",
    },
    "electro latin": {
        "electro bailando",
        "electro latino",
    },
    "bangla folk": {
        "bangla",
    },
    # # initial `a capella` genre is mixed
    # "acapella": {
    #     "barbershop",
    # },
    "barnemusikk": {
        "barnmusik",
    },
    "baroque": {
        "baroque ensemble",
    },
    "jazz": {
        "bebop",
        "big band",
        "post-bop",
        "cool jazz",
        "dixieland",
        "free jazz",
        "doo-wop",
        "gypsy jazz",
        "hard bop",
        "highlife",
    },
    "belorus folk": {
        "belorush",
    },
    "kenyan folk": {
        "benga",
    },
    "punjab folk": {
        "bhangra",
    },
    "cuban folk": {
        "bolero",
    },
    "bossa nova": {
        "bossa nova jazz",
    },
    "brazilian folk": {
        "brega",
        "forro",
    },
    "dubstep": {
        "brostep",
        "catstep",
        "chillstep",
        "cinematic dubstep",
        "dubstep product",
        "dubsteppe",
        "experimental dubstep",
        "filthstep",
        "ghoststep",
    },
    "flamenco": {
        "cante flamenco",
    },
    "indian folk": {
        "carnatic",
    },
    "scottish folk": {
        "ceilidh",
    },
    "celtik folk": {
        "celtic",
    },
    "bulgarian folk": {
        "chalga",
    },
    "argentine folk": {
        "chamame",
        "folklore argentino",
    },
    "chanson": {
        "chanson quebecois",
    },
    "synth": {
        "c64",
        "c86",
        "chiptune",
    },
    "orchestral": {
        "deep orchestral",
        "cello",
        "chamber choir",
        "choral",
        "clarinet",
        "classical flute",
        "classical organ",
        "classical performance",
        "classical period",
        "classical piano",
        "composition d",
        "concert piano",
        "consort",
        "contemporary classical",
        "early music",
        "harp",
        "harpsichord",
    },
    "piano": {
        "classify",
    },
    "marching": {
        "college marching band",
        "brass",
        "brass band",
        "brass ensemble",
    },
    "dominican folk": {
        "cornetas y tambores",
    },
    "colombian folk": {
        "cumbia",
        "cumbia funk",
        "cumbia pop",
        "cumbia sonidera",
        "cumbia villera",
    },
    "indonesian folk": {
        "dangdut",
        "gamelan",
    },
    "dance punk rock": {
        "danspunk",
        "dance-punk",
    },
    "drum and bass": {
        "darkstep",
        "drumfunk",
    },
    "sahara folk": {
        "desert blues",
    },
    "australian folk": {
        "didgeridoo",
    },
    "downtempo": {
        "downtempo fusion",
    },
    "reggae": {
        "dub",
        "gospel reggae",
    },
    "mexican folk": {
        "duranguense",
        "grupera",
    },
    "ecuadoria folk": {
        "ecuadoria",
    },
    "japanese folk": {
        "enka",
    },
    "balkan folk": {
        "entehno",
    },
    "portuguese folk": {
        "fado",
    },
    "german folk": {
        "fussball",
    },
    "spanish folk": {
        "galego",
    },
    "arab folk": {
        "ghazal",
    },
    "hawaiian folk": {
        "hawaiian",
    },
    "hindustani folk": {
        "hindustani classical",
    },
    "folk": {
        "folkmusik",
    },
    # to remove
    "classical": None,
}

GENRE_VARIATIONS = {
    "abstract": False,
    "album": False,
    "alt-": False,
    "alternative": False,
    "ambient": False,
    "atmospheric": False,
    "avant-garde": False,
    "classic": False,
    "contemporary": False,
    "dark": False,
    "deep": False,
    "drone": False,
    "experimental": True,
    "garage": False,
    "geek": False,
    "hard": False,
    "hardcore": True,
    "lounge": False,
    "melodic": False,
    "modern": False,
    "old school": False,
    "regional": False,
    "retro": False,
    "traditional": False,
    "underground": False,
    "vintage": False,
    "vocal": False,
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
    "basque",
    "bay area",
    "belgian",
    "boston",
    "brazilian",
    "breton",
    "british",
    "brooklyn",
    "bulgarian",
    "canadian",
    "catalan",
    "caucasian",
    "celtic",
    "chicago",
    "c-",
    "chilean",
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
    "faroese",
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
        elif genre in merged and not override_mappings:
            del merged[genre]

    keys_in_values = set(merged.keys()) & reduce(
        lambda a, b: a.union(b), list(merged.values())
    )
    if keys_in_values:
        value_index = {}
        for key, vals in merged.items():
            for val in vals:
                old_key = value_index.pop(val, None)
                if old_key:
                    print(
                        f"<{val}> is duplicates in keys <{old_key}> and <{key}>",
                        file=sys.stderr,
                    )
                value_index[val] = key
        for key in sorted(keys_in_values, key=lambda x: len(x), reverse=True):
            mapping_key = value_index[key]
            print(
                f"Key <{key}> is present as value in <{mapping_key}>. Merging {merged[key]} in <{mapping_key}>"
            )
            merged[mapping_key].update(merged[key])
            del merged[key]
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
