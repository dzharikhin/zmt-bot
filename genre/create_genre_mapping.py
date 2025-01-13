import pathlib
import pprint
import sys
from collections import defaultdict
from functools import reduce

import polars as pl

GENRE_COLUMN_NAME = "genre_name"
ID_COLUMN_NAME = "track_id"

# genres grouped by not sound criteria or sounding very non-uniform
MIXED_GENRES = {
    "a cappella",
    "abstract",
    "abstractro",
    "acousmatic",
    "adult standards",
    "afrikaans",
    "ambient",
    "anime cv",
    "anime score",
    "anime",
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
    "broadway",
    "broken beat",
    "cantautor",
    "chamber psych",
    "children's music",
    "chill",
    "chill-out",
    "classical",
    "college a cappella",
    "comedy",
    "comic",
    "complextro",
    "composition",
    "compositional ambient",
    "core",
    "coverchill",
    "dance band",
    "dance",
    "dansband",
    "danseband",
    "dansktop",
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
    "filmi",
    "free improvisation",
    "funky",
    "fusion",
    "future ambient",
    "gamecore",
    "garage",
    "girl group",
    "glitch beats",
    "glitch hop",
    "glitch",
    "guidance",
    "halloween",
    "hauntology",
    "hoerspiel",
    "hollywood",
    "hop",
    "idm",
    "industrial",
    "intelligent dance music",
    "invasion",
    "jam band",
    "jazztronica",
    "kids dance party",
    "kindermusik",
    "library music",
    "light music",
    "lo star",
    "lo-fi",
    "lowercase",
    "martial industrial",
    "mashup",
    "meditation",
    "melancholia",
    "metropopolis",
    "minimal",
    "motivation",
    "movie tunes",
    "music",
    "musica para ninos",
    "musica per bambini",
    "musiikkia lapsille",
    "musique concrete",
    "musique pour enfants",
    "muziek voor kinderen",
    "new weird america",
    "ninja",
    "nintendocore",
    "nu electro",
    "nursery",
    "old-time",
    "oratory",
    "organic ambient",
    "oshare kei",
    "otacore",
    "outsider",
    "performance",
    "poetry",
    "polyphony",
    "poprock",
    "preverb",
    "prog",
    "progressive rock",
    "psych-pop",
    "psychill",
    "reading",
    "redneck",
    "relaxative",
    "remix",
    "rock noise",
    "russelater",
    "schlager",
    "scorecore",
    "shimmer psych",
    "show tunes",
    "singer-songwriter",
    "skweee",
    "sleep",
    "solipsynthm",
    "song poem",
    "soundtrack",
    "speed garage",
    "spoken word",
    "spytrack",
    "steampunk",
    "strut",
    "talent show",
    "theme",
    "tico",
    "tin pan alley",
    "tone",
    "tracestep",
    "tribute",
    "tzadik",
    "video game music",
    "visual kei",
    "warm drone",
    "wonky",
    "workout",
    "zeuhl",
}

# fixing and enhancing automapping
MANUAL_GENRE_MAPPING = {
    "hip hop": {
        "bounce",
        "chicano rap",
        "chip hop",
        "crunk",
        "deep hip hop",
        "dirty south rap",
        "dirty texas rap",
        "flick hop",
        "flow",
        "fluxwork",
        "football",
        "fourth world",
        "francoton",
        "g funk",
        "gangster rap",
        "grime",
        "hardcore hip hop",
        "hip hop quebecois",
        "hip hop tuga",
        "hip pop",
        "horrorcore",
        "hyphy",
        "nerdcore",
        "outer hip hop",
        "rap chileno",
        "rap",
        "turntablism",
        "underground hip hop",
        "underground latin hip hop",
    },
    "trap": {
        "bass trap",
        "dwn trap",
        "trap francais",
        "trap music",
    },
    "accordeon": {
        "accordion",
    },
    "house": {
        "-greek house",
        "-hard house",
        "acid house",
        "balearic",
        "ballroom",
        "chicago house",
        "deep house",
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
        "groove room",
        "hip house",
        "kwaito house",
        "melbourne bounce",
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
        "dark hardcore",
        "doomcore",
        "gabba",
        "hands up",
        "happy hardcore",
        "hard house",
        "hardcore techno",
        "jumpstyle",
        "speedcore",
        "tekno",
        "terrorcore",
    },
    "hardcore": {
        "-dark hardcore",
        "hardcore rock",
        "hardcore punk",
    },
    "trance": {
        "bubble trance",
        "chill-out trance",
        "glitter trance",
        "progressive house",
        "progressive trance",
        "progressive uplifting trance",
        "sky room",
        "uplifting trance",
    },
    "psytrance": {
        "full on",
        "goa trance",
        "progressive psytrance",
        "psychedelic trance",
    },
    "techno": {
        "acid techno",
        "dub techno",
        "re:techno",
        "schranz",
    },
    "minimal techno": {
        "minimal melodic techno",
    },
    "industrial": {
        "aggrotech",
        "ebm",
        "electro-industrial",
        "grave wave",
        "minimal wave",
        "neue deutsche harte",
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
        "folk-pop",
        "gauze pop",
        "gospel",
        "idol pop",
        "idol",
        "indie anthem-folk",
        "italo dance",
        "karneval",
        "lds",
        "levenslied",
        "lilith",
        "louvor",
        "lovers rock",
        "mandopop",
        "mellow gold",
        "neo-singer-songwriter",
        "new romantic",
        "new wave pop",
        "poptimism",
        "post-teen pop",
        "praise",
        "rockism",
        "shimmer pop",
        "soft rock",
        "stomp pop",
        "synthpop",
        "teen pop",
        "viral pop",
        "worship",
        "wrestling",
    },
    "ccm": {
        "-alternative ccm",
    },
    "nu metal": {
        "alternative metal",
        "alternative pop rock",
        "alternative rock",
        "funk metal",
        "heavy alternative",
        "neo metal",
        "post-doom metal",
        "post-metal",
        "post-post-hardcore",
        "post-screamo",
        "progressive alternative",
        "progressive post-hardcore",
        "rap metal",
        "rap metalcore",
        "rap rock",
    },
    "screamo": {
        "screamocore",
    },
    "metal rock": {
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
        "jazz metal",
        "metal guitar",
        "metal",
        "neo classical metal",
        "neo-trad metal",
        "nwobhm",
        "nwothm",
        "power metal",
        "progressive metal",
    },
    "metal": {
        "-alternative metal",
        "speed metal",
        "stoner metal",
        "symphonic metal",
    },
    "pop rock": {
        "-alternative pop rock",
    },
    "rock": {
        "-alternative rock",
        "alternative",
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
        "indorock",
        "jangle pop",
        "jangle rock",
        "lift kit",
        "madchester",
        "mod revival",
        "neo-industrial rock",
        "new wave",
        "ostrock",
        "piano rock",
        "post-hardcore",
        "power blues-rock",
        "power pop",
        "protopunk",
        "psychedelic rock",
        "psychobilly",
        "pub rock",
        "punk blues",
        "rock 'n roll",
        "rock catala",
        "rock en espanol",
        "rock gaucho",
        "rock-and-roll",
        "roots rock",
        "simple rock",
        "sleaze rock",
        "space rock",
        "stoner rock",
        "surf music",
        "swamp pop",
        "symphonic rock",
        "trash rock",
    },
    "alternative rock": {
        "-australian alternative rock",
    },
    "stoner rock": {
        "-hard stoner rock",
    },
    "glam": {"-hard glam"},
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
        "instrumental post rock",
        "kraut rock",
        "math pop",
        "neo mellow",
        "neo-progressive",
        "neo-psychedelic",
        "new rave",
        "noise punk",
        "noise rock",
        "nu gaze",
        "permanent wave",
        "psych",
        "psych-rock",
        "psychedelic blues-rock",
        "psychedelic",
        "uplift",
        "wrock",
    },
    "hardcore rock": {
        "chaotic hardcore",
        "electronicore",
        "sludge metal",
    },
    "black metal": {
        "black death",
        "black sludge",
        "black thrash",
        "chaotic black metal",
        "cryptic black metal",
        "depressive black metal",
        "doom metal",
        "raw black metal",
        "symphonic black metal",
        "unblack metal",
        "usbm",
    },
    "post black metal": {
        "blackgaze",
        "pagan black metal",
        "psychedelic doom",
    },
    "death metal": {
        "brutal death metal",
        "brutal deathcore",
        "charred death",
        "death core",
        "deathgrind",
        "grindcore",
        "grisly death metal",
        "necrogrind",
        "slam death metal",
        "technical brutal death metal",
        "technical death metal",
    },
    "ska": {
        "euroska",
        "punk ska",
        "reggae rock",
        "ska revival",
    },
    "indie": {
        "alternative ccm",
        "austindie",
        "chillwave",
        "indie folk",
        "indiecoustica",
        "indietronica",
        "irish indie",
        "noise pop",
        "ok indie",
        "pixie",
        "popgaze",
        "shiver pop",
        "slow core",
        "triangle indie",
        "twee indie pop",
        "twee pop",
        "twin cities indie",
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
        "neo honky tonk",
        "neo-traditional country",
        "new americana",
        "outlaw country",
        "piedmont blues",
        "progressive bluegrass",
        "string band",
        "tejano",
        "vintage country folk",
    },
    "blues": {
        "acoustic blues",
        "bluegrass",
        "blues-rock guitar",
        "blues-rock",
        "country blues",
        "delta blues",
        "electric blues",
        "gospel blues",
        "harmonica blues",
        "jazz blues",
        "piano blues",
        "swamp blues",
    },
    "punk rock": {
        "anarcho-punk",
        "crack rock steady",
        "crust punk",
        "fast melodic punk",
        "horror punk",
        "oi",
        "orgcore",
        "poppunk",
        "power-pop punk",
        "punk",
        "riot grrrl",
        "ska punk",
        "skate punk",
        "skinhead oi",
        "slash punk",
        "straight edge",
        "street punk",
        "vegan straight edge",
    },
    "post punk rock": {
        "dance punk rock",
        "electropunk",
        "no wave",
        "post-punk",
        "screamo punk",
        "vaporwave",
        "wave",
    },
    "emo rock": {
        "emo",
    },
    "disco": {
        "italo disco",
        "post-disco",
        "post-disco soul",
    },
    "andean folk": {
        "andean",
    },
    "emo": {
        "anthem emo",
        "emo punk",
    },
    "tropical": {
        "azonto",
        "azontobeats",
        "banda",
        "choro",
        "cubaton",
        "hiplife",
        "kizomba",
        "kuduro",
        "makossa",
        "mande pop",
        "merengue urbano",
        "merengue",
        "orquesta tropical",
        "punta",
        "reggaeton flow",
        "reggaeton",
        "sega",
        "soukous",
        "zouk riddim",
        "zouk",
    },
    "electro tropical": {
        "electro bailando",
        "electro latino",
    },
    "bangla folk": {
        "bangla",
    },
    "barnemusikk": {
        "barnmusik",
    },
    "baroque": {
        "baroque ensemble",
    },
    "jazz": {
        "bebop",
        "big band",
        "cool jazz",
        "dixieland",
        "doo-wop",
        "free jazz",
        "gypsy jazz",
        "hard bop",
        "highlife",
        "jazz bass",
        "jazz brass",
        "jazz composition",
        "jazz fusion",
        "jazz guitar",
        "jazz orchestra",
        "jazz orkester",
        "jazz piano",
        "jazz trio",
        "post-bop",
        "rhythm and boogie",
        "soul jazz",
        "stride",
        "swing",
    },
    "bop": {"-hard bop"},
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
        "nueva cancion",
        "timba",
        "trova",
    },
    "bossa nova": {
        "bossa nova jazz",
    },
    "brazilian folk": {
        "brega",
        "forro",
        "mpb",
        "musica nativista",
        "sertanejo tradicional",
        "sertanejo universitario",
        "sertanejo",
        "velha guarda",
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
        "zapstep",
    },
    "breaks": {
        "nu skool breaks",
    },
    "liquid": {
        "liquid bass",
        "liquid funk",
    },
    "flamenco": {
        "cante flamenco",
    },
    "indian folk": {
        "carnatic",
        "indian classical",
        "kirtan",
    },
    "scottish folk": {
        "ceilidh",
        "traditional scottish folk",
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
        "rio de la plata",
    },
    "chanson": {
        "chanson quebecois",
    },
    "synth instrumental": {
        "c64",
        "c86",
        "chiptune",
    },
    "orchestral": {
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
        "deep orchestral",
        "early music",
        "harp",
        "harpsichord",
        "modern classical",
        "serialism",
        "string quartet",
        "wind ensemble",
    },
    "piano instrumental": {
        "piano",
        "classify",
    },
    "marching": {
        "blaskapelle",
        "brass band",
        "brass ensemble",
        "brass",
        "college marching band",
        "marching band",
        "military band",
    },
    "dominican folk": {
        "cornetas y tambores",
    },
    "colombian folk": {
        "cumbia funk",
        "cumbia pop",
        "cumbia sonidera",
        "cumbia villera",
        "cumbia",
        "nu-cumbia",
        "porro",
        "vallenato",
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
        "reggae fusion",
        "roots reggae",
        "skinhead reggae",
    },
    "mexican folk": {
        "duranguense",
        "grupera",
        "mariachi",
        "mexican",
        "norteno",
        "ranchera",
        "son",
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
        "volksmusik",
    },
    "spanish folk": {
        "galego",
        "spanish classical",
        "villancicos",
    },
    "arab folk": {
        "ghazal",
        "rai",
    },
    "hawaiian folk": {
        "hawaiian",
    },
    "hindustani folk": {
        "hindustani classical",
        "qawwali",
    },
    "finnish folk": {
        "iskelma",
        "yoik",
    },
    "funk": {
        "jazz funk",
        "p funk",
    },
    "irish folk": {
        "jig and reel",
    },
    "israeli folk": {
        "judaica",
        "klezmer",
    },
    "croatian folk": {
        "klapa",
    },
    "haitian folk": {
        "kompa",
    },
    "greek folk": {"laiko", "rebetiko"},
    "lithuanian folk": {
        "lithumania",
    },
    "thai folk": {
        "luk thung",
    },
    "maghreb folk": {
        "maghreb",
    },
    "hungarian folk": {
        "magyar",
    },
    "metalcore": {
        "melodic metalcore",
    },
    "cabo verde folk": {
        "morna",
    },
    "american folk": {
        "native american",
    },
    "soul": {
        "neo soul",
        "neo soul-jazz",
        "soul blues",
        "urban contemporary",
    },
    "r&b": {
        "new jack smooth",
        "new jack swing",
        "quiet storm",
        "slow game",
        "smooth urban r&b",
        "pop r&b",
    },
    "rockabilly": {
        "neo-rockabilly",
    },
    "nepali folk": {
        "nepali",
    },
    "neurofunk": {
        "neurostep",
    },
    "new age": {
        "new age piano",
        "nu age",
    },
    "phillipines folk": {
        "opm",
        "pinoy alternative",
    },
    "samba": {
        "pagode",
        "samba-enredo",
    },
    "grunge": {
        "post-grunge",
    },
    "noise": {
        "power electronics",
        "power noise",
    },
    "shoegaze": {"psych gaze", "voidgaze"},
    "quebec folk": {
        "quebecois",
    },
    "jungle": {
        "ragga jungle",
    },
    "opera": {
        "romantic",
    },
    "salsa": {
        "salsa international",
    },
    "angola folk": {
        "semba",
    },
    "british folk": {
        "shanty",
        "skiffle",
        "traditional british folk",
    },
    "sri lanka folk": {
        "sinhala",
    },
    "trinidad folk": {
        "calypso",
        "soca",
    },
    "folk": {
        "deep neofolk",
        "folkmusik",
        "neofolk",
        "stomp and flutter",
        "stomp and holler",
        "stomp and whittle",
        "string folk",
        "traditional folk",
        "traditional",
        "world",
    },
    "steelpan instrumental": {
        "steelpan",
    },
    "accordeon instrumental": {
        "accordeon",
    },
    "fingerstyle instrumental": {
        "fingerstyle",
    },
    "scratch instrumental": {
        "scratch",
    },
    "guitar instrumental": {
        "classical guitar",
    },
    "mallet instrumental": {
        "mallet",
    },
    "lounge": {
        "sunset lounge",
    },
    "mediterranean folk": {
        "mizrahi",
    },
    "liturgical": {
        "byzantine",
        "monastic",
        "renaissance",
    },
    "pipe band instrumental": {
        "pipe band",
    },
    "tanzanian folk": {
        "tanzlmusi",
    },
    "thrash metal": {
        "thrash core",
        "thrash-groove metal",
    },
    "tibetan folk": {
        "tibetan",
    },
    "turkish folk": {
        "turkish classical",
    },
    "ukulele instrumental": {
        "ukulele",
    },
    "violin instrumental": {
        "violin",
    },
    "western": {
        "western swing",
    },
    "french folk": {
        "ye ye",
    },
    "austrian folk": {
        "zillertal",
    },
    "zimbabwe folk": {
        "zim",
        "zolo",
    },
    "cote d'ivoire folk": {
        "zouglou",
    },
    "louisiana folk": {
        "zydeco",
    },
    "islamic folk": {
        "islamic recitation",
    },
    "romanian folk": {
        "manele",
    },
    "percussion instrumental": {
        "percussion",
    },
    "throat singing instrumental": {
        "throat singing",
    },
    "classical": {
        "-contemporary classical",
        "-indian classical",
        "-modern classical",
        "-spanish classical",
        "-turkish classical",
        "neoclassical",
    },
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
    "indie": False,
    "lounge": False,
    "melodic": False,
    "modern": False,
    "old school": False,
    "pop": False,
    "regional": False,
    "retro": False,
    "traditional": False,
    "underground": False,
    "vapor": False,
    "vintage": False,
    "vocal": False,
    "world": False,
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
    "c-",
    "canadian",
    "catalan",
    "caucasian",
    "celtic",
    "chicago",
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
    "detroit",
    "dominican",
    "dutch",
    "east coast",
    "estonian",
    "ethiopian",
    "faroese",
    "finnish",
    "french",
    "german",
    "greek",
    "hebrew",
    "hungarian",
    "icelandic",
    "indian",
    "indonesian",
    "irish",
    "israeli",
    "italian",
    "j-",
    "japanese",
    "k-",
    "kc",
    "kiwi",
    "korean",
    "kurdish",
    "la",
    "latin",
    "latvian",
    "leeds",
    "louisiana",
    "louisville",
    "malaysian",
    "memphis",
    "mexican",
    "michigan",
    "new orleans",
    "nordic",
    "northern",
    "norwegian",
    "nz",
    "pakistani",
    "persian",
    "perth",
    "peruvian",
    "polish",
    "polynesian",
    "portland",
    "portuguese",
    "puerto rican",
    "romanian",
    "russian",
    "rva",
    "scottish",
    "seattle",
    "sheffield",
    "singaporean",
    "slavic",
    "slc",
    "slovak",
    "slovenian",
    "southern",
    "spanish",
    "stl",
    "suomi",
    "swedish",
    "swiss",
    "taiwanese",
    "texas",
    "thai",
    "turkish",
    "uk",
    "ukrainian",
    "vancouver",
    "vegas",
    "venezuelan",
    "vienna",
    "vietnamese",
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
        if genre not in merged and override_mappings:
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
            if mapping_key == key:
                print(
                    f"<{key}> duplicates as key and value, ignoring",
                    file=sys.stderr,
                )
                continue
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
    print(f"{pprint.pp(merged_mapping, sort_dicts=True, )}")
    inverse_mapping = {
        specific_name: common_name
        for common_name, specific_names in merged_mapping.items()
        for specific_name in specific_names
    }

    def mapper(genre_name: str) -> str:
        return inverse_mapping.get(genre_name, genre_name)

    (
        genre_dataset.with_columns(
            pl.col(GENRE_COLUMN_NAME)
            .map_elements(mapper, pl.String)
            .alias(f"{GENRE_COLUMN_NAME}_mapped")
        )
        .with_columns(pl.col(f"{GENRE_COLUMN_NAME}_mapped").alias(GENRE_COLUMN_NAME))
        .filter(pl.col(GENRE_COLUMN_NAME).is_in(list(MIXED_GENRES)).not_())
        # .filter(
        #     pl.col(GENRE_COLUMN_NAME).is_in(list(merged_mapping.keys())).not_()
        # )
        .select(pl.col(ID_COLUMN_NAME), pl.col(GENRE_COLUMN_NAME))
        .group_by(pl.col(GENRE_COLUMN_NAME))
        .agg(pl.col(f"{ID_COLUMN_NAME}"))
        .with_columns(pl.col(ID_COLUMN_NAME).list.join(";"))
        .collect()
        .sort(by=pl.col(GENRE_COLUMN_NAME))
        .write_csv(grouped_by_genre_path)
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
