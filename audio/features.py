import dataclasses
import functools
import pathlib
import re
import tempfile
import textwrap
import time
import typing
from typing import Callable, Literal

import essentia
import essentia.standard as es
import numpy
import yaml

from audio.models import (
    get_model_name,
    ml_model_links,
    get_meta_and_embedding_model,
    get_or_create_model,
    get_classes_for_model,
    get_model_params,
)

essentia.EssentiaLogger().warningActive = False


# generated with audio.features.__generate_dto_class
@dataclasses.dataclass
class AudioFeatures:
    lowlevel______average_loudness: float
    lowlevel______barkbands_crest______max: float
    lowlevel______barkbands_crest______mean: float
    lowlevel______barkbands_crest______min: float
    lowlevel______barkbands_crest______var: float
    lowlevel______barkbands_flatness_db______max: float
    lowlevel______barkbands_flatness_db______mean: float
    lowlevel______barkbands_flatness_db______min: float
    lowlevel______barkbands_flatness_db______var: float
    lowlevel______barkbands_kurtosis______max: float
    lowlevel______barkbands_kurtosis______mean: float
    lowlevel______barkbands_kurtosis______min: float
    lowlevel______barkbands_kurtosis______var: float
    lowlevel______barkbands_skewness______max: float
    lowlevel______barkbands_skewness______mean: float
    lowlevel______barkbands_skewness______min: float
    lowlevel______barkbands_skewness______var: float
    lowlevel______barkbands_spread______max: float
    lowlevel______barkbands_spread______mean: float
    lowlevel______barkbands_spread______min: float
    lowlevel______barkbands_spread______var: float
    lowlevel______dissonance______max: float
    lowlevel______dissonance______mean: float
    lowlevel______dissonance______min: float
    lowlevel______dissonance______var: float
    lowlevel______dynamic_complexity: float
    lowlevel______erbbands_crest______max: float
    lowlevel______erbbands_crest______mean: float
    lowlevel______erbbands_crest______min: float
    lowlevel______erbbands_crest______var: float
    lowlevel______erbbands_flatness_db______max: float
    lowlevel______erbbands_flatness_db______mean: float
    lowlevel______erbbands_flatness_db______min: float
    lowlevel______erbbands_flatness_db______var: float
    lowlevel______erbbands_kurtosis______max: float
    lowlevel______erbbands_kurtosis______mean: float
    lowlevel______erbbands_kurtosis______min: float
    lowlevel______erbbands_kurtosis______var: float
    lowlevel______erbbands_skewness______max: float
    lowlevel______erbbands_skewness______mean: float
    lowlevel______erbbands_skewness______min: float
    lowlevel______erbbands_skewness______var: float
    lowlevel______erbbands_spread______max: float
    lowlevel______erbbands_spread______mean: float
    lowlevel______erbbands_spread______min: float
    lowlevel______erbbands_spread______var: float
    lowlevel______hfc______max: float
    lowlevel______hfc______mean: float
    lowlevel______hfc______min: float
    lowlevel______hfc______var: float
    lowlevel______loudness_ebu128______integrated: float
    lowlevel______loudness_ebu128______loudness_range: float
    lowlevel______loudness_ebu128______momentary______max: float
    lowlevel______loudness_ebu128______momentary______mean: float
    lowlevel______loudness_ebu128______momentary______min: float
    lowlevel______loudness_ebu128______momentary______var: float
    lowlevel______loudness_ebu128______short_term______max: float
    lowlevel______loudness_ebu128______short_term______mean: float
    lowlevel______loudness_ebu128______short_term______min: float
    lowlevel______loudness_ebu128______short_term______var: float
    lowlevel______melbands_crest______max: float
    lowlevel______melbands_crest______mean: float
    lowlevel______melbands_crest______min: float
    lowlevel______melbands_crest______var: float
    lowlevel______melbands_flatness_db______max: float
    lowlevel______melbands_flatness_db______mean: float
    lowlevel______melbands_flatness_db______min: float
    lowlevel______melbands_flatness_db______var: float
    lowlevel______melbands_kurtosis______max: float
    lowlevel______melbands_kurtosis______mean: float
    lowlevel______melbands_kurtosis______min: float
    lowlevel______melbands_kurtosis______var: float
    lowlevel______melbands_skewness______max: float
    lowlevel______melbands_skewness______mean: float
    lowlevel______melbands_skewness______min: float
    lowlevel______melbands_skewness______var: float
    lowlevel______melbands_spread______max: float
    lowlevel______melbands_spread______mean: float
    lowlevel______melbands_spread______min: float
    lowlevel______melbands_spread______var: float
    lowlevel______pitch_salience______max: float
    lowlevel______pitch_salience______mean: float
    lowlevel______pitch_salience______min: float
    lowlevel______pitch_salience______var: float
    lowlevel______silence_rate_20dB______max: float
    lowlevel______silence_rate_20dB______mean: float
    lowlevel______silence_rate_20dB______min: float
    lowlevel______silence_rate_20dB______var: float
    lowlevel______silence_rate_30dB______max: float
    lowlevel______silence_rate_30dB______mean: float
    lowlevel______silence_rate_30dB______min: float
    lowlevel______silence_rate_30dB______var: float
    lowlevel______silence_rate_60dB______max: float
    lowlevel______silence_rate_60dB______mean: float
    lowlevel______silence_rate_60dB______min: float
    lowlevel______silence_rate_60dB______var: float
    lowlevel______spectral_centroid______max: float
    lowlevel______spectral_centroid______mean: float
    lowlevel______spectral_centroid______min: float
    lowlevel______spectral_centroid______var: float
    lowlevel______spectral_complexity______max: float
    lowlevel______spectral_complexity______mean: float
    lowlevel______spectral_complexity______min: float
    lowlevel______spectral_complexity______var: float
    lowlevel______spectral_decrease______max: float
    lowlevel______spectral_decrease______mean: float
    lowlevel______spectral_decrease______min: float
    lowlevel______spectral_decrease______var: float
    lowlevel______spectral_energy______max: float
    lowlevel______spectral_energy______mean: float
    lowlevel______spectral_energy______min: float
    lowlevel______spectral_energy______var: float
    lowlevel______spectral_energyband_high______max: float
    lowlevel______spectral_energyband_high______mean: float
    lowlevel______spectral_energyband_high______min: float
    lowlevel______spectral_energyband_high______var: float
    lowlevel______spectral_energyband_low______max: float
    lowlevel______spectral_energyband_low______mean: float
    lowlevel______spectral_energyband_low______min: float
    lowlevel______spectral_energyband_low______var: float
    lowlevel______spectral_energyband_middle_high______max: float
    lowlevel______spectral_energyband_middle_high______mean: float
    lowlevel______spectral_energyband_middle_high______min: float
    lowlevel______spectral_energyband_middle_high______var: float
    lowlevel______spectral_energyband_middle_low______max: float
    lowlevel______spectral_energyband_middle_low______mean: float
    lowlevel______spectral_energyband_middle_low______min: float
    lowlevel______spectral_energyband_middle_low______var: float
    lowlevel______spectral_entropy______max: float
    lowlevel______spectral_entropy______mean: float
    lowlevel______spectral_entropy______min: float
    lowlevel______spectral_entropy______var: float
    lowlevel______spectral_flux______max: float
    lowlevel______spectral_flux______mean: float
    lowlevel______spectral_flux______min: float
    lowlevel______spectral_flux______var: float
    lowlevel______spectral_kurtosis______max: float
    lowlevel______spectral_kurtosis______mean: float
    lowlevel______spectral_kurtosis______min: float
    lowlevel______spectral_kurtosis______var: float
    lowlevel______spectral_rms______max: float
    lowlevel______spectral_rms______mean: float
    lowlevel______spectral_rms______min: float
    lowlevel______spectral_rms______var: float
    lowlevel______spectral_rolloff______max: float
    lowlevel______spectral_rolloff______mean: float
    lowlevel______spectral_rolloff______min: float
    lowlevel______spectral_rolloff______var: float
    lowlevel______spectral_skewness______max: float
    lowlevel______spectral_skewness______mean: float
    lowlevel______spectral_skewness______min: float
    lowlevel______spectral_skewness______var: float
    lowlevel______spectral_spread______max: float
    lowlevel______spectral_spread______mean: float
    lowlevel______spectral_spread______min: float
    lowlevel______spectral_spread______var: float
    lowlevel______spectral_strongpeak______max: float
    lowlevel______spectral_strongpeak______mean: float
    lowlevel______spectral_strongpeak______min: float
    lowlevel______spectral_strongpeak______var: float
    lowlevel______zerocrossingrate______max: float
    lowlevel______zerocrossingrate______mean: float
    lowlevel______zerocrossingrate______min: float
    lowlevel______zerocrossingrate______var: float
    metadata______audio_properties______analysis______equal_loudness: float
    metadata______audio_properties______analysis______length: float
    metadata______audio_properties______analysis______sample_rate: float
    metadata______audio_properties______analysis______start_time: float
    metadata______audio_properties______bit_rate: float
    metadata______audio_properties______length: float
    metadata______audio_properties______lossless: float
    metadata______audio_properties______number_channels: float
    metadata______audio_properties______replay_gain: float
    metadata______audio_properties______sample_rate: float
    rhythm______beats_count: float
    rhythm______beats_loudness______dmean: float
    rhythm______beats_loudness______dmean2: float
    rhythm______beats_loudness______dvar: float
    rhythm______beats_loudness______dvar2: float
    rhythm______beats_loudness______max: float
    rhythm______beats_loudness______mean: float
    rhythm______beats_loudness______median: float
    rhythm______beats_loudness______min: float
    rhythm______beats_loudness______stdev: float
    rhythm______beats_loudness______var: float
    rhythm______bpm: float
    rhythm______bpm_histogram_first_peak_bpm: float
    rhythm______bpm_histogram_first_peak_weight: float
    rhythm______bpm_histogram_second_peak_bpm: float
    rhythm______bpm_histogram_second_peak_spread: float
    rhythm______bpm_histogram_second_peak_weight: float
    rhythm______danceability: float
    rhythm______onset_rate: float
    tonal______chords_changes_rate: float
    tonal______chords_number_rate: float
    tonal______chords_strength______dmean: float
    tonal______chords_strength______dmean2: float
    tonal______chords_strength______dvar: float
    tonal______chords_strength______dvar2: float
    tonal______chords_strength______max: float
    tonal______chords_strength______mean: float
    tonal______chords_strength______median: float
    tonal______chords_strength______min: float
    tonal______chords_strength______stdev: float
    tonal______chords_strength______var: float
    tonal______hpcp_crest______dmean: float
    tonal______hpcp_crest______dmean2: float
    tonal______hpcp_crest______dvar: float
    tonal______hpcp_crest______dvar2: float
    tonal______hpcp_crest______max: float
    tonal______hpcp_crest______mean: float
    tonal______hpcp_crest______median: float
    tonal______hpcp_crest______min: float
    tonal______hpcp_crest______stdev: float
    tonal______hpcp_crest______var: float
    tonal______hpcp_entropy______dmean: float
    tonal______hpcp_entropy______dmean2: float
    tonal______hpcp_entropy______dvar: float
    tonal______hpcp_entropy______dvar2: float
    tonal______hpcp_entropy______max: float
    tonal______hpcp_entropy______mean: float
    tonal______hpcp_entropy______median: float
    tonal______hpcp_entropy______min: float
    tonal______hpcp_entropy______stdev: float
    tonal______hpcp_entropy______var: float
    tonal______key_edma______strength: float
    tonal______key_krumhansl______strength: float
    tonal______key_temperley______strength: float
    tonal______tuning_diatonic_strength: float
    tonal______tuning_equal_tempered_deviation: float
    tonal______tuning_frequency: float
    tonal______tuning_nontempered_energy_ratio: float
    lowlevel______barkbands______max: numpy.ndarray[tuple[Literal[27]], numpy.float32]
    lowlevel______barkbands______mean: numpy.ndarray[tuple[Literal[27]], numpy.float32]
    lowlevel______barkbands______min: numpy.ndarray[tuple[Literal[27]], numpy.float32]
    lowlevel______barkbands______var: numpy.ndarray[tuple[Literal[27]], numpy.float32]
    lowlevel______erbbands______max: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______erbbands______mean: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______erbbands______min: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______erbbands______var: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______gfcc______mean: numpy.ndarray[tuple[Literal[13]], numpy.float32]
    lowlevel______melbands______max: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______melbands______mean: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______melbands______min: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______melbands______var: numpy.ndarray[tuple[Literal[40]], numpy.float32]
    lowlevel______melbands128______max: numpy.ndarray[
        tuple[Literal[128]], numpy.float32
    ]
    lowlevel______melbands128______mean: numpy.ndarray[
        tuple[Literal[128]], numpy.float32
    ]
    lowlevel______melbands128______min: numpy.ndarray[
        tuple[Literal[128]], numpy.float32
    ]
    lowlevel______melbands128______var: numpy.ndarray[
        tuple[Literal[128]], numpy.float32
    ]
    lowlevel______mfcc______mean: numpy.ndarray[tuple[Literal[13]], numpy.float32]
    lowlevel______spectral_contrast_coeffs______max: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_coeffs______mean: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_coeffs______min: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_coeffs______var: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_valleys______max: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_valleys______mean: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_valleys______min: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    lowlevel______spectral_contrast_valleys______var: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______dmean: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______dmean2: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______dvar: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______dvar2: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______max: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______mean: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______median: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______min: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______stdev: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    rhythm______beats_loudness_band_ratio______var: numpy.ndarray[
        tuple[Literal[6]], numpy.float32
    ]
    tonal______hpcp______dmean: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______dmean2: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______dvar: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______dvar2: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______max: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______mean: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______median: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______min: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______stdev: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______hpcp______var: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    rhythm______bpm_histogram: numpy.ndarray[tuple[Literal[250]], numpy.float32]
    tonal______chords_histogram: numpy.ndarray[tuple[Literal[24]], numpy.float32]
    tonal______thpcp: numpy.ndarray[tuple[Literal[36]], numpy.float32]
    tonal______chords_key: str
    tonal______chords_scale: str
    tonal______key_edma______key: str
    tonal______key_edma______scale: str
    tonal______key_krumhansl______key: str
    tonal______key_krumhansl______scale: str
    tonal______key_temperley______key: str
    tonal______key_temperley______scale: str
    danceability___msd___musicnn___1______danceable: numpy.float32
    engagement_regression___discogs___effnet___1______engagement: numpy.float32
    deam___msd___musicnn___2______valence: numpy.float32
    emomusic___msd___musicnn___2______valence: numpy.float32
    engagement_regression___discogs___effnet___1______engagement: numpy.float32
    mood_acoustic___msd___musicnn___1______acoustic: numpy.float32
    mood_aggressive___msd___musicnn___1______aggressive: numpy.float32
    mood_electronic___msd___musicnn___1______electronic: numpy.float32
    mood_happy___msd___musicnn___1______happy: numpy.float32
    mood_party___msd___musicnn___1______non_party: numpy.float32
    mood_relaxed___msd___musicnn___1______non_relaxed: numpy.float32
    mood_sad___msd___musicnn___1______non_sad: numpy.float32
    moods_mirex___msd___musicnn___1______passionate_rousing_confident_boisterous_rowdy: (
        numpy.float32
    )
    moods_mirex___msd___musicnn___1______rollicking_cheerful_fun_sweet_amiable_good_natured: (
        numpy.float32
    )
    moods_mirex___msd___musicnn___1______literate_poignant_wistful_bittersweet_autumnal_brooding: (
        numpy.float32
    )
    moods_mirex___msd___musicnn___1______humorous_silly_campy_quirky_whimsical_witty_wry: (
        numpy.float32
    )
    moods_mirex___msd___musicnn___1______aggressive_fiery_tense_anxious_intense_volatile_visceral: (
        numpy.float32
    )
    muse___msd___musicnn___2______valence: numpy.float32
    nsynth_acoustic_electronic___discogs___effnet___1______acoustic: numpy.float32
    nsynth_bright_dark___discogs___effnet___1______bright: numpy.float32
    timbre___discogs___effnet___1______bright: numpy.float32
    tonal_atonal___msd___musicnn___1______atonal: numpy.float32
    voice_instrumental___msd___musicnn___1______instrumental: numpy.float32


key_columns = [
    field.name
    for field in dataclasses.fields(AudioFeatures)
    if field.type == str and field.name.endswith("_key")
]

scale_columns = [
    field.name
    for field in dataclasses.fields(AudioFeatures)
    if field.type == str and field.name.endswith("_scale")
]


def extract_features_for_mp3(
    mp3_path: pathlib.Path,
    extractor_from_path,
) -> AudioFeatures:

    @functools.cache
    def get_or_create_audio_data(sample_rate: int):
        return es.MonoLoader(sampleRate=sample_rate, filename=str(mp3_path))()

    def get_audio_data(model_params: dict) -> numpy.ndarray[tuple[int], numpy.float32]:
        return get_or_create_audio_data(model_params["inference"]["sample_rate"])

    ml_features = functools.reduce(
        lambda a, b: a | b,
        (
            model_features
            for model_name in [get_model_name(link) for link in ml_model_links]
            if (model_features := _get_features_from_model(model_name, get_audio_data))
        ),
    )

    raw_features = extractor_from_path(mp3_path)
    features = raw_features | ml_features
    feature_mapping = {
        field.name: features[_build_key_from_property(field.name)]
        for field in dataclasses.fields(AudioFeatures)
    }
    return AudioFeatures(**feature_mapping)


def prepare_extractor() -> (
    Callable[[pathlib.Path], dict[str, str | float | int | list | numpy.ndarray]]
):
    with tempfile.TemporaryDirectory() as tmp:
        # extractor = es.Extractor(rhythm=False)
        # raw_features = extractor(get_audio_data({"inference": {"sample_rate": sr}}))
        # aggregationPool = es.PoolAggregator(
        #     defaultStats = [ "mean", "stdev" ],
        #     # exceptions={},
        # )(features)
        tmp_path = pathlib.Path(tmp)
        options_file = tmp_path.joinpath("options.yaml")
        options_file.write_text(yaml.dump(_music_extractor_profile))
        func = es.MusicExtractor(
            lowlevelStats=["mean", "var", "min", "max"], profile=str(options_file)
        )

    def extract_dict(audio_path: pathlib.Path):
        result = func(str(audio_path))[0]
        return {name: result[name] for name in result.descriptorNames()}

    return extract_dict


def _get_features_from_model(
    model_name: str,
    audio_parser: Callable[[dict], numpy.ndarray[tuple[int], numpy.float32]],
) -> typing.Optional[dict]:
    model_metadata, embedding_model = get_meta_and_embedding_model(model_name)
    if not embedding_model:
        return None

    embeddings = embedding_model(audio_parser(model_metadata))

    model = get_or_create_model(
        model_name,
        get_model_params(model_metadata),
    )

    activations = model(embeddings)
    classes_ = [
        _build_key_for_ml_class(model_name, c) for c in model_metadata["classes"]
    ]
    mean_value = activations.mean(axis=0)
    return (
        dict(zip(classes_, mean_value))
        if len(classes_) > 2
        else {classes_[0]: mean_value[0]}
    )


_music_extractor_profile = {
    "lowlevel": {
        "frameSize": 2048,
        "hopSize": 1024,
        "zeroPadding": 0,
        "silentFrames": "keep",
        "windowType": "blackmanharris62",
    },
}

_property_separator = "___"


def _build_property_from_key(key: str):
    return key.replace(".", _property_separator * 2).replace("-", _property_separator)


def _build_key_from_property(property_name: str):
    return property_name.replace(_property_separator * 2, ".").replace(
        _property_separator, "-"
    )


def _build_key_for_ml_class(model_name: str, ml_class: str):
    return f"{model_name}.{re.sub("[^A-z0-9_]+", "_", ml_class)}"


def __generate_dto_class(numpy_prefix: str):

    model_names = [
        m_name
        for link in ml_model_links
        if get_meta_and_embedding_model((m_name := get_model_name(link)))[1]
    ]
    ml_keys = [
        _build_key_for_ml_class(model_name, cls_)
        for model_name in model_names
        for cls_ in get_classes_for_model(model_name)
    ]

    excluded_extractor_keys = [
        "metadata.tags.album",
        "metadata.tags.albumartist",
        "metadata.tags.artist",
        "metadata.tags.copyright",
        "metadata.tags.date",
        "metadata.tags.encoding",
        "metadata.tags.genre",
        "metadata.tags.label",
        "metadata.tags.title",
        "metadata.tags.tracknumber",
        "metadata.audio_properties.md5_encoded",
        "metadata.tags.file_name",
        "metadata.version.essentia",
        "metadata.version.essentia_git_sha",
        "metadata.version.extractor",
        "rhythm.beats_position",
        "lowlevel.gfcc.cov",
        "lowlevel.gfcc.icov",
        "lowlevel.mfcc.cov",
        "lowlevel.mfcc.icov",
        "metadata.audio_properties.analysis.downmix",
        "metadata.audio_properties.codec",
    ]
    extractor = prepare_extractor()
    features = extractor(
        pathlib.Path("../data/118517468/liked/CQADAgAD0AIAAlbXeUr25_ycgx2WEgI.mp3")
    )

    if diff := (
        set(k for k in set(features.keys()) - set(excluded_extractor_keys))
        | set(ml_keys)
    ) - set(
        _build_key_from_property(field.name)
        for field in dataclasses.fields(AudioFeatures)
    ):
        print(f"diff size is: {len(diff)}")
        raise Exception(f"Fields {diff} are not processed, exiting")

    def print_type(v, t: type):
        if t == numpy.ndarray:
            return f"{numpy_prefix}.ndarray[tuple[{", ".join([f"Literal[{axis_size}]" for axis_size in v.shape])}], {numpy_prefix}.{v.dtype}]"
        else:
            return t.__name__

    print(
        textwrap.dedent(
            """\
        @dataclasses.dataclass
        class AudioFeatures:\
        """
        ),
        "\n".join(
            [
                f"    {_build_property_from_key(feature_name)}: {print_type(feature_value, type(feature_value))}"
                for feature_name, feature_value in features.items()
                if feature_name not in excluded_extractor_keys
            ]
            + [
                f"    {_build_property_from_key(ml_key)}: {numpy_prefix}.{print_type(None, numpy.float32)}"
                for ml_key in ml_keys
            ]
        ),
        sep="\n",
    )


scale_mapping = {
    "major": 1,
    "minor": 0,
}
keys = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
alias_keys = {
    "Db": keys["C#"],
    "Eb": keys["D#"],
    "Gb": keys["F#"],
    "Ab": keys["G#"],
    "Bb": keys["A#"],
    "Cb": keys["B"],
}
key_mapping = {
    k: {
        "sin": numpy.sin(2 * numpy.pi * v / len(keys)),
        "cos": numpy.cos(2 * numpy.pi * v / len(keys)),
    }
    for k, v in (keys | alias_keys).items()
}

if __name__ == "__main__":
    # __generate_dto_class("numpy")

    track = pathlib.Path("../data/118517468/liked/CQADAgAD0AIAAlbXeUr25_ycgx2WEgI.mp3")
    start = time.perf_counter()
    data = extract_features_for_mp3(track, prepare_extractor())
    first_attempt = time.perf_counter() - start
    start = time.perf_counter()
    data = extract_features_for_mp3(track, prepare_extractor())
    second_attempt = time.perf_counter() - start
    print(f"processed in: {first_attempt=} seconds, {second_attempt=} seconds")
    print(f"{data=}")
