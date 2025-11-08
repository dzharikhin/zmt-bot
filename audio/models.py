import functools
import json
import logging
import pathlib
import typing
import essentia.standard as es

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_path_to_root = pathlib.Path(__file__).parent.parent

ml_model_links = (
    "https://essentia.upf.edu/models/classification-heads/danceability/danceability-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2",
    "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2",
    "https://essentia.upf.edu/models/classification-heads/engagement/engagement_regression-discogs-effnet-1",
    "https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/moods_mirex/moods_mirex-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/muse/muse-msd-musicnn-2",
    "https://essentia.upf.edu/models/classification-heads/nsynth_acoustic_electronic/nsynth_acoustic_electronic-discogs-effnet-1",
    "https://essentia.upf.edu/models/classification-heads/nsynth_bright_dark/nsynth_bright_dark-discogs-effnet-1",
    "https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1",
    "https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-msd-musicnn-1",
    "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1",
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1",
    "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1",
)


def get_model_name(url):
    return url.split("/")[-1]


@functools.cache
def get_or_create_embeddings_model(embedding_params: tuple[str, str, str]):
    cls, model_name, output = embedding_params
    return getattr(es, cls)(
        graphFilename=f"{_path_to_root}/essentia/models/{model_name}.pb",
        output=output,
    )


def get_meta_and_embedding_model(
    model_name: str,
) -> tuple[typing.Optional[dict], typing.Optional[typing.Any]]:
    model_params = json.loads(
        pathlib.Path(f"{_path_to_root}/essentia/models/{model_name}.json").read_text()
    )
    inference = model_params["inference"]
    if "embedding_model" not in inference:
        return None, None

    embedding_model_data = inference["embedding_model"]
    embedding_model_name = embedding_model_data["model_name"]
    embedding_model_meta = json.loads(
        pathlib.Path(
            f"{_path_to_root}/essentia/models/{embedding_model_name}.json"
        ).read_text()
    )
    embedding_model_output = [
        output["name"]
        for output in embedding_model_meta["schema"]["outputs"]
        if output["output_purpose"] == "embeddings"
    ][0]
    embedding_model = get_or_create_embeddings_model(
        (
            embedding_model_data["algorithm"],
            embedding_model_name,
            embedding_model_output,
        )
    )
    return model_params, embedding_model


@functools.cache
def get_or_create_model(model_name: str, model_params: tuple[str, str, str]):
    print(f"creating model {model_name}: {model_params}")
    cls, input, output = model_params

    return getattr(es, cls)(
        graphFilename=f"{_path_to_root}/essentia/models/{model_name}.pb",
        input=input,
        output=output,
    )


def get_classes_for_model(model_name: str):
    classes_ = json.loads(
        pathlib.Path(f"{_path_to_root}/essentia/models/{model_name}.json").read_text()
    )["classes"]
    return classes_ if len(classes_) > 2 else classes_[:1]


def get_model_params(model_metadata: dict):
    model_class = model_metadata["inference"]["algorithm"]
    model_input = [input["name"] for input in model_metadata["schema"]["inputs"]][0]
    model_output = [
        output["name"]
        for output in model_metadata["schema"]["outputs"]
        if output["output_purpose"] == "predictions"
    ][0]
    return (
        model_class,
        model_input,
        model_output,
    )


# warmup models for effective process fork
for model_link in ml_model_links:
    model_name = get_model_name(model_link)
    logger.info(f"Warmed up {model_name}")
    model_params, embedding_model = get_meta_and_embedding_model(model_name)
    if not embedding_model:
        continue
    get_or_create_model(
        model_name,
        get_model_params(model_params),
    )
