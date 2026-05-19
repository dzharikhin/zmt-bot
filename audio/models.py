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
