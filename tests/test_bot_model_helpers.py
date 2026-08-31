import asyncio
import pathlib

import config
from bot_model_helpers import format_model_response


def make_model(**overrides):
    params = dict(
        model_id=7178,
        pickle_file_path=pathlib.Path("/tmp/model.pkl"),
        liked_tracks_count=1376,
        disliked_tracks_count=603,
        outliers_removed_liked=12,
        outliers_removed_disliked=5,
        include_liked_tp=0.8,
        include_liked_tn=0.85,
        include_liked_fp=0.15,
        include_liked_fn=0.2,
        exclude_disliked_tp=0.9,
        exclude_disliked_tn=0.32,
        exclude_disliked_fp=0.68,
        exclude_disliked_fn=0.1,
    )
    params.update(overrides)
    return config.Model(**params)


def run_format_model_response(items):
    return asyncio.run(
        format_model_response(
            items,
            subscription_names={7178: ["chan_a", "chan_b"]},
            offset_stack=[0],
            previous_offset=None,
            next_offset=None,
        )
    )


def test_format_model_response_line_matches_train_stats():
    text, buttons, _ = run_format_model_response([make_model()])

    assert text == (
        "* [chan_a,chan_b] model `7178`: "
        "liked: 1376 (outliers removed: 12), "
        "disliked: 603 (outliers removed: 5), "
        "include_liked: tp=0.80 tn=0.85 fp=0.15 fn=0.20, "
        "exclude_disliked: tp=0.90 tn=0.32 fp=0.68 fn=0.10"
    )
    assert buttons == []


def test_format_model_response_one_line_per_model():
    text, _, _ = run_format_model_response(
        [make_model(model_id=1), make_model(model_id=2)]
    )

    lines = text.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("* [] model `1`: liked: ")
    assert lines[1].startswith("* [] model `2`: liked: ")


def test_format_model_response_empty_list():
    text, buttons, _ = run_format_model_response([])

    assert text == "No items to show"
    assert buttons == []
