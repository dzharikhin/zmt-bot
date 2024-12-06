import datetime
import json
import pathlib
import sys
import time
import traceback

import requests
from lxml.etree import ParseError
from path_dict import PathDict
from pyquery import PyQuery

from genre_dataset import Dataset, Track
from spotify import SpotifySession


def get_genre_dataset(
    *,
    csv_path: pathlib.Path = pathlib.Path("csv/songs.csv"),
    snippet_store_path: pathlib.Path = pathlib.Path("snippets"),
    error_limit: int = 1,
    error_timeout_sec: float = 5.0,
    https_proxy: str | None = None,
    rpm_limit: float | None = None,
    force: bool = False,
):
    dataset = Dataset(csv_path)
    snippet_store_path.mkdir(parents=True, exist_ok=True)
    processed_in_try_tracks = -1
    errors = []
    while processed_in_try_tracks != 0 and (
        error_limit <= 0 or len(errors) < error_limit
    ):
        iteration_start = datetime.datetime.now()
        try:
            processed_in_try_tracks = _download_snippets(
                dataset,
                snippet_store_path,
                https_proxy=https_proxy,
                rpm_limit=rpm_limit,
                force=force,
            )
            print(
                f"Processed {processed_in_try_tracks} in iteration of {(datetime.datetime.now() - iteration_start).total_seconds()} seconds long. Downloaded data stats by {dataset.stats().to_string()}"
            )
        except requests.RequestException as e:
            errors.append(e)
            print(
                f"{len(errors)} after {(datetime.datetime.now() - iteration_start).total_seconds()} seconds iteration, sleeping for {error_timeout_sec} sec. Downloaded data stats by {dataset.stats().to_string()}"
            )
            time.sleep(error_timeout_sec)
        finally:
            dataset.save()
    for e in errors:
        for line in traceback.format_exception(None, e, e.__traceback__):
            print(line, file=sys.stderr)


def _download_snippets(
    dataset: Dataset,
    snippet_store_path: pathlib.Path,
    *,
    https_proxy: str | None,
    rpm_limit: float | None,
    force: bool,
) -> int:
    processed_tracks = 0
    with SpotifySession(
        "https://api.spotify.com/v1",
        https_proxy=https_proxy,
        rpm_limit=rpm_limit,
        env_var_name_client_id="CLIENT_ID",
        env_var_name_client_secret="CLIENT_SECRET",
    ) as session:
        track: Track
        for track in dataset.dataframe.itertuples():
            if Dataset.has_snippet(track) and not force:
                print(
                    f"for track {track.spotify_id} snippet is already downloaded, skipping"
                )
                continue
            try:
                preview_link = None
                track_json = session.get(f"/tracks/{track.spotify_id}").json()
                # preview_link = track_json.get('preview_url')
                if not preview_link:
                    doc = PyQuery(
                        session.get(
                            f"https://open.spotify.com/embed/track/{track.spotify_id}"
                        ).content
                    )
                    data = PathDict(
                        json.loads(doc("script:contains(audioPreview)").text())
                    )
                    preview_link = data[
                        "props",
                        "pageProps",
                        "state",
                        "data",
                        "entity",
                        "audioPreview",
                        "url",
                    ]

                if preview_link:
                    _download_snippet(dataset, snippet_store_path, track, preview_link)
                else:
                    print(
                        f"marking as failed track {track.spotify_id} - preview is not found",
                        file=sys.stderr,
                    )
                    dataset.mark_snippet_failed(track.spotify_id)
                processed_tracks += 1
            except (ParseError, KeyError, ValueError) as e:
                print(
                    f"marking as failed track {track.spotify_id}, exception: {e}",
                    file=sys.stderr,
                )
                dataset.mark_snippet_failed(track.spotify_id)
                processed_tracks += 1
    return processed_tracks


def _download_snippet(
    dataset: Dataset, store_path: pathlib.Path, track: Track, preview_link: str
):
    snippet = store_path.joinpath(f"{track.spotify_id}.mp3")
    with snippet.open(mode="wb") as stream:
        r = requests.get(preview_link, allow_redirects=True)
        stream.write(r.content)
        dataset.add_snippet_path(track.spotify_id, snippet)


if __name__ == "__main__":
    get_genre_dataset()
