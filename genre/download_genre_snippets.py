import datetime
import json
import pathlib
import sys
import time
import traceback

import requests
from lxml.etree import LxmlError
from path_dict import PathDict
from pyquery import PyQuery

from genre.spotify import SpotifySession
from genre_dataset import Dataset, Track


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
        failed_track_ids = []
        iteration_start = datetime.datetime.now()
        try:
            processed_in_try_tracks = _download_snippets(
                dataset,
                snippet_store_path,
                failed_track_ids,
                https_proxy=https_proxy,
                rpm_limit=rpm_limit,
                force=force,
            )
            print(
                f"Processed {processed_in_try_tracks} in iteration of {(datetime.datetime.now() - iteration_start).total_seconds()} seconds long. Dataset stats: {dataset.dataframe.shape}"
            )
        except requests.RequestException as e:
            errors.append(e)
            print(
                f"{len(errors)} errors after {(datetime.datetime.now() - iteration_start).total_seconds()} seconds iteration, sleeping for {error_timeout_sec} sec.  Dataset stats: {dataset.dataframe.shape}"
            )
            time.sleep(error_timeout_sec)
        finally:
            dataset.remove_tracks(failed_track_ids)
            dataset.save()

    print(
        f"Finished execution with {len(errors)} errors. Dataset stats by {dataset.dataframe.shape}"
    )
    for e in errors:
        for line in traceback.format_exception(None, e, e.__traceback__):
            print(line, file=sys.stderr)


def _download_snippets(
    dataset: Dataset,
    snippet_store_path: pathlib.Path,
    failed_track_ids: list[str],
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
        for track in dataset.dataframe.itertuples(name="Track"):
            if not force:
                snippet_path = _get_snippet_path(snippet_store_path, track)
                if snippet_path.exists():
                    print(
                        f"for track {track.Index} snippet is already downloaded, skipping"
                    )
                    continue
            try:
                preview_link = None
                # track_json = session.get(f"/tracks/{track.Index}").json()
                # preview_link = track_json.get('preview_url')
                if not preview_link:
                    doc = PyQuery(
                        session.get(
                            f"https://open.spotify.com/embed/track/{track.Index}"
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
                    _download_snippet(snippet_store_path, track, preview_link)
                else:
                    print(
                        f"preview for track {track.Index} is not found, marking to remove from dataset",
                        file=sys.stderr,
                    )
                    failed_track_ids.append(track.Index)
                processed_tracks += 1
            except (LxmlError, KeyError, ValueError) as e:
                print(
                    f"preview for track {track.Index} failed with exception: {type(e)}, marking to remove from dataset",
                    file=sys.stderr,
                )
                failed_track_ids.append(track.Index)
                processed_tracks += 1
    return processed_tracks


def _download_snippet(store_path: pathlib.Path, track: Track, preview_link: str):
    snippet_path = _get_snippet_path(store_path, track)
    with snippet_path.open(mode="wb") as stream:
        r = requests.get(preview_link, allow_redirects=True)
        stream.write(r.content)


def _get_snippet_path(store_path: pathlib.Path, track: Track) -> pathlib.Path:
    return store_path.joinpath(f"{track.Index}.mp3")


if __name__ == "__main__":
    get_genre_dataset()