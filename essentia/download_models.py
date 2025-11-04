import pathlib

import requests

from audio.models import ml_model_links, get_model_name

models_base_path = pathlib.Path("models")


def main():
    for url_template in ml_model_links:
        metadata_path = models_base_path.joinpath(
            f"{get_model_name(url_template)}.json"
        )
        weights_path = models_base_path.joinpath(f"{get_model_name(url_template)}.pb")
        with requests.get(
            f"{url_template}.json", allow_redirects=True
        ) as resp, metadata_path.open(mode="wt") as dest:
            dest.write(resp.text)
        print(f"downloaded {url_template} meta")
        with weights_path.open(mode="wb") as dest:
            resp = requests.get(f"{url_template}.pb", allow_redirects=True, stream=True)
            for chunk in resp.iter_content(chunk_size=1024):
                dest.write(chunk)
        print(f"downloaded {url_template} weights")


if __name__ == "__main__":
    main()
