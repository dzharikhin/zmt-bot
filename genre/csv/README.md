- `songs.csv` - raw dataset from https://github.com/trebi/music-genres-dataset
- `songs-prepared.csv` - deduplicated, csv format-fixed data, for which sample was successfully downloaded(see tags `data-genre-*`). Built with [download_genre_snippets.py](../download_genre_snippets.py)
- `audio_features_dataset.csv`- audio features extracted for downloaded samples. Built with [prepare_features_dataset.py](../prepare_features_dataset.py)
- `audio_features_dataset-processing_failed.csv`- sample ids failed to extract features from. Built with [prepare_features_dataset.py](../prepare_features_dataset.py)
- `songs-prepared-grouped_by_genre.csv` - `songs-prepared.csv` grouped by genre_name to be able to listen examples of genre. Built with [create_genre_mapping.py](../create_genre_mapping.py)
  to play audio you can use script:
  ```python
  #!/usr/bin/env python3
  import pathlib
  import subprocess
  import sys
  
  if __name__ == "__main__":
    genre_file_path = pathlib.Path(sys.argv[1])
    snippets_path = pathlib.Path(sys.argv[2])
    genre_name = " ".join(sys.argv[3:])
    print(f"{genre_file_path} {snippets_path} {genre_name}")
    lines = [line.split(",", 1) for line in genre_file_path.read_text().splitlines()[1:]]
    snippet_ids_for_genre = sum((line[1].split(";") for line in lines if line[0] == genre_name), [])
    files_to_play = [str(snippets_path.joinpath(f"{snippet_id}.mp3")) for snippet_id in snippet_ids_for_genre]
    # here you add command call to play multiple files like subprocess.run()
  ```
- `clustered_genres.csv` - `songs.csv.prepared` ML-clustered then manually corrected. Built with [create_genre_clusterization.py](../create_genre_clusterization.py)
- `songs-processed.csv` - `songs.csv.prepared` enriched with clustered genre column