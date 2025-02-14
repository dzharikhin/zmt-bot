> Archive with files is attached to release must be unpacked to the current folder

Train data. Built with [train_guess_genre_model.py](../train_guess_genre_model.py)
- `songs.csv` - raw dataset from https://github.com/trebi/music-genres-dataset
- `songs-downloaded.csv` - if you need raw data - consider this as your source. deduplicated, csv format-fixed data, for which sample was successfully downloaded(see tags `data-genre-*`and releases). Built with [download_genre_snippets.py](../download_genre_snippets.py)
- `audio_features_dataset.csv`- audio features extracted for downloaded samples. Built with [prepare_features_dataset.py](../prepare_features_dataset.py)
- `audio_features_dataset-processing_failed.csv`- sample ids failed to extract features from. Built with [prepare_features_dataset.py](../prepare_features_dataset.py)
- `songs-grouped_by_genre.csv` - `songs-downloaded.csv` applied genre aggregation then grouped by mapped genre name to be able to listen examples of genre mapping. Built with [create_genre_mapping.py](../create_genre_mapping.py)
  to play audio you can use script:
  ```python
  #!/usr/bin/env python3
  import pathlib
  import subprocess
  import sys
  
  if __name__ == "__main__":
      print("going to build vlc playlist")
      genre_file_path = pathlib.Path(sys.argv[1])
      snippet_ids_column_index = int(sys.argv[2])
      snippets_path = pathlib.Path(sys.argv[3])
      genre_name = " ".join(sys.argv[4:])
      print(f"{genre_file_path} {snippets_path} {genre_name}")
      lines = [
          line.split(",", 1) for line in genre_file_path.read_text().splitlines()[1:]
      ]
      snippet_ids_for_genre = sum(
          (line[snippet_ids_column_index].split(";") for line in lines if line[0] == genre_name), []
      )
      files_to_play = [
          str(snippets_path.joinpath(f"{snippet_id}.mp3"))
          for snippet_id in snippet_ids_for_genre
      ]
    # here you add command call to play multiple files like subprocess.run()
  ```
- `songs-mapped_genres.csv` - track id, genre, mapped genre. Built with [create_genre_mapping.py](../create_genre_mapping.py)
- `songs-genre_filtered.csv` - dataset without skipped outliners per mapped genre. Built with [filter_outliers.py](../filter_outliers.py)
- `songs-genre_filtered-outliners.csv` - skipped outliners per mapped genre. Built with [filter_outliers.py](../filter_outliers.py)
- `genre_model.pickle` - binary of the trained model
- `genre_model-stat.csv` - accuracy stats of the trained model
- `test_predictions_match.csv` - model test data with predictions for manual analysis
