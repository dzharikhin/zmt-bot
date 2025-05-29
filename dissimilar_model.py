import polars as pl

from train import train_dissimilar_model

if __name__ == "__main__":
    train_data = pl.read_csv("data/train.csv")
    test_data = pl.read_csv("data/test.csv")
    model, accuracy = train_dissimilar_model(
        train_data=train_data, test_data=test_data, nu=0.66, contamination_fraction=0.2
    )
    print(accuracy)
