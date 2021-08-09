from src.dataset.kay import load_dataset


def test_load_dataset():
    assert load_dataset(data_path="./data") == "dataset"
