from src.dataset.kay import load_dataset


def test_load_dataset():
    assert type(load_dataset(data_path="./data/")) == dict
