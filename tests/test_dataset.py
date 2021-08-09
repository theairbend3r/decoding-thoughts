from src.dataset.kay import load_dataset


def test_load_dataset():
    assert len(load_dataset(data_path="./data/")) == 3
