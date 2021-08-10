import pytest
import numpy as np

from src.explore.summary import plot_hierarchical_labels

all_data_info = {
    "stimuli": (1750, 128, 128),
    "stimuli_test": (120, 128, 128),
    "responses": (1750, 8428),
    "responses_test": (120, 8428),
    "roi": (8428,),
    "roi_names": (8,),
    "train_labels": (1750, 4),
    "test_labels": (120, 4),
}

dummy_data = {k: None for k in all_data_info.keys()}

for k in all_data_info.keys():
    dummy_data[k] = np.random.random_sample(size=(all_data_info[k]))


class TestSummary:
    def test_plot_hierarchical_labels(self):
        # label_level can only be one of - 0, 1, 2, 3
        with pytest.raises(ValueError):
            plot_hierarchical_labels(all_data=dict(), label_level=5)

        # shape of output must be 1750
        assert (
            len(
                plot_hierarchical_labels(all_data=dummy_data, label_level=0, plot=False)
            )
            == 1750
        )

        # type of output must by np.ndarray
        assert (
            type(
                plot_hierarchical_labels(all_data=dummy_data, label_level=0, plot=False)
            )
            == np.ndarray
        )
