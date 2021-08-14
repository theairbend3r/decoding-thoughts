import pytest
from tests.dummy_data import dummy_data

import numpy as np
from src.explore.summary import plot_hierarchical_labels


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
