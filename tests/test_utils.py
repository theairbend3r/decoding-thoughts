import pytest
import numpy as np

from src.utils.utils import filter_by_roi, convert_arr_to_img

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


class TestUtils:
    def test_filter_by_roi(self):
        # roi can lie between [1, 7].
        with pytest.raises(ValueError):
            filter_by_roi(all_data=dummy_data, roi=10)

        assert type(filter_by_roi(all_data=dummy_data, roi=1)) == np.ndarray

    def convert_arr_to_img(self):
        assert (
            type(convert_arr_to_img(stimulus_img_arr=np.random.rand(1750, 128, 128)))
            == np.ndarray
        )

        assert (
            convert_arr_to_img(stimulus_img_arr=np.random.rand(120, 128, 128)).shape[0]
            == 120
        )
