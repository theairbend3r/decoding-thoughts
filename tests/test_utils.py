import pytest
import numpy as np

from src.utils.util import (
    filter_data_by_roi,
    convert_arr_to_img,
    filter_data_by_class,
    prepare_stimulus_data,
)

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
    def test_filter_data_by_roi(self):
        # roi can lie between [1, 7]
        with pytest.raises(ValueError):
            filter_data_by_roi(all_data=dummy_data, roi=10)

        assert type(filter_data_by_roi(all_data=dummy_data, roi=1)) == np.ndarray

    def test_convert_arr_to_img(self):
        # output is a numpy array
        assert (
            type(convert_arr_to_img(stimulus_img_arr=np.random.rand(1750, 128, 128)))
            == np.ndarray
        )

        # output array has the same number of samples as input
        assert (
            convert_arr_to_img(stimulus_img_arr=np.random.rand(120, 128, 128)).shape[0]
            == 120
        )

    def test_filter_data_by_class(self):
        # label level can lies between [0, 3]
        with pytest.raises(ValueError):
            filter_data_by_class(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["fungus"],
                label_level=7,
            )

        # data_subset can take values - "train" or "test"
        with pytest.raises(ValueError):
            filter_data_by_class(
                all_data=dummy_data,
                data_subset="xyz",
                class_ignore_list=["fungus"],
                label_level=1,
            )

        # ignore_class_list must have atleast 1 value
        with pytest.raises(ValueError):
            filter_data_by_class(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=[],
                label_level=1,
            )

        # output is an array
        assert (
            type(
                filter_data_by_class(
                    all_data=dummy_data,
                    data_subset="test",
                    class_ignore_list=["person", "fungus", "plant"],
                    label_level=1,
                )
            )
            == np.ndarray
        )

    def test_prepare_data_arrays(self):
        # label level can lies between [0, 3]
        with pytest.raises(ValueError):
            prepare_stimulus_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["fungus"],
                label_level=7,
            )

        # data_subset can take values - "train" or "test"
        with pytest.raises(ValueError):
            prepare_stimulus_data(
                all_data=dummy_data,
                data_subset="xyz",
                class_ignore_list=["fungus"],
                label_level=1,
            )

        # ignore_class_list must have atleast 1 value
        with pytest.raises(ValueError):
            prepare_stimulus_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=[],
                label_level=1,
            )

        # output is a tuple
        assert (
            type(
                prepare_stimulus_data(
                    all_data=dummy_data,
                    data_subset="test",
                    class_ignore_list=["person", "fungus", "plant"],
                    label_level=1,
                )
            )
            == tuple
        )
