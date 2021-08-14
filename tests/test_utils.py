import pytest
import numpy as np

from tests.dummy_data import dummy_data

from src.utils.util import (
    convert_arr_to_img,
    filter_voxel_by_roi,
    filter_stimulus_by_class,
    prepare_fmri_data,
    prepare_stimulus_data,
)


class TestUtils:
    def test_convert_arr_to_img(self):
        # output is a numpy array
        assert (
            type(convert_arr_to_img(stimulus_img_arr=dummy_data["stimuli"]))
            == np.ndarray
        )

        # output array has the same number of samples as input
        assert (
            convert_arr_to_img(stimulus_img_arr=dummy_data["stimuli"]).shape[0] == 1750
        )

        # output array should be 3d
        assert (
            len(convert_arr_to_img(stimulus_img_arr=dummy_data["stimuli"]).shape) == 3
        )

        # output array should be float64
        assert (
            convert_arr_to_img(stimulus_img_arr=dummy_data["stimuli"]).dtype
            == np.float64
        )

        # output array max element should be 255.0
        assert convert_arr_to_img(stimulus_img_arr=dummy_data["stimuli"]).max() == 255.0

        # output array min element should be 0.0
        assert convert_arr_to_img(stimulus_img_arr=dummy_data["stimuli"]).min() == 0.0

    def test_filter_voxel_by_roi(self):
        # roi can lie between [1, 7]
        with pytest.raises(ValueError):
            filter_voxel_by_roi(all_data=dummy_data, roi_list=[10])

        # output must be an array
        assert (
            type(filter_voxel_by_roi(all_data=dummy_data, roi_list=[1, 2]))
            == np.ndarray
        )

    def test_filter_stimulus_by_class(self):
        # label level can lies between [0, 3]
        with pytest.raises(ValueError):
            filter_stimulus_by_class(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["fungus"],
                label_level=7,
            )

        # data_subset can take values - "train" or "test"
        with pytest.raises(ValueError):
            filter_stimulus_by_class(
                all_data=dummy_data,
                data_subset="xyz",
                class_ignore_list=["fungus"],
                label_level=1,
            )

        # ignore_class_list must have atleast 1 value
        with pytest.raises(ValueError):
            filter_stimulus_by_class(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=[],
                label_level=1,
            )

        # output is an array
        assert (
            type(
                filter_stimulus_by_class(
                    all_data=dummy_data,
                    data_subset="test",
                    class_ignore_list=["person", "fungus", "plant"],
                    label_level=1,
                )
            )
            == np.ndarray
        )

    def test_prepare_stimulus_data(self):
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

        # output tuple has 2 objects
        assert (
            len(
                prepare_stimulus_data(
                    all_data=dummy_data,
                    data_subset="test",
                    class_ignore_list=["person", "fungus", "plant"],
                    label_level=1,
                )
            )
            == 2
        )

        # stimuli array in output has same number of arrays as input
        assert (
            prepare_stimulus_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["person", "fungus", "plant"],
                label_level=1,
            )[0].shape[0]
            == 1750
        )

        # class labels in output has same number of arrays as input
        assert (
            prepare_stimulus_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["person", "fungus", "plant"],
                label_level=1,
            )[1].shape[0]
            == 1750
        )

    #################################
    #################################
    #################################
    #################################
    def test_prepare_fmri_data(self):
        # label level can lies between [0, 3]
        with pytest.raises(ValueError):
            prepare_fmri_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["fungus"],
                label_level=7,
                roi_select_list=[1, 2, 3],
            )

        # data_subset can take values - "train" or "test"
        with pytest.raises(ValueError):
            prepare_fmri_data(
                all_data=dummy_data,
                data_subset="xyz",
                class_ignore_list=["fungus"],
                label_level=1,
                roi_select_list=[1, 2, 3],
            )

        # ignore_class_list must have atleast 1 value
        with pytest.raises(ValueError):
            prepare_fmri_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=[],
                label_level=1,
                roi_select_list=[1, 2, 3],
            )

        # roi_select_list must lie between [1, 7]
        with pytest.raises(ValueError):
            prepare_fmri_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=[],
                label_level=1,
                roi_select_list=[1, 2, 10],
            )

        # output is a tuple
        assert (
            type(
                prepare_fmri_data(
                    all_data=dummy_data,
                    data_subset="test",
                    class_ignore_list=["person", "fungus", "plant"],
                    label_level=1,
                    roi_select_list=[1, 2, 3],
                )
            )
            == tuple
        )

        # output tuple has 2 objects
        assert (
            len(
                prepare_fmri_data(
                    all_data=dummy_data,
                    data_subset="test",
                    class_ignore_list=["person", "fungus", "plant"],
                    label_level=1,
                    roi_select_list=[1, 2, 3],
                )
            )
            == 2
        )

        # stimuli array in output has same number of arrays as input
        assert (
            prepare_fmri_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["person", "fungus", "plant"],
                label_level=1,
                roi_select_list=[1, 2, 3],
            )[0].shape[0]
            == 1750
        )

        # class labels in output has same number of arrays as input
        assert (
            prepare_fmri_data(
                all_data=dummy_data,
                data_subset="train",
                class_ignore_list=["person", "fungus", "plant"],
                label_level=1,
                roi_select_list=[1, 2, 3],
            )[1].shape[0]
            == 1750
        )
