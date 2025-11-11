import pytest
from train_pose_inference.src.__main__ import Training_And_Save_App

class Test_main:
    """Functions to test critical functions in main.py"""

    def test_init(self):
        # 1. Test non-existent directory
        dir_does_not_exist = r"random" 
        with pytest.raises(ValueError):
            Training_And_Save_App(dir_does_not_exist)

        # 2. Test for successfull initialization of a valid directory of images
        dir_exists = r"S:\Documents\OpenCVApps\pose_estimation_rough\train_pose_inference\test\samples\valid_dir"
        assert Training_And_Save_App(dir_exists).source_directory == dir_exists

    # <PENDING: INSERT TESTS FOR FUCNTIONS CONCERNING SUCCESFUL MODEL CREATNON AND SAVING>