import pytest

# To empty the save directory for the artifacts.
import os
import shutil

from pose_estimation_rough.train_pose_inference.src.__main__ import Training_And_Save_App
from pose_estimation_rough import config

class Test_main:
    """Functions to test critical functions in __main__.py"""
    
    non_existent_directory = r"non_existent_directory" 
    
    valid_source_dir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir_three_subdir")
    valid_dummy_save_dir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "dummy_save_dir_for_artifacts")

    def test_failed_initialization_of_non_existent_source_directory_and_invalid_save_dir(self):
        """
        Test for failed initialization of a invalid directory of images, 
        and an invalid save directory
        """
        with pytest.raises(ValueError):
            Training_And_Save_App(self.non_existent_directory, self.non_existent_directory)

    def test_failed_initialization_of_non_existent_source_directory_and_valid_save_dir(self):
        """
        Test for failed initialization of a invalid directory of images, 
        and a valid save directory.
        """
        with pytest.raises(ValueError):
            Training_And_Save_App(self.non_existent_directory, self.valid_dummy_save_dir)
    
    def test_failed_init_of_valid_directory_and_invalid_save_dir(self):
        """
        Test for failed initialization of a valid directory of images, 
        and an invalid save directory
        """
        with pytest.raises(ValueError):
            Training_And_Save_App(self.valid_source_dir, self.non_existent_directory)

    def test_successful_init_of_valid_directory_and_valid_save_dir(self):
        """
        Test for succcessful initialization of a valid directory of images, 
        and a valid save directory
        """
        # First, empty the save directory, to ensure that only the saved artifacts are counted later on.
        _empty_directory(self.valid_dummy_save_dir)

        # Run the training and save sequence
        Training_And_Save_App(self.valid_source_dir, self.valid_dummy_save_dir)

        # Count the number of artifacts in the save dir
        count_of_created_artifacts = len(os.listdir(self.valid_dummy_save_dir))

        # Empty the save directory, to return it back to starting state.
        # Best practice, to avoid later conflicts.
        _empty_directory(self.valid_dummy_save_dir)

        # Assert
        # Counting the direct XGBoost Clasifier model and the Label Encoder.
        # Later changes in artifact creation would require updating this test case, as expected.
        assert count_of_created_artifacts == 2

def _empty_directory(a_dir: str):
    """Empties the directory at a_dir."""

    # Removes the entire directory
    shutil.rmtree(a_dir)
    # Recreates the directory, now empty.
    os.mkdir(a_dir)