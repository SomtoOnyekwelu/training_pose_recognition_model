import pytest
import random   # To sample from AllLandmarks

import pose_estimation_rough.train_pose_inference.tests.config as tc
from pose_estimation_rough.train_pose_inference.src.processing import ProcessedLandmarks
from pose_estimation_rough.train_pose_inference.src import utils as utils
from pose_estimation_rough.train_pose_inference.src import data_defs as defs


class Test_isValidDirectory:
    # ---- HELPER FUNCTION
    def getbool(self, func_output: tuple) -> bool:
            """Extracts the boolean part of the output returned by the tested function
            Based on the output of the isValidDirectory function"""
            return func_output[0]
        
    # Cases
    def test_valid_dir(self):
        # True Case: The directory is made up of only second directory folders, which contain only images
        valid_dir = tc.valid_dir
        assert self.getbool(utils.isValidDirectory(valid_dir)) is True

    def test_nonexistent_dir(self):
        # The directory does not exist
        nonexistent_dir = r"does_not_exist"
        assert self.getbool(utils.isValidDirectory(nonexistent_dir)) is False

    def test_empty_dir(self):
        # The directory is empty
        empty_dir = str(tc.ROOT_OF_TEST_DIR / "samples" / "empty_directory")
        assert self.getbool(utils.isValidDirectory(empty_dir)) is False

    def test_invalid_dir_not_only_dirs_at_top_level(self):
        # The directory is not made up of only second directory folders
        invalid_dir = str(tc.ROOT_OF_TEST_DIR / "samples" / "invalid_dir_non_folder")
        assert self.getbool(utils.isValidDirectory(invalid_dir)) is False

    def test_invalid_dir_non_image_file(self):
        # The seemingly second directory folders contains non-images.
        file_corruption_dir = str(tc.ROOT_OF_TEST_DIR / "samples" / "invalid_dir_non_image")
        assert self.getbool(utils.isValidDirectory(file_corruption_dir)) is False

def sample(population: list, sample_size: int = -1) -> list:
    """Returns a sample of the population\n
    The sample size is defined with the corresponding parameter"""

    sample = []
    population_size = len(population)
    
    if sample_size == -1:
        # Set the sample size to be less than 100, chosen because a list of that length can be gone through fast enough
        sample_size = (population_size % 100) + 1   # Added 1 to prevent a zero sample size when the population size is a multiple of 100
    
    elif 1 <= sample_size <= population_size:
        pass

    else:
        raise ValueError(f"Sample size should be a positive number and must be less than the size of the AllLandmarks parameter. Population size: {population_size}, Sample size: {sample_size}")
    
    max_selection_idx = population_size - 1 # To prevent accessing an out of bounds index

    # Generates random integers in [0, sample_size), and uses them to sample
    # Does not bother to prevent non-unique sampling since ensuring uniqueness by checking an accumulator would be compuationally intensive
    for selection in range(sample_size):
        idx = random.randint(0, max_selection_idx)

        selection = population[idx]

        sample.append(selection)
        
    return sample

class Test_get_next_xyz_and_rest_of_OneSetOfLandmarks:
    
    def test_empty_part_of_an_OneSetOfLandmarks(self):
        """Tests that the function can handle empty input"""

        output = utils.get_next_xyz_and_rest_of_OneSetOfLandmarks([])

        assert output == (None, None, None, [])

    def test_non_empty_part_of_an_OneSetOfLandmarks(self):
        """Tests that the function can handle non-empty input"""
        
        dummy_input = [0.1, 0.2, 0.3, 0.4] * 3

        output = utils.get_next_xyz_and_rest_of_OneSetOfLandmarks(dummy_input)

        assert output == (0.1, 0.2, 0.3, [0.1, 0.2, 0.3, 0.4] * 2)
    
class Test_convert_image_paths_to_OneSetOfLandmarks:
    """Task: Tests the function that converts the image paths to an OneSetOfLandmarks"""


    

