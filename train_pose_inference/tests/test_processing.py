import pytest
import numpy as np
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from types import SimpleNamespace   # For monkey patching of tests.
from typing import Optional # For static typing in function contracts.
import matplotlib
matplotlib.use('Agg')   # To avoid Tcl errors when testing

from pose_estimation_rough import config
from pose_estimation_rough.train_pose_inference.src import processing as proc
from pose_estimation_rough.train_pose_inference.src.processing import ProcessedLandmarks
from pose_estimation_rough.train_pose_inference.src import utils as utils
from pose_estimation_rough.train_pose_inference.src import data_defs as defs
from pose_estimation_rough.train_pose_inference.src import landmarker_model as lm
from pose_estimation_rough.train_pose_inference.tests import test_utils as tutils


"""
DATA DEFINITIONS
-----------------
A directory contains Second_level_Directory's or other and is one of:
    Empty, 
    length=1 
    length>1
A Second_level_Directory contains Files and is one of: 
    Empty, 
    One File, 
    File>1
A File is one of:
    Image
    unknown
An Image is one of:
    cannot_be_read
    can_be_read_Image
A can_be_read_Image is one of: 
    no person in image
    one person in image
    multiple persons in image.
"""
def create_dummy_OneSetOfLandmarks(xyz_presence: list[float] = [0.1, 0.2, 0.3, 0.4]) -> defs.OneSetOfLandmarks:
    """Builds and returns a dummyOneSetOfLandmarks from the input
    The passed value must be a list of our floats."""

    return xyz_presence * 33

class Test_processing:

    # ---- HELPER FUNCTIONS 
    def _is_center_at_zero(self, an_AllLandmarks: defs.AllLandmarks, sample_size: int = -1) -> bool:
        """Returns true if the x, y and z of the hip midpoint represented by each landmark is (0, 0, 0) """
        for SetOfLandmarks in an_AllLandmarks:
            midpoint = lm.extract_hip_midpoint(SetOfLandmarks)

            if midpoint != (0, 0, 0):
                return False
    
        return True

    def _sample_landmarks(self, an_AllLandmarks: defs.AllLandmarks, sample_size: int = -1) -> defs.AllLandmarks:
        """Returns a random sample of the landmarks.
        The sample size is defined with the corresponding parameter"""
        
        return tutils.sample(an_AllLandmarks, sample_size)
    
    def _is_size_consistent(self, an_AllLandmarks: defs.AllLandmarks, max_error: float = 0.01) -> bool:
        """
        Returns True if the sizes are the same within some degree of error.\n
        Method: The size of each OneSetOfLandmarks is the average distance of all landmark components from the midpoint component.\n
        Assumes that the midpoint coordinate has been translated to (x=0, y=0, z=0).
        """
        # To avoid an O(n^2) check through a loop to check the maximum intra-deviation of sum-of-distance, we keep track of the min and max sum of distances across all members of the an_AllLandmarks and finally, check if their divergence is greater than error.
        min_distance_sum, max_distance_sum = 0, 0

        for an_OneSetOfLandmarks in an_AllLandmarks:
            # Get the sum of intra-distances with respect to its midpoint.
            sum_of_distances = lm.get_sum_of_distances_from_midpoint(an_OneSetOfLandmarks)

            ## Update the min and max distances if neccesary
            if sum_of_distances < min_distance_sum:
                 min_distance_sum = sum_of_distances
            elif sum_of_distances > max_distance_sum:
                 max_distance_sum = sum_of_distances
            
        # Check if the divergence is more than the acceptable margin
        error = max_distance_sum - min_distance_sum
        
        if error > max_error:
            return False
        else: 
            return True  
    
    def _are_shoulders_horozontal(self, an_AllLandmarks: defs.AllLandmarks) -> bool:
        """Returns True if the shoulders of each OneSetOfLandmarks form a horozontal line\n
        Method: If the y and z of both shoulder coordinates of an_OneSetOfLandmarks are both zero, => they form a horozontal line."""
        # Retrieves the (x, y, z) of the two shoulders
        for an_OneSetOfLandmarks in an_AllLandmarks:
            xl, yl, zl, xr, yr, zr = lm.extract_shoulder_coords(an_OneSetOfLandmarks)
            
            if (yl, zl, yr, zr) != (0, 0, 0, 0):
                return False
        
        return True
    
    # ---- MAIN FUNCTIONS
        
class Test_convert_directory_to_landmarks:
    """
    See definition of a directory in data_defs.
    """
    def _areLengthsEqual(self, result_landmarks: np.ndarray, result_pose_labels: np.ndarray) -> bool:
        """Returns True if both parameters have equal number of first degree elements.\n
        Otherwise, returns False."""

        if result_landmarks.shape[0] == result_pose_labels.shape[0]:
            return True

        else:
            return False
    
    def _isNumberOfUniquePoseLabels(self, result_pose_labels: np.ndarray, expected_count: int) -> bool:
        """Returns True if expected_count == number of unique pose labels."""
        # Since the pose_labels are already flattened, we pass it to set directly
        actual_count_of_unique_pose_labels = len(set(result_pose_labels))

        print(f"Set of unique pose labels: {set(result_pose_labels)}")

        return actual_count_of_unique_pose_labels == expected_count

    def test_non_existent_directory(self):
        # The directory does not exist
        non_existent_directory = r"non_existent_directory"
        with pytest.raises(AssertionError):
            ProcessedLandmarks(non_existent_directory).convert_directory_to_landmarks()
    
    def test_directory_is_not_a_folder(self):
        """Case: The directory is not a folder.\n
        SHould raise an error"""
        non_folder = r"tests\samples\random.txt"
        with pytest.raises(AssertionError):
            ProcessedLandmarks(non_folder).convert_directory_to_landmarks()

    def test_empty_directory(self):
        """The directory is empty.
        Should raise an error."""
        empty_directory = r"tests\samples\empty_directory"
        with pytest.raises(AssertionError):
            ProcessedLandmarks(empty_directory).convert_directory_to_landmarks()
    
    def test_one_subdirectory_valid_directory(self):
        """Case: self.directory contains only one sub-directory and is a valid directory. 
        Should always return a tuple, containing the AllLandmarks and the list of all labels, with only one unique label"""
        # The directory is a valid directory, even down to the subdirectries and their contents
        valid_directory_with_one_subdir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir_one_subdir")
        result_landmarks, result_pose_lables = ProcessedLandmarks(valid_directory_with_one_subdir).convert_directory_to_landmarks()

        # Checks that there is only one unique pose label and, the length of the result_landmarks and the result_pose_lables are equal.
        isExpectedResult = self._areLengthsEqual(result_landmarks, result_pose_lables) and self._isNumberOfUniquePoseLabels(result_pose_lables, 1)

        print(f"Number of unique pose labels: {self._isNumberOfUniquePoseLabels(result_pose_lables, 1)}")

        print(f"Equal length?: {self._areLengthsEqual(result_landmarks, result_pose_lables)}")

        assert isExpectedResult

    def test_one_subdirectory_invalid_directory(self):
        """Case: self.directory contains only one sub-directory and is an invalid directory.\n
        It should raise an error."""
        # The directory is an invalid directory, such as containing a corrupted file or a phote in the sub-directory level.
        invalid_directory_with_one_subdir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "invalid_dir_one_subdir")
        with pytest.raises(AssertionError):
            ProcessedLandmarks(invalid_directory_with_one_subdir).convert_directory_to_landmarks()
    
    def test_multiple_subdirectories_valid_directory(self):
        """Case: self.directory contains multiple sub-directories and is a valid directory. 
        Should always return a tuple, containing the AllLandmarks and the list of all labels, with multiple unique labels."""
        # The directory is a valid directory, even down to the subdirectries and their contents
        valid_directory_with_three_subdir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir_three_subdir")
    
        result_landmarks, result_pose_lables = ProcessedLandmarks(valid_directory_with_three_subdir).convert_directory_to_landmarks()

        # Checks that there are only three unique pose labels and, the length of the result_landmarks and the result_pose_lables are equal.
        isExpectedResult = self._areLengthsEqual(result_landmarks, result_pose_lables) and self._isNumberOfUniquePoseLabels(result_pose_lables, 3)

        print(f"Number of unique pose labels: {self._isNumberOfUniquePoseLabels(result_pose_lables, 1)}")

        assert isExpectedResult

    def test_multiple_subdirectories_invalid_directory(self):
        """Case: self.directory contains multiple sub-directories and is an invalid directory.\n
        It should raise an error."""
        # The directory is an invalid directory, such as containing a corrupted file or a phote in the sub-directory level.
        invalid_directory_with_three_subdir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "invalid_dir_three_subdir")

        with pytest.raises(AssertionError):
            ProcessedLandmarks(invalid_directory_with_three_subdir).convert_directory_to_landmarks()
        
class Test_extract_all_x_y_z_of_OneSetOfLandmarks:
    """Tests the function that extracts the x, y and z of a OneSetOfLandmarks"""

    def test_valid_OneSetOfLandmarks(self):
        """Checks if the sole case of extracting from a valid OneSetOfLandmarks works.
        Neccessary to ensure normal execution"""
        
        # Build our test case 
        a_valid_OneSetOfLandmarks = [0.1, -0.2, 0.3, -0.4] * 33

        # Build the result of combining the correct output
        xyz_of_a_valid_OneSetOfLandmarks = [(0.1, -0.2, 0.3)] * 33

        # Test
        all_x, all_y, all_z = proc.extract_all_x_y_z_of_OneSetOfLandmarks(a_valid_OneSetOfLandmarks)

        ## Combine the output, to compare with the baseline result.
        result_flat_x_y_z = list(zip(all_x, all_y, all_z))
        
        assert result_flat_x_y_z == xyz_of_a_valid_OneSetOfLandmarks

class Test_create_scatter_and_line_artists_for_AllLandmarks:
    """Tests the function that creates a list-of PlotTuple from an AllLandmarks"""

    example_threeD_axes = proc.create_3D_axes()
    
    def test_empty_AllLandmarks(self):
        """The case of reciving an empty AllLandmarks_numpy. Should return an empty list"""
        expected = []

        empty_AllLandmarks = np.empty(shape=[0,])
        
        assert (proc.create_scatter_and_line_artists_for_AllLandmarks(self.example_threeD_axes, empty_AllLandmarks) == expected)

    def test_non_empty_AllLandmarks(self):
        """"The case for receiving a non-empty AllLandmarks Should return a list of PlotTuple of the same length."""

        AllLandmarks_with_two_OneSetOfLandmarks = [create_dummy_OneSetOfLandmarks(), 
        create_dummy_OneSetOfLandmarks()]

        output = proc.create_scatter_and_line_artists_for_AllLandmarks(self.example_threeD_axes, np.array(AllLandmarks_with_two_OneSetOfLandmarks))

        # Should return a PlotTuple for each OneSetOfLandmarks in AllLandmarks
        assert len(output) == 2

class Test_filter_for_components:
    """Tests the function that returns the x, y and z of an OneSetOfLandmarks for specific component names."""

    dummy_x_coords = [float(i) for i in range(33)]  # Input must be a list of floats
    dummy_y_coords = [i * 2.0 for i in (range(33))]  # 
    dummy_z_coords = [i * 3.0 for i in (range(33))]

    def test_empty_component_list(self):
        """Tests for an empty component list.
        Should return empty x, y and z lists"""

        no_component_names = []

        output = proc.filter_for_components(no_component_names, self.dummy_x_coords, self.dummy_y_coords, self.dummy_z_coords)

        assert output == ([], [], [])

    def test_non_empty_component_list(self, monkeypatch):
        """Tests for a non-empty component list.
        Should return a tuple containing three lists, where:
        - each is of the same length as the component list and,
        - is made up of coordinates of the components in the component list (for the corresponding axis: x, y, and z)
        """

        # To test only the function's behaviour and not global state, and to make the test results more obvious from here, we have to make the test self contained.
        # So, we create a dummy mapping of the component names.

        # This would replace the mapping called within the tested function, as long as it is within this test function.
        dummy_mapping = {"nose": 0, "right_elbow": 14, "left_ankle": 27, "right_foot_tip": 32}   # All that is needed for this test case

        monkeypatch.setattr(
            proc, # The tested module object
            "lm", # How the module containing the below mapping is imported
            SimpleNamespace(
                map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks = dummy_mapping
            ),
            raising = True  # To catch typo errors
        )

        # The list of component names to be passed into the function to be tested
        non_empty_component_names = list(dummy_mapping.keys())[1:]  # Test onl a part of the mappings
        
        # Building the ground truth answers.
        ground_truth_x, ground_truth_y, ground_truth_z = ([], [], [])

        for name in non_empty_component_names:
            idx_of_name_in_OneSetOfLandmarks = dummy_mapping[name]

            ## Add the corresponding coordinates 
            ground_truth_x.append(self.dummy_x_coords[idx_of_name_in_OneSetOfLandmarks])

            ground_truth_y.append(self.dummy_y_coords[idx_of_name_in_OneSetOfLandmarks])

            ground_truth_z.append(self.dummy_z_coords[idx_of_name_in_OneSetOfLandmarks])

        output = proc.filter_for_components(non_empty_component_names, self.dummy_x_coords, self.dummy_y_coords, self.dummy_z_coords)

        assert output == (ground_truth_x, ground_truth_y, ground_truth_z)

    def test_all_names_in_component_mapping(self, monkeypatch):
        """Tests that the function can access any component in the mapping"""

        dummy_mapping = {"nose": 0, "right_elbow": 14, "left_ankle": 27, "right_foot_tip": 32}   # All that is needed for this test case

        monkeypatch.setattr(
            proc, # The tested module object
            "lm", # How the module containing the below mapping is imported
            SimpleNamespace(
                map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks = dummy_mapping
            ),
            raising = True  # To catch typo errors
        )

        # The list of component names to be passed into the function to be tested
        non_empty_component_names = list(dummy_mapping.keys())  # Test only a part of the mappings
        
        # Building the ground truth answers.
        ground_truth_x, ground_truth_y, ground_truth_z = ([], [], [])

        for name in non_empty_component_names:
            idx_of_name_in_OneSetOfLandmarks = dummy_mapping[name]

            ## Add the corresponding coordinates 
            ground_truth_x.append(self.dummy_x_coords[idx_of_name_in_OneSetOfLandmarks])

            ground_truth_y.append(self.dummy_y_coords[idx_of_name_in_OneSetOfLandmarks])

            ground_truth_z.append(self.dummy_z_coords[idx_of_name_in_OneSetOfLandmarks])

        output = proc.filter_for_components(non_empty_component_names, self.dummy_x_coords, self.dummy_y_coords, self.dummy_z_coords)

        assert output == (ground_truth_x, ground_truth_y, ground_truth_z)

    def test_invalid_component_name_in_list(self, monkeypatch):
        """Tests for a component list containing an invalid name, that is, containing a string that is not the name of any pose component in the mapping.
        
        Should raise an error.
        """

        dummy_mapping = {"nose": 0, "right_elbow": 14}   # All that is needed for this test case

        monkeypatch.setattr(
            proc, # The tested module object
            "lm", # How the module containing the below mapping is imported
            SimpleNamespace(
                map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks = dummy_mapping
            ),
            raising = True  # To catch typo errors
        )

        # The list of component names to be passed into the function to be tested
        with_invalid_component_names = ["neck", "nose", 1, "right_elbow"] # neck and 1 are invalid component names in this context, since they are not inn the mapping.
        
        with pytest.raises(ValueError):
            proc.filter_for_components(with_invalid_component_names, self.dummy_x_coords, self.dummy_y_coords, self.dummy_z_coords)

    def test_duplicate_component_name_in_list(self, monkeypatch):
        """
        Tests for a component list containing duplicate name.
        Should raise an error.
        """

        # The list of component names to be passed into the function to be tested
        with_duplicate_component_names = ["nose", "right_elbow", "nose"]
    
        with pytest.raises(ValueError):
            proc.filter_for_components(with_duplicate_component_names, self.dummy_x_coords, self.dummy_y_coords, self.dummy_z_coords)
        
class Test_create_scatter_artists_for_key_components:
    def test_normal_case(self, monkeypatch):
        """The single use case: given an OneSetOfLandmarks and an Axes3D"""

        dummy_OneSetOfLandmarks = create_dummy_OneSetOfLandmarks([0.1, 0.2, 0.3, 0.4])
        dummy_Axes_3D = proc.create_3D_axes()

        # Use the list of artist objects created to know if each key point of the OneSetOfLandmarks (components in the mapping) is included in the scatter plot.

        # To get ground truth.
        # Extract x, y and z of key points to be plotted
        number_of_key_components = len(lm.map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks)
        number_of_keypoints = 2 # Hip and shoulder midpoints
        
        total_len_of_coordinate_lists = number_of_key_components + number_of_keypoints

        ground_x = [0.1] * total_len_of_coordinate_lists
        ground_y = [0.2] * total_len_of_coordinate_lists
        ground_z = [0.3] * total_len_of_coordinate_lists
        
        # Get function output
        scatter_artist = proc.create_scatter_artists_for_key_components(dummy_Axes_3D, dummy_OneSetOfLandmarks)
        
        ## Extract x, y and z
        output_x, output_y, output_z = scatter_artist._offsets3d

        # Assert output x, y, and z == ground x, y and z
        assert (ground_x, ground_y, ground_z) == (output_x.tolist(), output_y.tolist(), output_z.tolist())

class Test__maybe_give_next_idx:
    """Tests the function that decides the PlotTuple to render when a navigation key is pressed."""

    dummy_next_key = "r"
    dummy_prev_key = "l"
    dummy_other_key = "any_other_key"

    @pytest.mark.parametrize("the_pressed_key",
                              [dummy_prev_key, dummy_next_key, dummy_other_key])
    # The below covers the cases for the currently displayed PlotTuple, that is, if it is the first, second or sixth.
    ## If this is larger than the number of PlotTuples to display in some context, say when it is called from outside the function, it should return an error.
    @pytest.mark.parametrize("idx_of_PlotTuple_currently_displayed",
                              [0, 2, 6])   
    # The below covers the cases for the number of PlotTupls to display
    @pytest.mark.parametrize("no_of_PlotTuples_for_display",
                              [0, 1, 5])
    
    def test_all_cases(self, the_pressed_key, idx_of_PlotTuple_currently_displayed, no_of_PlotTuples_for_display):
        """Tests the functions against all possible cases of input combination, as generated by the pytest parametrize function."""

        # Test that an error is raused for these edge cases: where the index of current PlotTuple is larger than the number of PlotTuples.
        max_idx = no_of_PlotTuples_for_display - 1
        if idx_of_PlotTuple_currently_displayed > max_idx:
            with pytest.raises(ValueError):
                proc._maybe_give_next_idx(the_pressed_key, idx_of_PlotTuple_currently_displayed, no_of_PlotTuples_for_display, self.dummy_next_key, self.dummy_prev_key)
        else: 
            output = proc._maybe_give_next_idx(the_pressed_key, idx_of_PlotTuple_currently_displayed, no_of_PlotTuples_for_display, self.dummy_next_key, self.dummy_prev_key)

            assert self._is_correct(output, idx_of_PlotTuple_currently_displayed, the_pressed_key, no_of_PlotTuples_for_display)

    def _is_correct(self, result: tuple[bool, Optional[int]], idx_of_currently_displayed_PlotTuple: int, the_pressed_key: str, count_of_PlotTuples: int) -> bool:
        """Validates if the result is the expected answer based on the inputs.\n
        Expects that the function would not raise an error on the set of inputs it would receive """

        # The cases where a non-navigation key is pressed.
        if the_pressed_key == self.dummy_other_key:
            return (result == (False, None))
        # The cases where a navigation key is pressed.
        ## If the previous key is presssed
        elif the_pressed_key == self.dummy_prev_key:
            ### If we are at the first plot, the result is the last plot
            if idx_of_currently_displayed_PlotTuple == 0:
                max_idx = count_of_PlotTuples - 1
                expected_prev_at_0_idx = (True, max_idx)
                return (result == expected_prev_at_0_idx)
            ### At any other plot, the result is the preceeding plot
            else:
                expected_prev_at_other_idx = (True, idx_of_currently_displayed_PlotTuple - 1)
                return (result == expected_prev_at_other_idx)
        ## The next key is pressed
        elif the_pressed_key == self.dummy_next_key:
            max_idx = count_of_PlotTuples - 1
            ### If we are at the last plot, the result is the first plot
            if idx_of_currently_displayed_PlotTuple == max_idx:
                expected_next_at_0_idx = (True, 0)
                return (result == expected_next_at_0_idx)
            ### At any other plot, the result is the following plot
            else:
                expected_next_at_other_idx = (True, idx_of_currently_displayed_PlotTuple + 1)
                return (result == expected_next_at_other_idx)
        else:
            ## This case is not reachable since we have only three possible pressed_key values. 
            ## This is left here as a net.
            raise ValueError("This test case is invalid.")

class Test__next_idx_of_keypress:
    """Tests the function that decides the next index to move on to, given a keypress and the current state."""

    valid_next_direction = "next"
    valid_prev_directions = "prev"
    
    all_valid_directions = [valid_next_direction, valid_prev_directions]    # Combined to aid in parametization
    
    invalid_direction = "other" # Represents all other strings apart from the valid ones above.

    valid_and_invalid_directions = all_valid_directions + [invalid_direction]

    # ---- CASE: LENGTH OF ITERABLE < 0
    @pytest.mark.parametrize("any_direction",
                            valid_and_invalid_directions)
    @pytest.mark.parametrize("any_index",
                            [-2, -1, 0, 1, 2])
    def test_negative_len(self, any_direction, any_index):
        """Case: 
        length < 0
        index = unimportant
        direction: unimportant
        Should raise an error."""

        negative_length = -3
        with pytest.raises(AssertionError):
            proc._next_idx_of_keypress(any_direction, any_index, negative_length)

    # ---- CASE: LENGTH OF ITERABLE = 0
    @pytest.mark.parametrize("any_direction",
                            valid_and_invalid_directions)
    @pytest.mark.parametrize("any_index",
                            [-2, -1, 0, 1, 2])
    def test_zero_len(self, any_direction, any_index):
        """Case: 
        length = 0
        index = unimportant
        direction: unimportant
        Should raise an error."""

        zero_length = 0
        with pytest.raises(AssertionError):
            proc._next_idx_of_keypress(any_direction, any_index, zero_length)
    
    # CASES: LENGTH OF ITERABLE = 1
    @pytest.mark.parametrize("valid_direction",
                                all_valid_directions)
    def test_len_one_valid_direction_at_index_zero(self, valid_direction):
        """Case: 
        length = 1
        index = 0
        direction: valid
        Should always return zero as the next_idx."""
        mono_length = 1
        zero_idx = 0

        expected = 0

        output = proc._next_idx_of_keypress(valid_direction, zero_idx, mono_length)

        assert output == expected
    
    @pytest.mark.parametrize("any_index",
                                [-2, -1, 0, 1, 3])
    def test_len_one_invalid_direction(self, any_index):
        """Case: 
        length = 1
        index: unimportant
        direction: invalid
        Should always return an error."""

        mono_length = 1

        with pytest.raises(AssertionError):
            proc._next_idx_of_keypress(self.invalid_direction, any_index, mono_length)

    @pytest.mark.parametrize("non_zero_index",
                                [-2, -1, 1, 3])
    @pytest.mark.parametrize("valid_direction",
                                all_valid_directions)
    def test_len_one_nonzero_index(self, non_zero_index, valid_direction):
        """Case: 
        length = 1
        index != 0
        direction: valid
        Should always return an error."""

        mono_length = 1

        with pytest.raises(AssertionError):
            proc._next_idx_of_keypress(valid_direction, non_zero_index, mono_length)

    # ---- CASES: GENERAL LENGTH
    def test_len_above_one_next_direction_at_index_zero(self):
        """Case: 
        length > 1
        index = 0
        direction = "next"
        Should always return one as the next_idx."""
        length = 5
        zero_idx = 0
        direction = "next"

        expected = 1

        output = proc._next_idx_of_keypress(direction, zero_idx, length)

        assert output == expected
    
    def test_len_above_one_next_direction_at_common_index(self):
        """Case: 
        length > 1
        index: 0 < index < max_index, where max_index = length - 1
        direction = "next"
        Should always return index + 1."""

        length = 5
        idx = 2
        direction = "next"

        expected = 3

        output = proc._next_idx_of_keypress(direction, idx, length)

        assert output == expected

    def test_len_above_one_next_direction_at_last_index(self):
        """Case: 
        length > 1
        index = max_index, where max_index = length - 1
        direction = "next"
        Should always return 0."""

        length = 5
        max_idx = 4
        direction = "next"

        expected = 0

        output = proc._next_idx_of_keypress(direction, max_idx, length)

        assert output == expected

    def test_len_above_one_prev_direction_at_index_zero(self):
        """Case: 
        length > 1
        index = 0
        direction = "prev"
        Should always return max_idx as the next_idx, where max_idx = length - 1"""
        length = 5
        zero_idx = 0
        direction = "prev"

        expected = 4

        output = proc._next_idx_of_keypress(direction, zero_idx, length)

        assert output == expected
    
    def test_len_above_one_prev_direction_at_common_index(self):
        """Case: 
        length > 1
        index: 0 < index < max_index, where max_index = length - 1
        direction = "prev"
        Should always return index - 1."""

        length = 5
        idx = 2
        direction = "prev"

        expected = 1

        output = proc._next_idx_of_keypress(direction, idx, length)

        assert output == expected

    def test_len_above_one_prev_direction_at_last_index(self):
        """Case: 
        length > 1
        index = max_index, where max_index = length - 1
        direction = "prev"
        Should always return max_index - 1."""

        length = 5
        max_idx = 4
        direction = "prev"

        expected = 3

        output = proc._next_idx_of_keypress(direction, max_idx, length)

        assert output == expected
    
    @pytest.mark.parametrize("invalid_index_wrt_len_five",
                                [-2, -1, 5, 6])
    @pytest.mark.parametrize("valid_direction",
                                all_valid_directions)
    def test_len_above_one_invalid_index(self, invalid_index_wrt_len_five, valid_direction):
        """Case: 
        length > 1
        invalid_index = negative int, or index >= length
        direction = unimportant
        Should always raise an error."""

        length = 5

        with pytest.raises(AssertionError):
            proc._next_idx_of_keypress(valid_direction, invalid_index_wrt_len_five, length)

    def test_len_above_one_invalid_direction(self):
        """Case: 
        length > 1
        invalid_index: valid
        direction: invalid
        Should always raise an error."""

        length = 5
        example_valid_idx = 3

        with pytest.raises(AssertionError):
            proc._next_idx_of_keypress(self.invalid_direction, example_valid_idx, length)
    
# Prospective tests for normalization pipeline for datasets
    #     ## Check that the features are translated properly, a
    #     ## and remain so even after scaling and rotation
    #     assert self._is_center_at_zero(sample_landmarks)

    #     ## Test for proper scaling
    #     ## that is, the size of each set of landmarks is constant within some level
    #     assert self._is_size_consistent(sample_landmarks)

    #     ## Test for proper rotation
    #     ### that is, the shoulders form a horozontal line
    #     assert self._are_shoulders_horozontal(sample_landmarks)



        



    

    







    



        







        



