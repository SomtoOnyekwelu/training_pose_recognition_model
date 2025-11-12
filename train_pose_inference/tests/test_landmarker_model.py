import pytest

import pose_estimation_rough.config as config
from pose_estimation_rough.train_pose_inference.src import landmarker_model as lm
from pose_estimation_rough.train_pose_inference.src import processing as proc
from pose_estimation_rough.train_pose_inference.src import data_defs as defs
from pose_estimation_rough.train_pose_inference.src import utils as utils
from pose_estimation_rough.train_pose_inference.tests import config as tc
from pose_estimation_rough.train_pose_inference.tests import test_utils as tutils


"""Tests the sub functions"""
landmarker_model = lm.PoseLandmarkerModel().model

test_image_full_path = config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir" / "Boxing" / "boxing1.jpg"

test_image_rgb = utils.read_image_at_path(str(test_image_full_path))
real_OneSetOfLandmarks = lm.convert_image_to_landmarks(landmarker_model, test_image_rgb)

zero_OneSetOfLandmarks = [0.0] * (33 * 4)    # MediaPipe predicts for 33 Landmark Components. Each Component is composed of x, y, z, and presence 


def test_extract_hip_midpoint_zero_edge_case():
    # Check the function can handle a case where both hips have coordinates (0, 0, 0)

    assert lm.extract_hip_midpoint(zero_OneSetOfLandmarks) == (0.0, 0.0, 0.0)

def test_extract_hip_midpoint_tiny_values():
    ## Extremely small values
    ### Build the OneSetOfLandmarks
    tiny_coord = (1e-4, 1e-4, 1e-4)

    tiny_hips_OneSetOfLandmarks = copy_and_set_coords_of_component(
        tiny_coord,
        "left_hip",
        zero_OneSetOfLandmarks)
    
    tiny_hips_OneSetOfLandmarks = copy_and_set_coords_of_component(
        tiny_coord,
        "right_hip",
        tiny_hips_OneSetOfLandmarks)
        
    ### Check for the midpoint.
    assert lm.extract_hip_midpoint(tiny_hips_OneSetOfLandmarks) == pytest.approx(tiny_coord)

def test_extract_hip_midpoint_for_extremely_large_values():
    ## Extremely large values
    ### Build the OneSetOfLandmarks
    large_coord = (1e4, 1e4, 1e4)

    large_hips_OneSetOfLandmarks = copy_and_set_coords_of_component(
        large_coord,
        "left_hip",
        zero_OneSetOfLandmarks)
    
    large_hips_OneSetOfLandmarks = copy_and_set_coords_of_component(
        large_coord,
        "right_hip",
        large_hips_OneSetOfLandmarks)
    
    ### Check for the midpoint.
    assert lm.extract_hip_midpoint(large_hips_OneSetOfLandmarks) == pytest.approx(large_coord)

def test_extract_landmark_component_xyz():
    # The benchmark values were gotten by running the commented test_print() function
    assert lm.extract_landmark_component_xyz(real_OneSetOfLandmarks, "nose") == pytest.approx((0.049, -0.63, -0.17), rel=1e-1)

def test_raises_error_extract_landmark_component_xyz():
    """Check that an error is raised for an invalid landmark component name not in lm.PoseLandmarkerModel.map_of_keypoint_names_to_idx
    """
    with pytest.raises(KeyError):
        lm.extract_landmark_component_xyz(real_OneSetOfLandmarks, "hdhhd")

def test_component_idx_in_OneSetOfLandmarks():
    """Test that it correctly extracts the indexes for each component name"""
    map = lm.map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks
    component_names = map.keys()

    ground_truth = []
    function_output = []

    for name in component_names:
        x_idx = map[name] * 4   # Since each landmark component is a flattened x, y, z and presence
        z_idx = x_idx + 2
        ground_truth.append((x_idx, z_idx))

        # The results of the function
        func_xz_idx = lm.component_idx_in_OneSetOfLandmarks(name)
        function_output.append(func_xz_idx)
        
    assert function_output == ground_truth


def copy_and_set_coords_of_component(coords: tuple[float, float, float], component_name: str, an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> defs.OneSetOfLandmarks:
    """Returns a copy of the OneSetOfLandmarks, with the coordinates of the desired component set to the input coordinates."""
    x_left_hip_idx, z_left_hip_idx = lm.component_idx_in_OneSetOfLandmarks(component_name)

    an_OneSetOfLandmarks[x_left_hip_idx:z_left_hip_idx+1] = coords
    
    return an_OneSetOfLandmarks

def test_compute_midpoint_of_zero_inputs():
    origin = (0, 0, 0)
    assert lm.compute_midpoint(origin, origin) == origin

def test_compute_midpoint_normal():
    coord_1 = 1, 0.5, 0.3
    coord_2 = 2, 0.4, 0.1

    assert lm.compute_midpoint(coord_1, coord_2) == (1.5, 0.45, 0.2)

def test_extract_shoulder_coords_zeros_OneSetOfLandmarks():
    """Especially useful in unusual poses."""
    assert lm.extract_shoulder_coords(zero_OneSetOfLandmarks) == (0, 0, 0, 0, 0, 0)

def test_extract_shoulder_coords():
    """The common use case."""
    assert lm.extract_shoulder_coords(real_OneSetOfLandmarks) == pytest.approx((0.19, -0.48, 0.0017, -0.13, -0.47, -0.049), rel=1e-1)

def test_error_noone_convert_image_to_landmarks():
    # No person in the image. An error should be raised.
    no_person_image = utils.read_image_at_path(tc.image_with_no_person)
    with pytest.raises(ValueError):
        lm.convert_image_to_landmarks(landmarker_model, no_person_image)
    
def test_error_multiple_persons_convert_image_to_landmarks():
    # Multiple persons in the image. An error should be raised
    multiple_person_image = utils.read_image_at_path(tc.image_with_multiple_persons)

    with pytest.raises(ValueError):
        lm.convert_image_to_landmarks(landmarker_model, multiple_person_image)

# --- Valid for only one person images.
# Tests for the class of poses that the system can generate image for.
def sample_from_directory_and_convert_to_OneSetOfLandmarks(PoseLandmarker, directory: str, sample_size: int = -1) -> dict[str, defs.OneSetOfLandmarks]:
    """Randomly samples n images from each pose class within the directory and converts to OneSetOfLandmarks\n
    If population size < sample_size, it takes the entire population as the sample.\n
    If the sample_size parameter is not passed, samples one-fourth of the images within each pose class.\n
    Output: Dict[image_path, OneSetOfLandmarks]"""

    path_of_each_image, label_of_each_image = utils.image_paths_and_labels_in_directory(directory)

    # Take the image samples
    sample_of_images_names = tutils.sample(path_of_each_image, sample_size)

    # Map each image [dir] in the sample to its OneSetOfLandmarks 

    an_AllLandmarks, an_valid_pose_labels = utils.convert_image_paths_to_OneSetOfLandmarks(path_of_each_image, label_of_each_image, landmarker_model)

    # Add the key-value pair: image_path, OneSetOfLandmarks
    dict_of_image_path_and_OneSetOfLandmarks = dict(zip(sample_of_images_names, an_AllLandmarks))

    return dict_of_image_path_and_OneSetOfLandmarks

# For each pose class.
## Sample n images from it
OneSetOfLandmarks_of_sample_images = sample_from_directory_and_convert_to_OneSetOfLandmarks(landmarker_model, tc.valid_dir)

@pytest.mark.parametrize(
    "sample_image_path, an_OneSetOfLandmarks",
    OneSetOfLandmarks_of_sample_images.items(),
    ids=OneSetOfLandmarks_of_sample_images.keys()   # The path of the image. Helpful in understanding failure cases.
)
def test_valid_convert_image_to_landmarks(sample_image_path, an_OneSetOfLandmarks):
    ## Assert that each image gets a valid OneSetOfLandmark
    ## If not, print the name of the image.
    print(f"Testing image: {sample_image_path}")
    
    assert isValidOneSetOfLandmarks(an_OneSetOfLandmarks)

def isValidOneSetOfLandmarks(an_OneSetOfLandmarks) -> bool:
    """Returns True if the input is a defs.OneSetOfLandmarks\n
    Returns False otherwise"""

    # Is it a list?
    if not isinstance(an_OneSetOfLandmarks, list):
        return False
    # Is the length = 33 * 4. Assumes it is teh MediaPipe model. See defs.OneSetOfLandmarks?
    if not (len(an_OneSetOfLandmarks) == 132):
        return False
    # Is each element a numerical number?
    for element in an_OneSetOfLandmarks:
        if not isinstance(element, (float, int)): return False

    return True
    


# def test_print():
#     for i in range(len(real_OneSetOfLandmarks)):
#         print(f"{i}: {real_OneSetOfLandmarks[i]}")

class Test_compute_midpoint:
    # Build our test case
    ## Parameterize to enable generation of all possible combinations of test cases.
    ### zero, -ve, +ve, and large, medium and small for each
    
    @pytest.mark.parametrize("first_x_y_z", 
                             [0, 
                              1e4, 0.1, 1e-4, 
                              -1e4, -0.1, -1e-4])
    @pytest.mark.parametrize("second_x_y_z", 
                             [0, 
                              1e4, 0.1, 1e-4, 
                              -1e4, -0.1, -1e-4])
    def test_all_cases(self, first_x_y_z, second_x_y_z):
        """Tests all possible combinations, to test all use cases."""

        # Build the coordinates.
        first_coord = (first_x_y_z,) * 3
        second_coord = (second_x_y_z,) * 3

        # Create the correct midpoint, as a tuple (x, y, z)
        value_for_each_axis = (first_x_y_z + second_x_y_z) / 2
        correct_midpoint = (value_for_each_axis,) * 3

        # Test, within some degree of error to account for floating point inaccuracies
        func_output = lm.compute_midpoint(first_coord, second_coord)

        assert func_output == pytest.approx(correct_midpoint)

class Test_extract_shoulder_midpoint:
    # Build our test case 
    a_valid_OneSetOfLandmarks = [0.0] * (33 * 4)
    
    l_x_idx, l_z_idx = lm.component_idx_in_OneSetOfLandmarks("left_shoulder")

    r_x_idx, r_z_idx = lm.component_idx_in_OneSetOfLandmarks("right_shoulder")

    # Parameterize to enable generation of all possible combinations of test cases.
    ## zero, -ve, +ve, and large, medium and small for each
    
    @pytest.mark.parametrize("left_shoulder_value", 
                             [0, 
                              1e4, 0.1, 1e-4, 
                              -1e4, -0.1, -1e-4])
    @pytest.mark.parametrize("right_shoulder_value", 
                             [0, 
                              1e4, 0.1, 1e-4, 
                              -1e4, -0.1, -1e-4])
    def test_all_cases(self, left_shoulder_value, right_shoulder_value):
        """Tests all possible combinations, to test all use cases."""

        # Modify the valid OneSetOfLandmarks for this test case.
        self.a_valid_OneSetOfLandmarks[self.l_x_idx: (self.l_z_idx + 1)] = [left_shoulder_value] * 3

        self.a_valid_OneSetOfLandmarks[self.r_x_idx: (self.r_z_idx + 1)] = [right_shoulder_value] * 3

        # Create the correct midpoint, as a tuple (x, y, z)
        value_for_each_axis = (left_shoulder_value + right_shoulder_value) / 2
        correct_midpoint = (value_for_each_axis,) * 3

        # Test, within some degree of error to account for floating point inaccuracies
        func_output = lm.extract_shoulder_midpoint(self.a_valid_OneSetOfLandmarks)

        assert func_output == pytest.approx(correct_midpoint)

class Test_extract_hip_midpoint:
    # Build our test case 
    a_valid_OneSetOfLandmarks = [0.0] * (33 * 4)
    
    l_x_idx, l_z_idx = lm.component_idx_in_OneSetOfLandmarks("left_hip")

    r_x_idx, r_z_idx = lm.component_idx_in_OneSetOfLandmarks("right_hip")

    # Parameterize to enable generation of all possible combinations of test cases.
    ## zero, -ve, +ve, and large, medium and small for each
    
    @pytest.mark.parametrize("left_hip_value", 
                             [0, 
                              1e4, 0.1, 1e-4, 
                              -1e4, -0.1, -1e-4])
    @pytest.mark.parametrize("right_hip_value", 
                             [0, 
                              1e4, 0.1, 1e-4, 
                              -1e4, -0.1, -1e-4])
    def test_all_cases(self, left_hip_value, right_hip_value):
        """Tests all possible combinations, to test all use cases."""

        # Modify the valid OneSetOfLandmarks for this test case.
        self.a_valid_OneSetOfLandmarks[self.l_x_idx: (self.l_z_idx + 1)] = [left_hip_value] * 3

        self.a_valid_OneSetOfLandmarks[self.r_x_idx: (self.r_z_idx + 1)] = [right_hip_value] * 3

        # Create the correct midpoint, as a tuple (x, y, z)
        value_for_each_axis = (left_hip_value + right_hip_value) / 2
        correct_midpoint = (value_for_each_axis,) * 3

        # Test, within some degree of error to account for floating point inaccuracies
        func_output = lm.extract_hip_midpoint(self.a_valid_OneSetOfLandmarks)

        assert func_output == pytest.approx(correct_midpoint)

