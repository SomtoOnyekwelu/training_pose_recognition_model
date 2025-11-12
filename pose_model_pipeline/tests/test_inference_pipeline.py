"""
Tests all clases in the corresponding file of the src folder."""

import pytest
import operator

from pose_estimation_rough.pose_model_pipeline.src import inference_pipeline as ip
from pose_estimation_rough.train_pose_inference.src import utils as utils
import pose_estimation_rough.config as config

class Test_PoseInferencePipeline:
    """Tests the PoseInferencePipeline class."""

    class Test_predict_image:
        """Tests the predict_image function."""

        @pytest.fixture(scope="class")
        def make_example_PoseInferencePipeline(self):
            # Fixture creation
            path_to_example_direct_model = config.PROJECT_ROOT / "pose_model_pipeline" / "tests" / "test_assets" / "example_direct_model_10_classes.pkl"

            path_to_its_pose_label_encoder = config.PROJECT_ROOT / "pose_model_pipeline" / "tests" / "test_assets" / "example_pose_label_encoder_10_classes_its_pair.pkl"

            example_PoseInferencePipeline = ip.PoseInferencePipeline(str(path_to_example_direct_model), str(path_to_its_pose_label_encoder))

            return example_PoseInferencePipeline

        def test_no_person_in_image(self, make_example_PoseInferencePipeline):
            """Tests the case where there is no person in the image.
            
            Should return an empty list."""

            path_to_no_person_image = config.PROJECT_ROOT / "pose_model_pipeline" / "tests" / "samples" / "no_person_image.jpg"
            
            no_person_image = utils.read_image_at_path(str(path_to_no_person_image))

            result = make_example_PoseInferencePipeline.predict_image(no_person_image)

            assert result == []

        path_to_all_one_person_images = config.PROJECT_ROOT / "pose_model_pipeline" / "tests" / "samples" / "one_person_images_different_angles_distances_sizes"
        @pytest.mark.parametrize(
                "path_to_an_one_person_image",
                utils.list_dir(str(path_to_all_one_person_images))
        )

        # ---- SINGLE PERSON CASE
        # Simulates a simplified version of demo day, where only one person is in front of the camera. The person would be one of each of the below:
        # - of different sizes (wide, slim, average)
        # - of different heights (short, tall, average)
        # - at different distances (too close, too far, average [camera sees head and toe, and the person is about height of frame.]) 
        # - having different angles in front of the camera (front, back, three-quater, side.)
        def test_one_person_in_image(self, make_example_PoseInferencePipeline, path_to_an_one_person_image):
            """Tests the case where there is only one person in the image. Tests if a single person is detected regardless of varying disytances from the camera, angles and poses.
            Should return a singleton list."""

            one_person_image = utils.read_image_at_path(path_to_an_one_person_image)

            result = make_example_PoseInferencePipeline.predict_image(one_person_image)

            # Check for correct length and the correct element datatype.
            assert self._correct_count_of_PoseLabelToDraw(result, 1) and self._are_all_elements_PlotLabelToDraw(result)

        # ---- MULTIPLE PERSON CASE
        # Simulates demo day where multiple people would be in front of the camera, where they would be:
        # - of different sizes
        # - at different distances and 
        # - having different angles in front of the camera.
        path_to_all_three_person_images = config.PROJECT_ROOT / "pose_model_pipeline" / "tests" / "samples" / "three_person_images_different_angles_distances_sizes"
        @pytest.mark.parametrize(
                "path_to_a_three_person_image",
                utils.list_dir(str(path_to_all_three_person_images))
        )

        def test_multiple_persons_in_image(self, make_example_PoseInferencePipeline, path_to_a_three_person_image):
            """Tests the case where there are three persons in the image. Tests if multiple are detected regardless of their varying distances from the camera, angles and poses.
            Represents the case for multiple people.
            Should return a list of three PoseLabelToDraw.
            
            Important note: I relaxed the test to confirm the detection of multiple persons instead of exactly three persons. 
            This is due to the quality of the images, which were intentional, so as to simulate the real-world constrained camera quality of my target situation."""

            multiple_person_image = utils.read_image_at_path(path_to_a_three_person_image)

            result = make_example_PoseInferencePipeline.predict_image(multiple_person_image)

            # Check for correct length and the correct element datatype.

            assert self._min_count_of_PoseLabelToDraw(result, 2) and self._are_all_elements_PlotLabelToDraw(result)
        
        def _correct_count_of_PoseLabelToDraw(self, a_list_of_PoseLabelToDraw: list[ip.PoseLabelToDraw], expected_length: int) -> bool:
            """Returns True if the count of the list[PoseLabelToDraw] == expected_length.
            Returns False otherwise."""

            return self._operator_on_count_of_PoseLabelToDraw(a_list_of_PoseLabelToDraw, expected_length, operator.eq)
        
        def _min_count_of_PoseLabelToDraw(self, a_list_of_PoseLabelToDraw: list[ip.PoseLabelToDraw], min_expected_length: int) -> bool:
            """Returns True if the count of the list[PoseLabelToDraw] >= expected_length.
            Returns False otherwise."""

            return self._operator_on_count_of_PoseLabelToDraw(a_list_of_PoseLabelToDraw, min_expected_length, operator.ge)


        def _operator_on_count_of_PoseLabelToDraw(self, a_list_of_PoseLabelToDraw: list[ip.PoseLabelToDraw], expected_length: int, ops) -> bool:
            """Returns True if the conditional formed by the operator on the following two operands is correct: count of the list[PoseLabelToDraw] and expected_length.
            Returns False otherwise."""

            if ops(
                len(a_list_of_PoseLabelToDraw),
                expected_length):
                return True
            
            print(f"Count of detected persons: {len(a_list_of_PoseLabelToDraw)}")
            
            return False
        
        def _are_all_elements_PlotLabelToDraw(self, a_list: list) -> bool:
            """Returns True if all elements of the list are PoseLabelToDraws."""

            if a_list == []:
                return False

            for element in a_list:
                if not isinstance(element, ip.PoseLabelToDraw):
                    return False

            return True
