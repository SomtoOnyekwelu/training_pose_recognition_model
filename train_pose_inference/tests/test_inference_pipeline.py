"""Tests all clases in the corresponding file of the src folder."""

import pytest
import os
import src.inference_pipeline as ip
import src.utils as utils
from src.train import PoseClassifierModel
from src.landmarker_model import PoseLandmarkerModel


class Test_PoseInferencePipeline:
    """Tests the PoseInferencePipeline class"""

    class Test_predict_image:
        """Tests the predict_image function."""

        @pytest.fixture(scope="class")
        def make_example_PoseInferencePipeline(self):
            # Fixture creation
            a_valid_directory = r"tests\samples\valid_dir_three_subdir"

            example_PoseClassifierModel = PoseClassifierModel(a_valid_directory, 0.25)

            example_PoseLandmarkerModel = PoseLandmarkerModel(use_case="production")

            example_PoseInferencePipeline = ip.PoseInferencePipeline(example_PoseClassifierModel, example_PoseLandmarkerModel)

            return example_PoseInferencePipeline

        def test_no_person_in_image(self, make_example_PoseInferencePipeline):
            """Tests the case where there is no person in the image.
            
            Should return an empty list."""

            path_to_no_person_image = r"tests\samples\no_person_in_this_image.jpg"
            
            no_person_image = utils.read_image_at_path(path_to_no_person_image)

            result = make_example_PoseInferencePipeline.predict_image(no_person_image)

            assert result == []

        path_to_all_one_person_images = r"tests\samples\one_person_images_different_angles_distances_sizes"
        @pytest.mark.parametrize(
                "path_to_an_one_person_image",
                utils.list_dir_as_abs_paths(path_to_all_one_person_images)
        )

        # ---- SINGLE PERSON CASE
        # Simulates a simplified version of demo day, where only one person is in front of the camera. The person would be one of each of the below:
        # - of different sizes (wide, slim, average)
        # - of different heights (short, tall, average)
        # - at different distances (too close, too far, average [camera sees head and toe, and the person is about height of frame.]) 
        # - having different angles in front of the camera (front, back, three-quater, side.)
        def test_one_person_in_image(self, make_example_PoseInferencePipeline, path_to_an_one_person_image):
            """Tests the case where there is only one person in the image.
            Should return a singleton list."""

            one_person_image = utils.read_image_at_path(path_to_an_one_person_image)

            result = make_example_PoseInferencePipeline.predict_image(one_person_image)

            # Check for correct length and the correct element datatype.
            assert self._correct_count_of_PlotLabelToDraw(result, 1) and self._are_all_elements_PlotLabelToDraw(result)

        # ---- MULTIPLE PERSON CASE
        # Simulates demo day where multiple people would be in front of the camera, where they would be:
        # - of different sizes
        # - at different distances and 
        # - having different angles in front of the camera.
        path_to_all_three_person_images = r"tests\samples\three_person_images_different_angles_distances_sizes"
        @pytest.mark.parametrize(
                "path_to_a_three_person_image",
                utils.list_dir_as_abs_paths(path_to_all_three_person_images)
        )

        def test_three_persons_in_image(self, make_example_PoseInferencePipeline, path_to_a_three_person_image):
            """Tests the case where there are three persons in the image. Represents the case for multiple people.
            Should return a list of three PoseLabelToDraw."""

            three_person_image = utils.read_image_at_path(path_to_a_three_person_image)

            result = make_example_PoseInferencePipeline.predict_image(three_person_image)

            # Check for correct length and the correct element datatype.

            assert self._correct_count_of_PlotLabelToDraw(result, 3) and self._are_all_elements_PlotLabelToDraw(result)
        
        def _correct_count_of_PlotLabelToDraw(self, a_list_of_PoseLabelToDraw: list[ip.PoseLabelToDraw], expected_length: int) -> bool:
            """Returns True if the count of the list[PoseLabelToDraw] == expected_length.
            Returns False otherwise."""

            if len(a_list_of_PoseLabelToDraw) == expected_length:
                return True
            
            return False
        
        def _are_all_elements_PlotLabelToDraw(self, a_list: list) -> bool:
            """Returns True if all elements of the list are PoseLabelToDraws."""

            if a_list == []:
                return False

            for element in a_list:
                if not isinstance(element, ip.PoseLabelToDraw):
                    return False

            return True
