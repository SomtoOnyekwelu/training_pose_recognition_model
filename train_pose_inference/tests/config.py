"""Defines several key variables for tests, so as to make refactoring easier.
One such change is renaming the tests directory """
import pathlib

import pose_estimation_rough.config as config

ROOT_OF_TEST_DIR = pathlib.Path(__file__).parent.resolve()

valid_dir = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir")

image_with_no_person = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "no_person_in_this_image.jpg")

image_with_multiple_persons = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "multiple_people_in_an_image.jpeg")

image_with_standing_person = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir" / "Standing" / "IMG_20250919_131936_875.jpg")

image_with_person_lying_down = str(config.PROJECT_ROOT / "train_pose_inference" / "tests" / "samples" / "valid_dir" / "Lying Down" / "IMG_20250919_132212_707.jpg")