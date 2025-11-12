import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import pathlib

from pose_estimation_rough.pose_model_pipeline.src.data_classes.pose_label_to_draw import PoseLabelToDraw
from pose_estimation_rough.pose_model_pipeline.src.pose_classifier_model_for_use import PoseClassifierModel_ForUse # A trained model, to be instandtiated with the file path strings to xgb_model pkl and label_encoder.
from pose_estimation_rough.train_pose_inference.src import landmarker_model as lm
from pose_estimation_rough.train_pose_inference.src.landmarker_model import PoseLandmarkerModel
from pose_estimation_rough.train_pose_inference.src import data_defs as defs

class PoseInferencePipeline:
    """
    This represents the whole inference architecture for production.
    It enables an image to be passed, and a list[PoseLabelToDraw] to be returned.
    Furthermore, it ensures maximum accuracy during inference by replicating the training conditions in terms of the below:
    - Pose landmarker model,
    - Conversion of a person in an image to a representative OneSetOfLandmarks
    ### Later, include:
    - Normalization script for OneSetOfLandmarks, and 
    - Augumentation Script 
    """
    def __init__(self, path_to_direct_model: str, path_to_pose_label_encoder: str):
        """
        Initializes the pipeline with a:
        - PoseClassifierModel_ForUse
        Creates the PoseLandmarker model too. Same as production.
        """
        loaded_direct_model = load_saved_direct_model(path_to_direct_model)

        its_pose_label_encoder = load_saved_label_encoder(path_to_pose_label_encoder)

        self.loaded_pose_classifier_model = PoseClassifierModel_ForUse(loaded_direct_model, its_pose_label_encoder)

        self.pose_landmarker_model = PoseLandmarkerModel("production")

    def predict_image(self, image: np.ndarray) -> list[PoseLabelToDraw]:
        """
        Returns a list of PoseLabelToDraw, where each identifies the pose of each person in the image and the coordinate to draw them in.\n

        The image should be an RGB image, for higher accuracy to simulate training conditions.\n

        The (x, y) of the PoseLabelToDraw is the mid-torso of the corresponding person.

        Computes for each PoseLabelToDraw the:
        - predicted pose_label of the corresponding person.
        - the probability of the predicted pose.
        - placement position of the label on the image.
        """

        # Error Handling: No person detected in the image
        try:
            all_OneSetOfLandmarks = self.pose_landmarker_model.production_convert_image_to_OneSetOfLandmarks(image)
        except ValueError as e:
            if "No person was detected in this image." in str(e):
                return []
            else:
                raise e

        # Predict the pose labels and probabilities.
        predicted_pose_labels, prob_of_predictions = self.loaded_pose_classifier_model.predict_pose_labels(all_OneSetOfLandmarks)

        # Extract the drawing coordinates for each PoseLabel
        # Task: Make the OneSetOfLandmarks class, add this normimg as attribute along with standard one, remove this. It causes two calls to model.
        all_NormImgOneSetOfLandmarks = extract_all_landmarks_as_NormImgOneSetOfLandmarks(self.pose_landmarker_model, image)

        all_drawing_coords = extract_drawing_coords_for_each_PoseLabelToDraw(all_NormImgOneSetOfLandmarks)

        # Package.
        label_prob_coord_tuples = zip(predicted_pose_labels, prob_of_predictions, all_drawing_coords)

        # Create the list of PlotLabelToDraw
        all_PoseLabelToDraw = convert_each_paired_info_to_PoseLabelToDraw(label_prob_coord_tuples)        
        
        return all_PoseLabelToDraw

def extract_all_landmarks_as_NormImgOneSetOfLandmarks(pose_landmarker_model: PoseLandmarkerModel, image: np.ndarray) -> list[defs.NormImgOneSetOfLandmarks]:
    """Extracts all landmarks of the detected persons in the image, wrt the image dimensions."""

    several_NormImgOneSetOfLandmarks = pose_landmarker_model.get_norm_OneSetOfLandmarks_wrt_img_coords_for_PoseLabelToDraw(image)

    return several_NormImgOneSetOfLandmarks

def load_saved_label_encoder(file_path: str) -> LabelEncoder:
    """Returns the LabelEncoder object saved at file_path."""

    try:
        loaded_pose_label_encoder = load_saved_files(file_path)
    except Exception as e:
        raise ValueError(f"This path to the saved pose_label_encoder does not exist: {file_path}")
    
    if not isinstance(loaded_pose_label_encoder, LabelEncoder):
        raise ValueError(f"This path does not lead to a saved pose_label_encoder {file_path}")

    return loaded_pose_label_encoder

def load_saved_direct_model(file_path: str) -> xgb.XGBClassifier:
    """Returns the direct_model object saved at file_path."""

    try:
        loaded_direct_model = load_saved_files(file_path)
    except Exception as e:
        raise ValueError(f"This path to the saved direct model does not exist: {file_path}")
    
    if not isinstance(loaded_direct_model, xgb.XGBClassifier):
        raise ValueError(f"This path does not lead to a saved direct model {file_path}")

    return loaded_direct_model


def extract_drawing_coords_for_each_PoseLabelToDraw(some_NormImgOneSetOfLandmarks: list[defs.NormImgOneSetOfLandmarks]) -> list[tuple[int, int]]:
    """Returns a list of (x, y) coordinates, corresponding to the torso midpoint of each NormImgOneSetOfLandmarks in the passed parameter."""
    if not some_NormImgOneSetOfLandmarks:
        return []

    all_drawing_coords = []

    for a_NormImgOneSetOfLandmark in some_NormImgOneSetOfLandmarks:
        xy_of_torso_center = compute_torso_center(a_NormImgOneSetOfLandmark)

        all_drawing_coords.append(xy_of_torso_center)

    return all_drawing_coords

def load_saved_files(file_path: str):
    """Loads the object saved at the file_path"""
    p = pathlib.Path(file_path)
    if not p.exists():
        raise ValueError("The file_path does not exist.")

    loaded_file = joblib.load(file_path)

    return loaded_file 

def compute_torso_center(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> tuple[int, int]:
    """Returns (x, y) of torso's center.
    No need for z beacause we are drawing on 2D. 
    Task: Unless for scaling of the text."""
    
    x, y, z = lm.extract_torso_midpoint(an_OneSetOfLandmarks)

    return x, y

def convert_each_paired_info_to_PoseLabelToDraw(label_prob_coord_tuples) -> list[PoseLabelToDraw]:
    """Converts each tuple to a PoseLabelToDraw"""

    list_of_PoseLabelToDraw = []

    for info_tuple in label_prob_coord_tuples:
        a_PoseLabelToDraw = convert_info_tuple_to_PoseLabelToDraw(info_tuple)

        list_of_PoseLabelToDraw.append(a_PoseLabelToDraw)

    return list_of_PoseLabelToDraw

def convert_info_tuple_to_PoseLabelToDraw(info_tuple: tuple[str, float, tuple[int, int]]) -> PoseLabelToDraw:
    """
    Converts the info_tuple to a PoseLabelToDraw.
    """
    predicted_pose_label = info_tuple[0]
    probability_of_predicted_pose = info_tuple[1]
    coordinate_for_drawing_pose_label_on_image = info_tuple[2]

    return PoseLabelToDraw(
        predicted_pose_label, 
        probability_of_predicted_pose,
        coordinate_for_drawing_pose_label_on_image
    )
