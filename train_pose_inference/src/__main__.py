"""
Creates the Pose Classifier model by training it on the path, then exports the direct_model and its pose label encoder, so that they can be loaded for use in production by a PoseInferencePipeline object.
"""

import xgboost as xgb
import joblib
import os
from sklearn.preprocessing import LabelEncoder

from pose_estimation_rough.train_pose_inference.src.train import PoseClassifierModel

# Using the config, train the pose classifier model.
# Train the pose classifier model
# Save the model to the project path

class Training_And_Save_App:
    # A training->save-model run

    def __init__(self, training_directory: str, save_directory: str, to_print_model_evaluation: bool = False):
        """Initilizes the "directory for source images" attribute for the class\n
        Inputs:
        - training_directory is the path to the Directory (as defined in data_defs) of images that would be used in training the direct_model.
        
        - save_directory is the path where the direct_model and its label encoder would be saved for use in downstream tasks."""

        # Error Handling: Non-existent Source Directory
        if not os.path.exists(training_directory):
            raise ValueError(f"This directory at this path does not exist: {training_directory}")

        # Error Handling: Invalid Save Path
        if not os.path.exists(save_directory):
            raise ValueError(f"This save path does not exist: {save_directory}")

        self.source_directory = training_directory
        self.save_directory = save_directory

        self.test_ratio = 0.25 # Fraction of the dataset that is kept aside for testing.

        # Creates the XGBoost classifier model, packaged with its label encoder.
        print("Model training has started.")
        pose_classifier_and_pose_label_encoder = self.pose_classifier_model_with_label_encoder = self.produce_model(to_print_model_evaluation)

        # Saves the direct_model and its label encoder for later use by the main client side program.
        direct_model = pose_classifier_and_pose_label_encoder.xgbClassifier_model

        pose_label_encoder = pose_classifier_and_pose_label_encoder.pose_label_encoder

        self.save_direct_model_and_label_encoder(save_directory, direct_model, pose_label_encoder)
        print(f"The model and its label encoder has been saved successfully at {save_directory}")


    def produce_model(self, to_print_model_evaluation: bool) -> PoseClassifierModel:
        """Fits a model on the landmarks derived from the directory images
        Then, save it at the save_path"""

        # Fits the model on the directroy
        pose_classifier_model_with_label_encoder = PoseClassifierModel(self.source_directory, to_print_evaluation=to_print_model_evaluation)

        # Returns the model
        return pose_classifier_model_with_label_encoder

    # def _create_Pose_Inference_Pipeline(self, model: PoseClassifierModel) -> PoseInferencePipeline:
    #     """Creates a model package for the pose inference model trained on the directory of images.
    #     This package would include the:
    #      - A single API call to pass in an image frame and get a list[PoseLabelToDraw].
    #      - XGBoost Classification model, 
    #      - label encoder, 
    #      - preprocessing and augumentation pipeline, and  """
        
    #     return PoseInferencePipeline()
    
    def save_direct_model_and_label_encoder(self, save_path: str, direct_model: xgb.XGBClassifier, pose_label_encoder: LabelEncoder) -> None:
        """Saves the direct_model and label encoder as files for later use at the save directory.
        """        
        # Save the direct_model to the path.
        direct_model_save_path = os.path.join(save_path, r"direct_model.pkl")

        joblib.dump(direct_model, direct_model_save_path)

        # Save the pose_label_encoder to the path.
        pose_label_encoder_save_path = os.path.join(save_path, r"pose_label_encoder.pkl")

        joblib.dump(pose_label_encoder, pose_label_encoder_save_path)

# ---- MAIN PROGRAM LOGIC ----
import config

# Path variables
save_path_for_direct_model_and_pose_label_encoder = config.CLASSIFICATION_MODEL_DIR
training_images_dir = config.PROJECT_ROOT / "train_pose_inference" / "pose_images"

# Train the direct_model on the directory of images
# Then, save the model to the save_directory.
if __name__ == "__main__":
    model_package = Training_And_Save_App(str(training_images_dir), str(save_path_for_direct_model_and_pose_label_encoder), True)