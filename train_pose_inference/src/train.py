# Provides functions to train a model
# Aim: Return a model trained on a directory of images.

# <USE LABEL ENCODER TO CONVERT LABELS TO NUMERICALS>
# RETURN ENCODER FOR USE TO TRANSATE FROM NUMERICAL TO STRING AT INFERENCE done1: added to POseCLassifierModel

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from pose_estimation_rough.train_pose_inference.src.processing import ProcessedLandmarks
from pose_estimation_rough.train_pose_inference.src import data_defs as defs

class PoseClassifierModel:
    """
    An encapsulator of the training steps for the xgb_model.\n
    Also, contains the direct_model and its label encoder as attributes.
    """
    def __init__(self, directory: str, test_ratio: float = 0.25, to_print_evaluation: bool = False):
        """
        Initializes a model trained on the input directory
        """
        assert 0 < test_ratio < 1

        self.directory = directory

        self.xgbClassifier_model, self.pose_label_encoder = self._get_model(directory, test_ratio, to_print_evaluation)
    
    def get_direct_model(self) -> xgb.XGBClassifier:
        """Retrieves the direct model of the object."""

        return self.xgbClassifier_model
    
    def get_pose_label_encoder(self) -> LabelEncoder:
        """Retrieves the pose label encoder of the object."""

        return self.pose_label_encoder
    
    def _get_model(self, directory: str, test_ratio: float, to_print_evaluation: bool) -> tuple[xgb.XGBClassifier, LabelEncoder]:
        """Returns a model fitted to the data at directory"""

        assert 0 < test_ratio < 1

        # Get the landmarks and output
        dataset_obj = ProcessedLandmarks(directory)

        ## Return the landmark features and the pose classes
        landmarks = dataset_obj.unprocessed_AllLandmarks

        pose_classes = dataset_obj.pose_labels

        # Train the model on the dataset
        ## Prepare the dataset
        encoded_pose_labels, pose_label_encoder = self._encode_labels_and_return_encoder(pose_classes)

        X_train, X_test, y_train, y_test = self._split_dataset(landmarks, encoded_pose_labels, test_ratio)

        # DEBUG
        print(f"Size of Training dataset: {len(X_train)}")
        print(f"Size of Test dataset: {len(X_test)}")
        print(f"Overall size of dataset: {len(encoded_pose_labels)}")

        model = self._train_model(X_train, y_train, pose_label_encoder)

        # Print the evaluation report for the model.
        if to_print_evaluation:
            evaluate_model(model, X_test, y_test, pose_label_encoder)

        return model, pose_label_encoder
    
    def _encode_labels_and_return_encoder(self, all_pose_labels: defs.PoseLabels) -> tuple[list[int], LabelEncoder]:
        """Encodes the labels to be numerical, and returns the label encoder needed to decode the model's outputs."""

        label_encoder = LabelEncoder()

        ## Add the labels to the internal mapping of the LabelEncoder and convert the pose labels to their encoded form.
        encoded_pose_labels = label_encoder.fit_transform(all_pose_labels)
        
        return encoded_pose_labels, label_encoder

    def _split_dataset(self, landmarks: np.ndarray, encoded_pose_labels: np.ndarray,test_ratio: float) -> tuple:
        """
        Split the dataset into the x_train, x_test, y_train, y_test, label_encoder.\n
        The size of the test split test_ratio * dataset_size.
        Also, return a label encoder.
        """
        ## The dataframe of the dataset's AllLandmarks
        X = pd.DataFrame(landmarks)

        # Do the split
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_pose_labels, test_size=test_ratio, random_state=30, stratify=encoded_pose_labels
        )

        # DEBUG
        print(f"Type of X_train: {type(X_train)}")

        return X_train, X_test, y_train, y_test
    
    def _train_model(self, X_train, y_train, pose_label_encoder) -> xgb.XGBClassifier:
        """Returns an XGBoost model, trained on the input dataset"""
        # Initialize the model
        xgboost_model = xgb.XGBClassifier(
            objective = "multi:softmax",
            num_class = len(pose_label_encoder.classes_),
            use_label_encoder = False,
            eval_metric = "mlogloss"
        )

        # Train the model
        print("Training the XGBoost model...")
        xgboost_model.fit(X_train, y_train)
        print("Model training completed")

        return xgboost_model
    
def evaluate_model(model: xgb.XGBClassifier, X_test, y_test, label_encoder: LabelEncoder) -> None:
    """Evaluates the performance of the model based on:
    Accuracy,
    Loss and
    Confusion Matrix."""

    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy * 100:.2f}%")

    # Compute classification report, including confusion matrix
    classification_performance_info = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print(f"""Model's Classification Performance Report\n{classification_performance_info}""")
