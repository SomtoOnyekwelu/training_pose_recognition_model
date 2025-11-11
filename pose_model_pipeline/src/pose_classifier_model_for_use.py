import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from ...train_pose_inference.src import data_defs as defs

class PoseClassifierModel_ForUse:
    """Instance of an already-trained direct_model and its label encoder"""

    def __init__(self, a_direct_model: xgb.XGBClassifier, its_pose_label_encoder: LabelEncoder):
        """Initializes the object"""

        self.direct_model = a_direct_model

        self.pose_label_encoder = its_pose_label_encoder

    def predict_pose_labels(self, some_OneSetOfLandmarks: list[defs.OneSetOfLandmarks]) -> tuple[defs.PoseLabels, list[float]]:
        """Returns the predicted Pose Label and probability of correctness for each OneSetOfLandmarks in the entry."""

        pose_label_encoder = self.pose_label_encoder

        batch_of_some_OneSetOfLandmarks_np = np.array(some_OneSetOfLandmarks)

        prediction_of_probabilities = self.direct_model.predict_proba(batch_of_some_OneSetOfLandmarks_np)

        ## For each result (corresponding to each OneSetOfLandmarks)

        ### extract the class with the highest probability.
        encoded_predicted_pose_labels = prediction_of_probabilities.argmax(axis=1)

        predicted_pose_labels = pose_label_encoder.classes_[encoded_predicted_pose_labels].tolist()

        ### extract the probability
        prob_of_predictions = prediction_of_probabilities.max(axis = 1).tolist()

        return predicted_pose_labels, prob_of_predictions
