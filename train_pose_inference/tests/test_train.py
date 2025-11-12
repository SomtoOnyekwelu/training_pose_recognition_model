import xgboost as xgb
import pytest

from pose_estimation_rough.train_pose_inference.src.train import PoseClassifierModel

class test_train:
    def test_init(self):
        # Invalid directory.
        # Raises an error
        invalid_dir = "random"
        with pytest.raises(ValueError):
            assert PoseClassifierModel(invalid_dir)

        # Valid directory
        # Completes successfully
        valid_dir = r"train_pose_inference/tests/samples/valid_dir"
        self.test_model = PoseClassifierModel(valid_dir)
        assert self.test_model.directory == valid_dir

    def test_get_model(self):
        # Check if the model is an XGBoost Classifier model
        result = self.test_model.get_model()
        assert isinstance(result, xgb.XGBClassifier)
