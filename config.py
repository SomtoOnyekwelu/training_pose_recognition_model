import pathlib

# The root directory of the project
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

# Path to the assets directory
ASSETS_DIR = PROJECT_ROOT / "assets"

# Path to the classification model
CLASSIFICATION_MODEL_DIR = ASSETS_DIR / "classification_model"
DIRECT_MODEL_PATH = CLASSIFICATION_MODEL_DIR / "direct_model.pkl"
POSE_LABEL_ENCODER_PATH = CLASSIFICATION_MODEL_DIR / "pose_label_encoder.pkl"

# Path to the pose landmarker model
POSE_LANDMARKER_LITE_PATH = PROJECT_ROOT / "train_pose_inference" / "assets" / "pose_landmarker_lite.task"
