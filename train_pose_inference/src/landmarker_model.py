"""
Class and Method Triage
Contains the:
- class representing the Pose Landmarker model. Currently, the Pose Landmarker model is Google's Mediapipe Pose Landmarker.
- provides APIs to access the model's functions such as initialization, and inference.

The expected output of calling an instantiated model (from this module) on an image is to get s OneSetOfLandmarks.

"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from math import sqrt

import pose_estimation_rough.config as config
from pose_estimation_rough.train_pose_inference.src import utils
from pose_estimation_rough.train_pose_inference.src import data_defs as defs

class Source_PoseLandmarkerModel:
  """Initializes the Pose Landmarker model."""

class PoseLandmarkerModel:
  """Initializes the Pose Landmarker model.
  Intended for dataset creation and processing. That is, for creating the OneSetOfLandmarks for the dataset.
  
  The main difference from the InferencePoseLandmarker model is that this can only detect at most two persons per image."""

  # --- METHODS
  def __init__(self, use_case: str = "training"):
    """Initializes the Pose Landmarker model\n
    Currently: The model is the Google MediaPipe Pose Landmarker model.

    use_case is one of: "training" or "production". It sets the maximum people that can be detected in an image to either two or max respectively.\n

    The default use_case is "training", which sets the max_detection_count to two, so as to detect invalid images (when multiple people are in an image, and so the exact person responsible for the pose is ambigious).
    """
    self.use_case = use_case

    if use_case == "training":
      self.max_detection_count = 2   # Set to two to be able to detect images with multiple people during training.
    
    elif use_case == "production":
      self.max_detection_count = 8

    self.model = self._model(self.max_detection_count)

  def _model(self, max_detection_count: int):
    """Initializes the Pose Landmarker model\n
    Currently: The model is the Google MediaPipe Pose Landmarker model.
    
    Inputs:
    - max_detection_count is the maximum number of people that the model can detect in an image."""

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=max_detection_count
    )
    return PoseLandmarker.create_from_options(options)
  
  def training_convert_image_to_OneSetOfLandmarks(self, an_image: np.ndarray) -> defs.OneSetOfLandmarks:
    """
    Returns one OneSetOfLandmarks corresponding to the person in the image.
    
    Asserts that this PoseLandmarkerModel object was initialized with "training" use case.
    
    The assumption is that that there is only one person in the image. Asserts this."""

    assert self.use_case == "training"
    
    an_OneSetOfLandmarks = convert_image_to_landmarks(self.model, an_image)

    return an_OneSetOfLandmarks
  
  def production_convert_image_to_OneSetOfLandmarks(self, an_image: np.ndarray) -> list[defs.OneSetOfLandmarks]:
    """
    Returns a list[defs.OneSetOfLandmarks]

    Asserts that this PoseLandmarkerModel object was initialized with "production" use case.
    """

    assert self.use_case == "production"
    
    several_OneSetOfLandmarks = general_convert_image_to_landmarks(self.model, an_image, self.max_detection_count)

    return several_OneSetOfLandmarks

  def get_norm_OneSetOfLandmarks_wrt_img_coords_for_PoseLabelToDraw(self, an_image: np.ndarray) -> list[defs.NormImgOneSetOfLandmarks]:
    """Returns the NormImgOneSetOfLandmarks dimensions for each person in the camera frame."""    
    one_or_several_NormImgOneSetOfLandmarks = general_convert_image_to_landmarks(self.model, an_image, self.max_detection_count, reference = "image")

    return one_or_several_NormImgOneSetOfLandmarks

# STATIC CLASS VARIABLES AND METHODS (also includes for OneSetOfLandmarks)
model_path = str(config.POSE_LANDMARKER_LITE_PATH)

count_of_components_in_OneSetOfLandmarks = 33 # The number of pose points (e.g.: nose, right hip, etc) that the model outputs

map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks = {
    'nose': 0, 
    'left_shoulder': 11,'right_shoulder': 12,'left_elbow': 13,'right_elbow': 14,
    'left_wrist': 15,'right_wrist': 16,'left_hip': 23,'right_hip': 24,
    'left_knee': 25,'right_knee': 26,'left_ankle': 27,'right_ankle': 28, "left_foot_tip": 31, "right_foot_tip": 32
} # The first index is zero, and idx * 4 is the staring idx in OneSetOfLandmarks [x, y, z, p, ...]

def extract_hip_midpoint(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> tuple:
  """Returns the (x, y, z) of the midpoint of the hips"""

  # Extract the x, y and z of the left hip
  left_hip_xyz = extract_landmark_component_xyz(an_OneSetOfLandmarks, "left_hip")

  # For the right hip
  right_hip_xyz = extract_landmark_component_xyz(an_OneSetOfLandmarks, "right_hip")

  # Compute the midpoint
  mid_x, mid_y, mid_z = compute_midpoint(left_hip_xyz, right_hip_xyz)

  return mid_x, mid_y, mid_z

def extract_torso_midpoint(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> tuple:
  """Returns the (x, y, z) of the torso center."""

  # Extract the x, y and z of the hip center
  center_hip_xyz = extract_hip_midpoint(an_OneSetOfLandmarks)

  # For the right hip
  center_shoulder_xyz = extract_shoulder_midpoint(an_OneSetOfLandmarks)

  # Compute the midpoint
  mid_x, mid_y, mid_z = compute_midpoint(center_hip_xyz, center_shoulder_xyz)

  return mid_x, mid_y, mid_z

def extract_shoulder_midpoint(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> tuple:
  """Returns the (x, y, z) of the midpoint of the shoulders."""

  # Extract the x, y and z of the left hip
  left_shoulder_xyz = extract_landmark_component_xyz(an_OneSetOfLandmarks, "left_shoulder")

  # For the right hip
  right_shoulder_xyz = extract_landmark_component_xyz(an_OneSetOfLandmarks, "right_shoulder")

  # Compute the midpoint
  mid_x, mid_y, mid_z = compute_midpoint(left_shoulder_xyz, right_shoulder_xyz)

  return mid_x, mid_y, mid_z

def extract_landmark_component_xyz(an_OneSetOfLandmarks: defs.OneSetOfLandmarks, landmark_component_name) -> tuple:
  """Returns the x, y, z of the above component in the input OneSetOfLandmarks
  
  Raises KeyError if landmark_component_name is not a key in the map_of_keypoint_names_to_idx dictionary 
  """
  # With respect to the OneSetOfLandmarks
  component_x_idx, component_z_idx = component_idx_in_OneSetOfLandmarks(landmark_component_name)  

  # We are not extracting the presence attribute of the component.
  end_idx = component_z_idx + 1
  xyz_list = an_OneSetOfLandmarks[component_x_idx:end_idx] 

  return tuple(xyz_list)

def component_idx_in_OneSetOfLandmarks(component_name: str) -> tuple[int, int]:
    """Returns the indexes for a component's x and z within a OneSetOfLandmarks, when given the component name.
    Maps the component names based on the PoseLandmarkerModel.map_of_keypoint_names_to_idx
    
    Raises KeyError if landmark_component_name is not a key in the map_of_keypoint_names_to_idx dictionary.
    """

    map = map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks
    
    x_idx = map[component_name] * 4 # Since OneSetOfLandmarks flattens x, y, z, presence per Landmark Component. The first component has index 0, not 1. This is the position of the x attribute for the Landmark Component.

    z_idx = x_idx + 2 # This is the index of the z attribute of the component. See above comment for more details
    return x_idx, z_idx

def compute_midpoint(xyz_1: tuple, xyz_2: tuple) -> tuple:
  """Returns the midpoint of the two input points"""

  x1, y1, z1 = xyz_1
  x2, y2, z2 = xyz_2

  mid_xyz = (x1 + x2)/2, (y1 + y2)/2, (z1 + z2)/2

  return mid_xyz

def extract_shoulder_coords(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> tuple:
  """Returns (x_l, y_l, z_l, x_r, y_r, z_r), containing the coordinates both shoulders as a flattened tuple"""

  l_shoulder = extract_landmark_component_xyz(an_OneSetOfLandmarks, "left_shoulder")

  r_shoulder = extract_landmark_component_xyz(an_OneSetOfLandmarks, "right_shoulder")
  
  both_shoulders = l_shoulder + r_shoulder
  return both_shoulders

def WorldLandmarks(all_landmarks_of_a_single_person: list) -> list:
  """Returns the landmarks of the input image in where their coordinates is in meters and the hip midpoint is the origin"""
  # Since we have gotten the world landmarks directly from the Mediapipe model, we return the landmarks

  world_landmarks = all_landmarks_of_a_single_person
  return world_landmarks

def general_convert_image_to_landmarks(landmarker_model, an_image: np.ndarray, max_count: int, reference: str = "hip_midpoint") -> list[defs.OneSetOfLandmarks:]:
  """Returns an OneSetOfLandmarks for each person in an image.\n

  Use case: 
  If used for dataset creation during training, max_count == 1\n
  Otherwise, it is for Production inference, where it is assumed that multiple people are in an image frame, and so we need each of their OneSetOfLandmarks to predict for them.

  Output: An flattened form of many WorldLandmarks x, y, z, presence.\n

  Raises a ValueError for any of these edge cases:\n
  - The number of people detected in the image is != max_count.
  - The image fails to be read.\n
    
  """
  # Run the pose landmarker model on it
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=an_image)
  detection_result = landmarker_model.detect(mp_image)

  # Return the landmarks with respect to the hip midpoint or image dimesnsions.
  assert (reference in ["hip_midpoint", "image"]), "You must request for the landmarks wrt either the hip midpoint or image."
  if reference == "hip_midpoint":
    all_landmarks = detection_result.pose_world_landmarks
  else:
    all_landmarks = detection_result.pose_landmarks

  # Edge case handling: Raise an error if no one is detected in the image
  if all_landmarks == []:
    raise ValueError("No person was detected in this image.")
  
  if len(all_landmarks) > max_count:
    raise ValueError(f"The number of persons detected in this image is greater than {max_count}.")
  
  else:
    lo_ASetOfLandmarks = []
    list_of_people_landmarks = all_landmarks # Now, len(all_landmarks) >= 1. In Mediapipe, this is a list[list of Landmarks objects]
    for a_person_landmarks in list_of_people_landmarks:

      world_landmarks = WorldLandmarks(a_person_landmarks)

      # Extract x, y, z and presence from all the normalized landmarks, then flatten them all
      flattened_xyzpresence = convert_to_OneSetOfLandmarks(world_landmarks)

      lo_ASetOfLandmarks.append(flattened_xyzpresence)

    return lo_ASetOfLandmarks
  
def convert_image_to_landmarks(landmarker_model, an_image: np.ndarray) -> defs.OneSetOfLandmarks:
  """Returns an OneSetOfLandmarks for the single person in an image.\n
  Output: An flattened form of many WorldLandmarks x, y, z, presence.\n
  Raises a ValueError for any of these edge cases:\n
    The number of persons detected in the image != 1
    The image fails to be read.\n
  Use case: When we are in the training use case, we assume that there is only one person per image. So, as to map the OneSetOfLandmarks to the pose label. If this condition is not met, the image is excluded. Hence, data cleaning is required.
  """
  # Raises an error if the number detected is more than 1, or no one was detected.
  try:
    OneSetOfLandmarks_in_a_list = general_convert_image_to_landmarks(landmarker_model, an_image, 1)
  except Exception as e:
    raise ValueError(e)
  
  an_OneSetOfLandmarks = OneSetOfLandmarks_in_a_list[0] # To select the first and only entry

  return an_OneSetOfLandmarks
  
def convert_to_OneSetOfLandmarks(all_landmarks_of_a_single_person: list) -> defs.OneSetOfLandmarks:
  """Helper: Returns an OneSetOfLandmarks. See defs for definition.\n
  Note: Receives a landmarks list, not an image.
  In essence: Extracts x, y, z and presence attributes of each landmaker component into a list, such that the final list is flattened"""

  an_OneSetOfLandmarks = []

  for landmark_component in all_landmarks_of_a_single_person:
    # Extract values from the Mediapipe object
    values_to_append = [landmark_component.x,
                        landmark_component.y,
                        landmark_component.z,
                        landmark_component.presence]
    
    an_OneSetOfLandmarks.extend(values_to_append)

  return an_OneSetOfLandmarks  

def get_sum_of_distances_from_midpoint(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> float:
  """Sums the distances of all other components landmarks with respect to the midpoint.\n
  Distance is computed with respect to midpoint of the hip."""

  mid_x, mid_y, mid_z = extract_hip_midpoint(an_OneSetOfLandmarks)

  # ----- RECURSIVE HELPER FUNCTION
  def do_sum_of_distances_from_midpoint(an_OneSetOfLandmarks: defs.OneSetOfLandmarks, previous_sum: float) -> float:
      """Computes and returns the sum of the distances of component landmarks in the an_OneSetOfLandmarks.\n
      Distance is computed with respect to midpoint of the hip."""

      if an_OneSetOfLandmarks == []:
          return previous_sum
      else:
          x, y, z, rest_of_OneSetOfLandmarks = utils.get_next_xyz_and_rest_of_OneSetOfLandmarks(an_OneSetOfLandmarks)

          distance_of_current_component = sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2 + (z - mid_z) ** 2)

          current_sum = previous_sum + distance_of_current_component

          return do_sum_of_distances_from_midpoint(rest_of_OneSetOfLandmarks, current_sum)
    
  sum_of_distances = do_sum_of_distances_from_midpoint(an_OneSetOfLandmarks, 0)
  
  return sum_of_distances