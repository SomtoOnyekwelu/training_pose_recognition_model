"""
Tests the current Pose Landmarker model for accuracy (and implicitly, preciseness) of its predictions. Does not test teh presence attribute of its predictions.\n
Hence, to pass the benchmark test, the chosen Pose Landmarker model must predict:
- for all of the keypoints and do so 
-- such that the coordinates are close enough to that of the accurate MediaPipe model. 
    Accuracy of the MediaPipe model was confirmed by repeated visual inspection.

Note on the measures 
- accuracy of the current model can be checked by running the model on the same directory, and checking if differences between its output and that of the saved file fall within a range. 
- preciseness can be measured by running it repeatedly, and checking if the results are within some limit of each other.
"""

import cv2
import mediapipe as mp
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import src.landmarker_model as lm
import tests.develop_later.create_benchmark_for_landmarker_model as cblm

# ---- DATA DEFINITIONS
"""A Keypoint is a Pose Landmark Component that is critical for infering the class of the pose in an image.
The goal is that any pose landmarker model that is used must predict for these Pose Landmark component
"""

# ---- VARIABLES
directory_of_benchmark_images = r"train_pose_inference\tests\samples\valid_dir"
benchmark_savefile = r"pose_landmark_model_benchmark_data.pkl"  # The file that stores the data to be used as a benchmark measure in judging if other pose landmark models are accurate.

# ---- METHODS
def extract_landmarks(image, PoseLandmarkerModel) -> list:
    """Returns the landmarks that correspond to the person in an image\n
    Raises ValueError if no one is detected in the image."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = PoseLandmarkerModel.detect(mp_image)
    all_landmarks = detection_result.pose_world_landmarks

    if all_landmarks == []:
       raise ValueError("No person was detected in this image")
    return all_landmarks

def extract_keypoints(someone_landmarks: list) -> list[tuple[float, float, float]]:
    """Extract a list of (x, y, z) of decisive keypoints\n
    The exact keypoints are defined within this function"""

    names_of_keypoints = list(keypoint_names_to_idx_map.keys())
    keypoints = []

    for keypoint_name in names_of_keypoints:
        xyz = get_keypoint(someone_landmarks, keypoint_name)
        keypoints.append(xyz)
       
    return keypoints

def get_keypoint(someone_landmarks: list, a_keypoint_name: str) -> tuple[float, float, float]:
  """Returns the x, y, z of the keypoint within the landmarks output.\n
  Not defined for all landmark names.
  Assumes that landmarks is the direct output of the Pose Landmarker model"""
  keypoint_idx = keypoint_names_to_idx_map[a_keypoint_name]
  world_landmark_of_keypoint = someone_landmarks[keypoint_idx]

  x = world_landmark_of_keypoint.x
  y = world_landmark_of_keypoint.y
  z = world_landmark_of_keypoint.z

  return x, y, z

def extract_keypoints_of_dir(PoseLandmarkerModel, directory: str):
  """Returns a dictionary that pairs: 
      key: image name
      value: keypoints of pose landmarks
    for each image in the directory
    The keypoints are extracted from the predictions of the PoseLandmarkerModel for each image.
  """
  imageNamesAsKeys_and_landmarksAsValues = dict()
  
  # Adds the keypoints for all images in the directory to the dictionary
  for class_subdir in os.listdir(directory):
      ## Access each sub-directory
      sub_dir_path = os.path.join(directory, class_subdir)
      
      for filename in os.listdir(sub_dir_path):
          ## Process each file
          filename_path = os.path.join(sub_dir_path, filename)

          ### Load each image as a cv2 image object
          #### Skip the image if it cannot be read (eg: empty file.)
          image = cv2.imread(filename_path)
          if image is None:
            print(f"The image at {filename_path} could not be read. Skipping ...")
            continue
          
          #### Convert the image to RGB
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          ### Get the image's landmarks
          try:
            image_landmarks = extract_landmarks(image_rgb, PoseLandmarkerModel)  # Only one person expected per image
          except ValueError as e:
            print(f"No one could be detected in the image at {filename_path}. Skipping ...") 
          
          ### Do not process images with multiple people.
          #### This is to prevent an unclean dataset, since we do not know who is performing the pose then.
          if len(image_landmarks) > 1:
            print(f"The image at {filename_path} seems to contain several people. Skipping ...")
            continue
      
          ### Get decisive keypoints: Shoulders, Hips, Feet, Hands, Knees and Elbows
          landmarks_for_one_person = image_landmarks[0]  # Only one person expected per image
          keypoints_landmarks = extract_keypoints(landmarks_for_one_person)

          ### Add the image name and its keypoint to the directory
          sub_dir_and_filename = os.path.join(class_subdir, filename)
          imageNamesAsKeys_and_landmarksAsValues[sub_dir_and_filename] = keypoints_landmarks
  
  return imageNamesAsKeys_and_landmarksAsValues

def save_python_object_in_file(object: object, filename: str) -> None:
  """
  Writes the python object to a pickle file to serve as a saved benchamrk measure.
  """
  with open(filename, "wb") as f:
      pickle.dump(object, f)

  print(f"Success! Images have been processed and their landmarks have been saved at {filename}")

# --- MAIN PROGRAM
PoseLandmarkerModel = initialize_model(model_path)
imageNamesAsKeys_and_landmarksAsValues = extract_keypoints_of_dir(PoseLandmarkerModel, directory_of_benchmark_images)
save_python_object_in_file(imageNamesAsKeys_and_landmarksAsValues, benchmark_savefile)