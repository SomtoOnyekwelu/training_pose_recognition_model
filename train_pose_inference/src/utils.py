"""
Function Triage:
The function must meet all the criteria below:
- Contains helpful functions that are used across several files (not would be, taht is, not in the future).
- Furthermore, the function does not process any custom datatype like OneSetOfLandmarks or AllLandmarks.
"""

import cv2
from math import sqrt
import pathlib

from pose_estimation_rough.train_pose_inference.src import data_defs as defs
from pose_estimation_rough.train_pose_inference.src import utils
from pose_estimation_rough.train_pose_inference.src import landmarker_model as lm

def isValidDirectory(absolute_directory: str) -> tuple:
    """
    Look into creating a Directory class 
    Returns:
        bool: True if the Directory is a valid pose-images Directory, else returns False otherwise
        str: a message descrubing the cause of a False return. 
            If the return is True, it is an empty string
    """
    p = pathlib.Path(absolute_directory)
    # The directory does not exist
    if not p.exists():
        return False, f"The directory does not exist: {absolute_directory}"
    
    # The so-called directory is not a folder
    if not p.is_dir():
       return False, f"This path does not lead to a folder: {absolute_directory}"
    
    # Confirm that the directory is not empty
    if not any(p.iterdir()):
        return False, f"The directory is empty: {absolute_directory}"

    for sub_dir in p.iterdir():
        
        # Confirm that the first level of the directory is made up of only folders
        if not sub_dir.is_dir():
            return False, f"There is a file instead of a subdirectory at {sub_dir}"
        
        # Confirm that each sub-dir contains only valid images.
        for filename in sub_dir.iterdir():

            # Confirm that each subdirectory does not contain non-images
            if not str(filename).lower().endswith(("png", "jpg", "jpeg")):
                return False, f"Non-image file seen at {filename}"

    # If the structure is a Directory-of-subdirectories, return True and an empty message    
    return True, ""

def extract_features_from_dataset(dataset: defs.Dataset) -> defs.AllLandmarks:
    """Extracts a list of AllLandmarks from the Dataset, as defined in the data definition file, where the Dataset ordinarily also contains the class labels"""
    an_AllLandmarks = dataset[0]
    return an_AllLandmarks

def read_image_at_path(a_path: str):
  """Returns the image at the path as an OpenCV RGB image"""
  image = cv2.imread(a_path)
  if image is None:
    raise ValueError(f"The image could not be read.")
  
  ### Convert the image to RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image_rgb

def image_paths_and_labels_in_directory(directory: str) -> tuple[list[str], list[str]]:
    """Returns:\n
    - paths of all the images in the directory, and \n
    - pose label of each image in the directory.\n
    Assumes: The directory is a valid directory.
    """
    paths_for_all_images = []
    pose_labels_for_all_images = []
    p = pathlib.Path(directory)

    # # Do not process an invalid directory.
    # # Checks for corrupt files, etc.
    # isValid, message = utils.isValidDirectory(directory)
    # if not isValid:
    #   raise ValueError(message)

    # Iterate through the directory, where:
    # Each subdirectory corresponds to a class
    for class_subdir in p.iterdir():
      
      for filename in class_subdir.iterdir():
        # Process each file

        paths_for_all_images.append(str(filename))
        pose_labels_for_all_images.append(class_subdir.name)

    return paths_for_all_images, pose_labels_for_all_images

def convert_image_paths_to_OneSetOfLandmarks(image_paths: list, pose_labels: list[defs.PoseLabel], PoseLandmarker) -> tuple[defs.AllLandmarks, list[defs.PoseLabel]]:
   """Converts each image path in the input iterable to a OneSetOfLandmarks.\n
   Removes any pose label whose corresponding image path cannot be read OR has either no person or multiple people in the image."""

   assert len(image_paths) == len(pose_labels)
   
   landmarks = []
   valid_pose_labels = []

   for idx in range(len(image_paths)):
      image_path = image_paths[idx]
      try:
        image_rgb = utils.read_image_at_path(image_path)
      except ValueError as e:
        print(f"The image at {image_path} could not be read. Skipping ...")
        continue

      ## Get the image's landmarks
      ### Skip image edge cases
      try:
        image_landmarks = lm.convert_image_to_landmarks(PoseLandmarker, image_rgb)
      except ValueError as e:
        print(f"The image {image_path} raised an error: {e}")
        continue
      
      ## Append to the landmarks and labels array
      landmarks.append(image_landmarks)
      ## Extract and append the corresponding pose label
      corresponding_pose_label = pose_labels[idx]
      valid_pose_labels.append(corresponding_pose_label)

   return landmarks, valid_pose_labels

def get_next_xyz_and_rest_of_OneSetOfLandmarks(part_of_an_OneSetOfLandmarks: list[float]) -> tuple:
  """
  If the input list is not empty, returns the first xyz component of the OneSetOfLandmarks, removes the next presence value, and returns the remaining OneSetOfLandmarks starting from the next xyz.\n
  Else, returns (None, None, None, []). Espcially useful within recursive calls.

  Assumes: An part_of_an_OneSetOfLandmarks is a list of multiple flattened (x, y, z, and visibility). Hence, it is one of:
  - [] or
  - [x, y, z, and visibility] * n"""

  if part_of_an_OneSetOfLandmarks == []:
     return None, None, None, []
  
  else:
    x, y, z = tuple(part_of_an_OneSetOfLandmarks[0:3])
    
    # Skip the presence value
    remainder = part_of_an_OneSetOfLandmarks[4:]
    
    return x, y, z, remainder

def list_dir(path: str) -> list[str]:
  """Lists all the paths of the files within the directory.
  The path does not need be the Directory datatype."""

  paths_for_all_files = []
  p = pathlib.Path(path)

  # Iterate through the directory, where:
  for filename in p.iterdir():
    # Process each file

    paths_for_all_files.append(str(filename.resolve()))
    
  return paths_for_all_files
