"""Aim: Return a dataset of landmarks built from a given directory of images.

Function Triage: 
The function must be involved in one of the following:
- converting a Directory to an AllLandmarks, and
- processing the OneSetOfLandmarks custom data type (for now, until refactoring into new file (can't stay in same file as others due to static reasons)) <more at landmarker_model.py>.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt # For visualizing the points.
from functools import partial # For the Keyboard event handler
from typing import Optional # For static typing, starting from the core helper function of Key event handler

# For static typing
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Path3DCollection 
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from . import landmarker_model as lm
from . import utils as utils
from . import data_defs as defs
from . import config

idx_of_PlotTuple_on_display = 0

class ProcessedLandmarks:
  """This class would initialized in two ways: through a directory str or through an already processed CSV file.
  Work on this second method later."""
  
  # ---- MAIN FUNCTIONS
  def __init__(self, directory: str):
    """
    Task: Remove all external calls to convert_directory_to_landmarks, and chnage to, access the landmarks and labels using the class attributes. 
    Also, prefix _ to convert_directory_to_landmarks.
    Initializes a Numpy array of processed landmarks points from the directory
    """
    self.directory = directory
    ## Used to generate each OneSetOfLandmarks
    self.PoseLandmarker = lm.PoseLandmarkerModel().model
    
    # The unprocessed landmarks and labels
    self.unprocessed_AllLandmarks, self.pose_labels = self.convert_directory_to_landmarks()

    # These would be filled by the dataset normalization function.
    self.processed_AllLandmarks = None

    # Implement local variables, so that they become true objects, eg self.landmarks

    # MOVE TO THE SECOND METHOD OF INITIALIZATION
    # csv_filename is the name of the CSV file with unnormalized landmarks
    # This would allow the user to work with a single CSV file even if the normalization functions are different
    # self.unnormalized_csv_filename = ""

  # def __init__(self, unnormalized_csv_filename: str):
  #   """Initializes the normalized landmarks and pose labels from the input unprocessed CSV file"""
  #   # Use method: Initialize which loads the dataset, call the normalize function, which outputs the landmarks etc and you can access as local attributes.

  def make_dataset(self, toPreprocess: bool, toVisualise: bool) -> tuple:
    """Task: Remove make_dataset for reasons mentioned inside and at back of longnote. Replace with the alternative described there.
    Returns the features and labels based on the directory of self
    Each of the outputs is a list.
    toVisualize decides.
    """

    # Save the corresponding dataset representation of the directory to the path of csv_filename
    # We use a CSV file so that we can test different normalization functions on the landmarks without reprocessing the entire directory for each test.
    landmarks, pose_classes = self.convert_directory_to_landmarks()
    # self.save_dataset(landmarks, pose_classes, self.unnormalized_csv_filename)

    # TASK: Complete this by adding labels per key point, drawing the linking lines and labelling them, AND adding the texts showing the pose label, angle of line formed by shoulders, and the coordinates of the hip midpoint.
    # Relevant to gain accuracy_opimization insights when visulaizing the dataset.
    # Visualize dataset
    if toVisualise:
      self.visualize_landmarks(landmarks, pose_classes)

    # TASK: Preprocess this list for training
    if toPreprocess:
      normalized_features = self.preprocess_landmarks(landmarks)
      return normalized_features, pose_classes

    return landmarks, pose_classes

  def convert_directory_to_landmarks(self) -> tuple:
    """
    Converts the self.directory to a dataset of landmarks and pose labels.\n
    Saves the dataset to the path of csv_filename.\n
    Returns: 
      The array of landmarks, 
      The array of corresponding pose class labels
    Reasoning: A Numpy array is chosen over a classic Python list to ensure efficient memory usage, given that the dastset would be large, e.g.: over 1000 rows.\n

    Raises: Assertion Error if the directory is not a two-tier directory. See the definition in data_defs.
    """  
    # Do not process an invalid directory.
    # Checks for incorrect directory structure, corrupt files, etc.
    isValid, message = utils.isValidDirectory(self.directory)
    ## Assert and display the error message.
    assert isValid, message

    # --- Main program logic    
    path_of_each_image, label_of_each_image = utils.image_paths_and_labels_in_directory(self.directory)

    # Map each image [dir] in path_of_each_image to its OneSetOfLandmarks
    an_AllLandmarks, all_pose_labels = utils.convert_image_paths_to_OneSetOfLandmarks(path_of_each_image, label_of_each_image, self.PoseLandmarker)

    # Convert the lists to NumPy arrays for efficient operations
    # landmarks = an_AllLandmarks
    # labels = label_of_each_image
    landmarks_np = np.array(an_AllLandmarks)
    labels_np = np.array(all_pose_labels)

    return landmarks_np, labels_np

  def save_dataset(self, landmarks: list, pose_labels: list, save_path: str) -> None:
      """Saves the landmarks and pose_labels list as a CSV file on disk"""

      # print(f"Succesfully saved unprocessed dataset at {self.normalized_csv_save_path}")
  
  def load_dataset(self, filename: str) -> tuple:
    """Read the created CSV file at extract dataset"""

    an_AllLandmarks = []
    all_labels = []

    return an_AllLandmarks, all_labels

  def visualize_landmarks(self, a_landmarks_df: np.ndarray, pose_labels: np.ndarray) -> None:
    """Displays an interactive window, where each frame is a plotted OneSetOfLandmarks in the the AllLandmarks.
    Implement later: If sample = True, a sample is chosen from each pose class in the AllLandmarks.
    To visualize conversion to detect any errors\n
    
    The output plot of each OneSetOfLandmarks visualizes:
    - The coordinates of key Pose Components.
    - Lines connecting key Pose Components.
    - Pose Label
    - Translation normalization: origin (0, 0, 0) and midpoint of the hips.
    - Rotation normalization: coordinates of the shoulders.
    - Scale normalization: size of the entire OneSetOfLandmarks.    
    - Other helpful visual checks such as:
    -- The names of the body parts that correspond to the plotted lines. These appear as labels of the lines (e.g: Right Thigh, Left Forearm, etc.)
    """
    # Handle an empty AllLandmarks
    if a_landmarks_df.size == 0:
      raise ValueError("There is no OneSetOfLandmarks to visualize.")

    # Initializes the 3D axes, where each OneSetOfLandmarks would be plotted one 
    fig, threeD_ax = create_figure_and_3D_axes()    

    # For each OneSetOfLandmarks in a_landmarks_df, create all the artists objects that compose its entire plot.
    list_of_scatter_and_line_artists = create_scatter_and_line_artists_for_AllLandmarks(threeD_ax, a_landmarks_df) # A list of PlotTuple. See comment of method.

    # Display the plots of the OneSetOfLandmarks in a window, including their pose labels and the sizes of their corresponding OneSetOfLandmarks
    ## We can flip through them using the arrow keys.
    next_key = config.next_key_for_visualization
    prev_key = config.prev_key_for_visualization

    # TASK: Add pose label and size
    # Combine the Pose Label and Pose Size into one structure.
    # Pose_Info_for_PlotTuple
    display_all_graphs_in_window(fig, list_of_scatter_and_line_artists, pose_labels, next_key, prev_key)
  
  def preprocess_landmarks(self, a_landmarks_df):
    """Aligns each landmark in the dataframe"""

    return a_processed_df

def display_all_graphs_in_window(a_fig: Figure, list_of_PlotTuple: list[defs.PlotTuple], pose_labels: np.ndarray, next_key: str, prev_key: str) -> None:
  """Displays an interactive window to display the plots represented by each PlotTuple in the list of PlotTuple.\n
  Also displays the corresponding pose label and the pose size. The plots are displayed in an xyz coordinate system.

  We can go the next and previous plots by pressing the next_key and back_key respectively on the keyboard.
  
  Task: Also, moves to next plot when timer condition is met."""

  # Handle an empty list of PlotTuple
  if list_of_PlotTuple == []:
    raise ValueError("There is no PlotTuple to display")
  
  # Display the first PlotTuple
  idx_of_first_PlotTuple = 0
  first_PlotTuple = list_of_PlotTuple[idx_of_first_PlotTuple]   # Should be 0 when run at first.
  show_PlotTuple(first_PlotTuple)

  # Link a keyboard event handler, to enable navigation between plots.
  ## Use the partial method to pass across the next and previous keys, while meeting the one parameter requirement of a keypress event handler.
  container_for_keyboard_event_handler = partial(
    keyboard_event_handler_with_params,
    a_figure = a_fig,
    a_list_of_PlotTuple = list_of_PlotTuple,
    a_next_key = next_key, 
    a_prev_key = prev_key)
  
  a_fig.canvas.mpl_connect("key_press_event", container_for_keyboard_event_handler)

  # Display the (one) visible PlotTuple
  plt.show()

def keyboard_event_handler_with_params(keyboard_event, a_figure: Figure, a_list_of_PlotTuple: list[defs.PlotTuple], a_next_key: str, a_prev_key: str) -> None:
  """Enables navigation to display the previous or next plot using the arrow keys.\n
  
  Method: 
  If a navigation key is pressed, it hides the artists corresponding to the currently shown plot, and shows the artists of the plot corresponding to the key press.
  However, if a navigation key is not pressed, it does nothing."""
  global idx_of_PlotTuple_on_display

  pressed_key = keyboard_event.key
  no_of_PlotTuples = len(a_list_of_PlotTuple)

  toMove, next_idx = _maybe_give_next_idx(pressed_key, idx_of_PlotTuple_on_display, no_of_PlotTuples, a_next_key, a_prev_key)

  # A navigation key was not pressed.
  if not toMove:
    return None
  
  # Hide the currently displayed PlotTuple
  currently_displayed_PlotTuple = a_list_of_PlotTuple[idx_of_PlotTuple_on_display]

  hide_PlotTuple(currently_displayed_PlotTuple)

  # Show the next PlotTuple
  the_PlotTuple_to_show = a_list_of_PlotTuple[next_idx]

  # Update the state of the display to the new PlotTuple
  idx_of_PlotTuple_on_display = next_idx

  show_PlotTuple(the_PlotTuple_to_show)

  # Redraw the canvas to display the new state of the interactive window.
  a_figure.canvas.draw()

def _maybe_give_next_idx(
    the_pressed_key: str, 
    idx_of_PlotTuple_currently_displayed: int, 
    no_of_PlotTuples_for_display: int, 
    a_next_key: str, 
    a_prev_key: str) -> tuple[bool, Optional[int]]:
  """If a navigation key has been pressed, it returns the index of the PlotTuple to display in the interactive window in the format: True, next_idx\n
  However, if the pressed_key is not a navigation key, it returns False, None.
  """
  max_idx_of_PlotTuples_store = no_of_PlotTuples_for_display - 1

  # Handle error case of an invalid "currently accessed index"
  if idx_of_PlotTuple_currently_displayed > max_idx_of_PlotTuples_store:
    raise ValueError(f"The index of the currently displayed PlotTuple is invalid since it is greater than the max_index of the store of PlotTuples. currently_dislplayed_idx = {idx_of_PlotTuple_currently_displayed}, max_idx = {max_idx_of_PlotTuples_store}")
  
  # Case, where a non-navigation key is pressed.
  if (the_pressed_key != a_next_key) and (the_pressed_key != a_prev_key): 
    return False, None

  # Cases, where a navigation key is pressed.
  if the_pressed_key == a_next_key:
    next_idx = _next_idx_of_keypress("next", idx_of_PlotTuple_currently_displayed, no_of_PlotTuples_for_display)

    return True, next_idx
  
  if the_pressed_key == a_prev_key:
    prev_idx = _next_idx_of_keypress("prev", idx_of_PlotTuple_currently_displayed, no_of_PlotTuples_for_display)

    return True, prev_idx
  
  # No case should reach here. Placed for error catching.
  raise ValueError("Exceptional case, and so could not generate the next_index.")
  # utils.raise_error_and_display_all_args(message)

def _next_idx_of_keypress(direction: str, current_idx: int, length_of_iterable: int) -> int:
  """Returns the index that should be shown with respect to the key press, and the current index.\n
  
  Inputs:
  - The pressed_key should be one of "next" or "prev"."""

  # Enforce non-zero and non-negative length.
  assert 0 < length_of_iterable, "The length parameter is not >= 1."

  # Enforce valid index, within [0, max_idx]
  max_idx = length_of_iterable - 1
  assert 0 <= current_idx <= max_idx
  
  # Enforce only valid values of the direction parameter
  assert direction in ("next", "prev"), "Direction parameter must be either 'next' or 'prev'."

  # The valid cases
  if direction == "next":
    next_idx = (current_idx + 1) % length_of_iterable
    return next_idx
  
  if direction == "prev":
    prev_idx = (current_idx - 1) % length_of_iterable
    return prev_idx

def hide_PlotTuple(a_PlotTuple: defs.PlotTuple) -> None:
  """Sets the PlotTuple to be invisible. 
  Does not redraw the canvas."""
  
  _modify_visibility_of_artists_inPlotTuple(a_PlotTuple, False)

def show_PlotTuple(a_PlotTuple: defs.PlotTuple) -> None:
  """Sets the PlotTuple to be visible. 
  Does not redraw the canvas."""
  
  _modify_visibility_of_artists_inPlotTuple(a_PlotTuple, True)
  
def _modify_visibility_of_artists_inPlotTuple(a_PlotTuple: defs.PlotTuple, is_visible: bool) -> None:
  """Sets the visibility parameter of all artists in the PlotTuple to is_visible"""

  scatter_artist, all_2DLines = a_PlotTuple

  # Display the scatter artist (coordinates of pose components)
  scatter_artist.set_visible(is_visible)
  
  # Display all the linking lines (to join specific pose component coordinates in the scatter artist)
  for line in all_2DLines:
    line.set_visible(is_visible)

def add_pose_label_and_size(self, an_axes3D: Axes3D, pose_label: str, pose_size: float) -> None:
  """Adds the pose label and size as the title of the axes3D object"""

  pose_label_and_size = f"""
  Pose Label: {pose_label},
  Size: {pose_size},
  Hip midpoint: """

  an_axes3D.set_title(pose_label_and_size)

def extract_all_x_y_z_of_OneSetOfLandmarks(an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> tuple[list, list, list]:
  """Returns lists of all x, y and z coordinates in the OneSetOfLandmarks"""

  # --- RECURSIVE HELPER FUNCTION
  def inner_recursive_function(part_of_an_OneSetOfLandmarks: defs.OneSetOfLandmarks, all_previous_x: list , all_previous_y, all_previous_z) -> tuple[list, list, list]:
    """Returns three lists, one each for all the x, y, z in the part_of_an_OneSetOfLandmarks"""

    # Base case
    if part_of_an_OneSetOfLandmarks == []:
      return all_previous_x, all_previous_y, all_previous_z
    
    else:
      current_x, current_y, current_z, smaller_part_of_the_OneSetOfLandmarks = utils.get_next_xyz_and_rest_of_OneSetOfLandmarks(part_of_an_OneSetOfLandmarks)
      
      # Combine into a solution
      all_previous_x.append(current_x)
      all_previous_y.append(current_y)
      all_previous_z.append(current_z)
      
      return inner_recursive_function(smaller_part_of_the_OneSetOfLandmarks, all_previous_x, all_previous_y, all_previous_z)

  all_x_coords, all_y_coords, all_z_coords = inner_recursive_function(an_OneSetOfLandmarks, [], [], [])

  return all_x_coords, all_y_coords, all_z_coords

def create_figure_and_3D_axes() -> tuple[Figure, Axes3D]:
  """Returns a Matplotlib Figure and 3D axes.
  The figure contains only one axes."""

  fig = plt.figure()
  threeD_ax = fig.add_subplot(111, projection="3d")
  threeD_ax.set_xlim(-2, 2)
  threeD_ax.set_ylim(-2, 2)
  threeD_ax.set_zlim(-2, 2)

  return fig, threeD_ax

def create_3D_axes() -> Axes3D:
  """Returns a 3D axes.
  The figure contains only one axes."""

  fig = plt.figure()
  threeD_ax = fig.add_subplot(111, projection="3d")
  threeD_ax.set_xlim(-2, 2)
  threeD_ax.set_ylim(-2, 2)
  threeD_ax.set_zlim(-2, 2)

  return threeD_ax

def create_scatter_and_line_artists_for_AllLandmarks(an_threeD_axes: Axes3D, a_landmarks_df: np.ndarray) -> list[tuple[Path3DCollection, list[Line2D]]]:
  """Returns a list-of PlotTuple, where each contains the plotted points and lines for the corresponding OneSetOfLandmarks in the AllLandmarks\n
  A PlotTuple corresponds to an OneSetOfLandmarks. It is a two-element tuple:
    - The first element is the scatter plot artist, which plots the coordinates of the key (i.e. in mapping) pose components.
    - The second element is a list of line plot artists, where each line plot connects two pose components (e.g.: nose to shoulder midpoint to form the neck).
  Each member of a PlotTuple has their visibility set to False."""

  list_of_PlotTuple = []

  # For creating corresponding Line2D artists.
  points_to_join = [
      ("nose_to_neck", "nose", "shoulder_midpoint"),
      ("shoulder", "left_shoulder", "right_shoulder"),
      ("left_torso", "left_shoulder", "left_hip"),
      ("l_arm", "left_shoulder", "left_elbow"),
      ("l_forearm", "left_elbow", "left_wrist"),
      ("right_torso", "right_shoulder", "right_hip"),
      ("r_arm", "right_shoulder", "right_elbow"),
      ("r_forearm", "right_elbow", "right_wrist"),
      ("hip", "left_hip", "right_hip"),
      ("l_thigh", "left_hip", "left_knee"),
      ("l_lower_leg", "left_knee", "left_ankle"),
      ("l_foot", "left_ankle", "left_foot_tip"),
      ("r_thigh", "right_hip", "right_knee"),
      ("r_lower_leg", "right_knee", "right_ankle"),
      ("r_foot", "right_ankle", "right_foot_tip")]
  
  for an_OneSetOfLandmarks_np in a_landmarks_df:
    # Build the scatter artist
    an_OneSetOfLandmarks = an_OneSetOfLandmarks_np.tolist()

    an_scatter_artist = create_scatter_artists_for_key_components(an_threeD_axes, an_OneSetOfLandmarks)

    # Build the list[Line2D] representing all the lines.
    all_lines = create_all_lines_linking_points(an_threeD_axes, an_OneSetOfLandmarks, points_to_join)

    # Tasks: Add coordinates of hip midpoints. 
    # Calculate angle of rotation of shoulders and display.
    a_PlotTuple = (an_scatter_artist, all_lines)
    
    list_of_PlotTuple.append(a_PlotTuple)
  
  return list_of_PlotTuple

def create_all_lines_linking_points(an_threeD_axes: Axes3D, an_OneSetOfLandmarks: defs.OneSetOfLandmarks, a_points_to_join: list[tuple[str, str, str]]) -> list[Line2D]:
  """Returns a list of artist objects representing lines linking specific points, as enumerated in the"""

  return []

def create_scatter_artists_for_key_components(an_axes3D: Axes3D, an_OneSetOfLandmarks: defs.OneSetOfLandmarks) -> Path3DCollection:
  """Plots key component coordinates of the OneSetOfLandmarks on the axes object.
  These include all the key pose components of the OneSetOfLandmarks, shoulder midpoint, and the hip midpoint.\n
  Returns the reference to the object representing the scatter plot. This is to enable making the plot visible or invisible.
  """  

  # Extract three lists, respectively containing the x, y and z of all key pose components. 
  # The key pose components are the pose components in map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks 

  all_x, all_y, all_z = extract_all_x_y_z_of_OneSetOfLandmarks(an_OneSetOfLandmarks)

  ## Filter for the key components
  key_component_names = list(lm.map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks.keys())

  only_key_x, only_key_y, only_key_z = filter_for_components(key_component_names, all_x, all_y, all_z)

  ## The colors of the keypoint markers
  color_for_keypoints = ["black"] * len(key_component_names)

  # Prepare the coordinates of both the shoulder and hip midpoints to be plotted after
  midpoint_hip_x, midpoint_hip_y, midpoint_hip_z = lm.extract_hip_midpoint(an_OneSetOfLandmarks)

  ## Preparing the shoulder midpoint
  midpoint_shoulder_x, midpoint_shoulder_y, midpoint_shoulder_z = lm.extract_shoulder_midpoint(an_OneSetOfLandmarks)

  x_of_midpoints = [midpoint_hip_x, midpoint_shoulder_x]
  y_of_midpoints = [midpoint_hip_y, midpoint_shoulder_y]
  z_of_midpoints = [midpoint_hip_z, midpoint_shoulder_z]

  ## The colors of the midpoint's markers
  ### We want to visually differentiate the midpoints, so we use different colors as teh last two of the colors list
  color_for_midpoints = ["red"] * len(x_of_midpoints)

  # Compile the x, y and z to plot
  all_x_to_plot = only_key_x + x_of_midpoints
  all_y_to_plot = only_key_y + y_of_midpoints
  all_z_to_plot = only_key_z + z_of_midpoints

  ## Compile the colors list for plotting key and mid points
  colors_list = color_for_keypoints + color_for_midpoints      

  # Plot all the keypoints and midpoints as scatter plots on the graph
  scatter_of_key_and_midpoints = an_axes3D.scatter(all_x_to_plot, all_y_to_plot, zs = all_z_to_plot, c = colors_list, marker = "o", visible = False)

  return scatter_of_key_and_midpoints

def filter_for_components(
    component_names: list[defs.LandmarkComponent],
    all_x: list[float], 
    all_y: list[float], 
    all_z: list[float]) -> tuple[list[float], list[float], list[float]]:
  """
  Returns the x, y and z lists of an OneSetOfLandmarks for only the listed component names.\n
  Assumes that the input x, y, and z list are all of length 33 (due to OneSetOfLandmarks data composition).
  """
  # Handle duplicates. Vital to ensure reliable downstream tasks.
  if len(set(component_names)) != len(component_names):
    raise ValueError(f"This list of component names has duplicates: {component_names}")

  x_of_listed_components, y_of_listed_components, z_of_listed_components = [], [], []

  for c_name in component_names:
    # Handle invalid component names.
    try:
      idx_of_c_name = lm.map_of_keypoint_names_to_start_idx_in_OneSetOfLandmarks[c_name]
    except KeyError as e:
      raise ValueError(f"This invalid component name {c_name} was included for filtering")

    x_of_listed_components.append(all_x[idx_of_c_name])
    y_of_listed_components.append(all_y[idx_of_c_name])
    z_of_listed_components.append(all_z[idx_of_c_name])
  
  return x_of_listed_components, y_of_listed_components, z_of_listed_components

