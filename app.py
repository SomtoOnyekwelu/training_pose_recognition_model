import cv2
import numpy as np
import math
import mediapipe as mp
import time # Used for the livestreaming of images.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import tasks

# ---- DATA STRUCTURE DEFINITIONS
"""
A list-of Landmarks is one of:
    [],
    [Landmarks, ...]

A Landmarks is a List-of NormalizedLandmark, of length 33

A NormalizedLandmark is an object with the following attributes:
x=float, y=float, z=float, visibility=float, presence=float

Direction_String_Vector is a
    [left|constant|right, down|constant|up, backward|constant|forward]
"""

# ---- META ----
APP_TITLE = "For the Love of Dance"

# ---- CONFIG ----
WIDTH = 640
HEIGHT = 480
NUM_CHANNELS = 3

# ---- DO NOT CHANGE
LANDMARK_PRESENCE_THRESHOLD_FOR_VISIBLE_CENTROID_CALC = 0.8 # Detect which pose landmarks are visible on the frame
MOVEMENT_DETECTION_SENSITIVITY = 0.8   #0.9 A higher value means that smaller movements would be used in deciding the direction of movement

# ---- HELPER VARIABLES
former_centroid = (0, 0, 0) # Used by helper function to detect the direction of movement

# ---- CONSTANTS TO DRAW DIRECTION OF MOVEMENTS ON FRAMES
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_FONT_SCALE = 2
HEADER_FONT_THICKNESS = int(2)  # Must be an integer
HEADER_FONT_COLOR = (0, 255, 0)
HEADER_LINE_SPACING = 1.15

SAMPLE_TEXT = "and"
FONT_SCALE = HEADER_FONT_SCALE * 3/4
FONT_THICKNESS = int(HEADER_FONT_THICKNESS * 3/4)  # Must be an integer
FONT_COLOR = HEADER_FONT_COLOR
LINE_SPACING = HEADER_LINE_SPACING

# ---- HELPER FUNCTIONS ----
def initializeLandmarker(model_path: str):
    """
    Returns an initialized the Pose Landmarker model.
    It has tracking to increase FPS.
    Output: PoseLandmarker
    Assumes: 
    The function accepts three parameters:
        PoseLandMarkerResult, input_image: mp.Image, and timestamp: int

    """
    BaseOptions = tasks.BaseOptions
    PoseLandmarker = tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = tasks.vision.PoseLandmarkerResult    # Class of output
    VisionRunningMode = tasks.vision.RunningMode    # For efficient live streaming

    # Limiter: Only detects one person
    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = model_path),
        running_mode = VisionRunningMode.VIDEO
    )

    landmarker = PoseLandmarker.create_from_options(options)

    return landmarker

# The main starter
def run(landmarker_model):
    """
    Shows a frame of a stick-man character in the same pose as person-in-camera/video.
    Selects a single person per image and copies their movement over another character in sync.
    """
    cap = cv2.VideoCapture(0)   # Expand to access arbritrary video file.

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if cap.isOpened():
        print("Camera is opened")
    else:
        print("Camera is not opened")

    # Initialize the timestamp, used for tracking in the mediapipe video/livestream function
    frame_timestamp_ms = 0  # No time passed in the first frame

    while True:
        success_status, frame = cap.read()

        if not success_status:
            print("Camera frame could not be read")
            break

        # Modify the frame
        # 1.1 Compute the timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        ms_per_frame = 1000 / fps
        frame_timestamp_ms += ms_per_frame
        # End 1.1
        result_image = frame_handler(landmarker_model, frame, int(frame_timestamp_ms))

        cv2.imshow(APP_TITLE, result_image)

        if cv2.waitKey(1) & 0xFF == 27: # If ESC key is pressed
            print("Closing the app ...")
            break

    cap.release()
    cv2.destroyAllWindows()

    return None

def frame_handler(landmarker_model, frame: np.ndarray, frame_timestamp_ms: int) -> np.ndarray:
    """
    Returns a frame with text showing the direction of the in-camera-person's movement.
    Would include stick-man character in same pose as that of the person in the input image. 
    
    Input: cv2 frame with or without a person, pose landmarker model
    Output: cv2 frame with the drawn direction.

    Limit: Assumes landmarker_model has livestream option
    """
    # ---- HELPER FUNCTIONS
    def extract_landmarks(detection_results):
        """
        Returns all the landmarks (of all persons) in the results of the model
        Input: poseLandmarkerResult
        Output: List-of Landmarks
        
        Docs: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
        """
        a_lo_landmarks = detection_results.pose_landmarks
        return a_lo_landmarks

    def movement_direction(a_landmarks, origin_pos: tuple) -> list:
        """
        Returns the list of strings representing the direction of movement wrt to the position in the last frame
        Input: poseLandmark, (x, y, z) where each is in [0, 1]
        Output: [left|constant|right, down|constant|up, backward|constant|forward]
        """
        global former_centroid  # To update the centroid tracker with the centroid of this frame

        # Compute poseLandmark midpoint
        # Uses the midpoint of the visible image, not waist midpoint (incase only upper body is available)
        
        def calc_visible_centroid(a_landmarks) -> tuple:
            """
            Computes centroid of the visible landmarks (=> person) 
            Method: Extract min and max of x, y and z across all landmarks with likely enough presence
            Then, computes centroid for each axis by averaging.
            Returns the centroid as an (x, y, z) tuple
            """
            min_x, min_y, min_z, max_x, max_y, max_z = (0, 0, 0, 0, 0, 0)

            for NormalizedLandmark in a_landmarks:
                # Do not include landmarks that are not in the frame
                if NormalizedLandmark.presence >= LANDMARK_PRESENCE_THRESHOLD_FOR_VISIBLE_CENTROID_CALC:
                    # Update the max and min x, y and z
                    this_x = NormalizedLandmark.x
                    if this_x < min_x:
                        min_x = NormalizedLandmark.x
                    if this_x > max_x:
                        max_x = NormalizedLandmark.x
                    
                    this_y = NormalizedLandmark.y
                    if this_y < min_y:
                        min_y = NormalizedLandmark.y
                    if this_y > max_y:
                        max_y = NormalizedLandmark.y

                    this_z = NormalizedLandmark.z
                    if this_z < min_z:
                        min_z = NormalizedLandmark.z
                    if this_z > max_z:
                        max_z = NormalizedLandmark.z
            
            # Compute the centroid by averaging the min and max
            x = (min_x + max_x ) / 2
            y = (min_y + max_y ) / 2
            z = (min_z + max_z ) / 2

            return x, y, z
        
        def translate_d_to_string(a_diff_across_axes: tuple) -> list:
            """
            Converts each of the differences  to its string equivalent.
            Input: (dx, dy, dz), where dx, dy are members of [-1, 1] and dz is in [-2, 2]
            Output: Direction_String_Vector
            """

            # ---- HELPER FUNCTION
            def d_to_string(diff: float, string_maps: tuple, epsilon: float) -> str:
                """
                Converts the difference to its string representation
                Inputs:
                    diff: The difference between two points in the same axis
                    string_maps: The string mappings for positive and negative values of the difference
                    epsilon: The amount of difference that is considered significant enough to output a value in string_maps. Prevents noisy outputs due to insignificant differences'
                Output: An element in string_maps or ""
                Assumes: 
                    e is zero or positive
                    string_maps has two elements
                """
                assert (epsilon >= 0 and epsilon <= 1)
                assert len(string_maps) == 2

                lower_boundary = 0 - epsilon
                upper_boundary = 0 + epsilon

                if diff < lower_boundary:
                    return string_maps[0]
                elif diff > upper_boundary: 
                    return string_maps[1]
                else: 
                    return ""
                
            dx, dy, dz = a_diff_across_axes

            e = 1 - MOVEMENT_DETECTION_SENSITIVITY    # 1 is used because the max value of the normalized coordinates is 1

            # For any given axis, if the displacement from the origin (d) in the two possible directions (positive and negative) is greater than e, produce the appropriate string, else return and empty string
            dx_str = d_to_string(dx, ("left", "right"), e)  # The axis is: right = 1, left = 0
            dy_str = d_to_string(dy, ("up", "down"), e) # The axis is built as: down is 1, up is 0
            dz_str = d_to_string(dz, ("forward", "backward"), e)    # The axis is built as camera = -1, waist = 0, behind waist is negative

            string_direction_vector = [dx_str, dy_str, dz_str]          

            return string_direction_vector
        
        # Compare midpoint with that of last x frames
        cent_x, cent_y, cent_z = calc_visible_centroid(a_landmarks)
        o_x, o_y, o_z = origin_pos

        diff_across_axes = cent_x - o_x, cent_y - o_y, cent_z - o_z

        ## Generate list of directions, one across each direction
        direction_string_vector = translate_d_to_string(diff_across_axes)

        # Update the centroid tracker
        former_centroid = (cent_x, cent_y, cent_z)

        return direction_string_vector
    
    def draw_direction(a_frame, a_direction_string_vector: list):
        """
        Returns the frame with the direction texts drawn on it.
        Input: 
            cv2 frame with person, direction of movement
            a_direction_vector is a Direction_String_Vector
        Output: cv2 frame with person
        """
        # ---- HELPER FUNCTION
        def convert_norm_pos_to_actual(x: float, y: float, a_frame: np.ndarray) -> tuple:
            """
            Returns the frame-dimensions equivalent of the normalized positions
            Input: x, y and the image
            Output: (x_rel, y_rel), where both elements are integers
            Assumes: 
                x, y are members of [0, 1]
                a_frame has dimensions (HEIGHT, WIDTH)
            """
            assert a_frame.shape == (HEIGHT, WIDTH, NUM_CHANNELS)
            
            x_rel = int(x * WIDTH)
            y_rel = int(y * HEIGHT)
            
            return (x_rel, y_rel)

        x, y = 0, 0

        # Draw header text
        ## Needed to compute baseline position and line height, to be used by the texts following the header
        HEADER_TEXT = "MOVEMENT VIEW"
        (HEADER_LINE_WIDTH, HEADER_LINE_HEIGHT), HEADER_BASELINE = cv2.getTextSize(
            HEADER_TEXT[0], FONT, HEADER_FONT_SCALE, HEADER_FONT_THICKNESS
        )
        ## Needed to compute baseline and line height of the directions, to be used in ensure clean spacing when placing the text for the directions
        (LINE_WIDTH, LINE_HEIGHT), BASELINE = cv2.getTextSize(
            SAMPLE_TEXT, FONT, FONT_SCALE, FONT_THICKNESS
        )

        ## Add the header text to the image
        cv2.putText(a_frame, HEADER_TEXT,
                    convert_norm_pos_to_actual(x, y, a_frame),
                    FONT, HEADER_FONT_SCALE, HEADER_FONT_COLOR, HEADER_FONT_THICKNESS)
        
        # Add all the directions
        i = 1   # The current line currently being considered to be drawn (includes header text)
        for direction_str in a_direction_string_vector:
            if i == 1:
                y = y + (i * HEADER_LINE_HEIGHT * HEADER_LINE_SPACING) + (i * HEADER_BASELINE * HEADER_LINE_SPACING)
            else:
                y = y + (i * LINE_HEIGHT * LINE_SPACING) + (i * BASELINE * LINE_SPACING)
            
            cv2.putText(a_frame, direction_str,
                    convert_norm_pos_to_actual(x, y, a_frame),
                    FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            i += 1
        
        # DEBUG: if a_direction_string_vector != ["", "", ""]:
        print(former_centroid)  # Already updated to the current centroid
        print(a_direction_string_vector)
        ## Visualize centroid
        cv2.circle(a_frame, (convert_norm_pos_to_actual(former_centroid[0], former_centroid[1], a_frame)), 
                   100, FONT_COLOR, -1)

        a_frame_with_directions = a_frame
        # Draw directions
        return a_frame_with_directions
    
    # Detect the persons in the image
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)   # Convert openCV image to mediapipe image
    lo_landmarks = extract_landmarks(
        landmarker_model.detect_for_video(mp_frame, frame_timestamp_ms))
    
    # Case handler
    # Return the original frame if no person was detected in the image
    if lo_landmarks == []:
        return frame
    
    # Choose the person closest to center
    # Limiter: Since only one person is detected due to current model settings, we just choose that.
    # Later do selection function

    one_person_landmarks = lo_landmarks[0]
    
    # Detect the direction of the person's movement.
    direction_of_movement = movement_direction(one_person_landmarks, former_centroid)

    # The frame showing the person's direction.
    frame_with_drawn_direction = draw_direction(frame, direction_of_movement)

    return frame_with_drawn_direction

# ---- POSE LANDMARKER INITIALIZATION ----
#x (Not really, if OpenCV reads images one at a time) LIMITER: For streams of images. See running_mode 
# See Docs: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream
landmarker_model_path = r"pose_landmarker_lite.task"

POSE_LANDMARKER = initializeLandmarker(model_path = landmarker_model_path)

run(POSE_LANDMARKER)


