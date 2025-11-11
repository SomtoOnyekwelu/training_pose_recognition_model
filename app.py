import cv2
from pose_model_pipeline.src.inference_pipeline import PoseInferencePipeline
from pose_model_pipeline.src.data_classes.pose_label_to_draw import draw_list_of_PoseLabelToDraw_on_frame

class RealtimePoseInferenceApp:
    """An app that runs a pose inference model over live-streamed images of the camera."""

    def __init__(self, path_to_direct_model: str, path_to_pose_label_encoder: str, frame_width: int, frame_height: int, window_name: str):
        """Initializes the state and runs the app."""

        self.pose_inference_pipeline = PoseInferencePipeline(path_to_direct_model, path_to_pose_label_encoder)

        self.run(frame_width, frame_height, window_name)

    def run(self, frame_width: int, frame_height: int, window_name: str):
        """Starts the inference over the camera stream and shows the results as video in a new window."""

        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return

        while True:
            toCloseWindow = False # The flag to close the video stream.

            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            # Predict the poses of each detected person in the frame.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prevent crashing when no person is detected in this frame."
            try:
                list_of_PoseLabelToDraw = self.pose_inference_pipeline.predict_image(frame_rgb)
            except ValueError as e:
                if str(e) == "No person was detected in this image.":
                    print(str(e))
                    toCloseWindow = self.show_image_and_check_for_exit_key_press(window_name, frame)
                    continue
                else:
                    raise e

            # Draw all the pose_labels on the frame.
            # On BGR frame, since imshow expects that.
            draw_list_of_PoseLabelToDraw_on_frame(list_of_PoseLabelToDraw, frame)
            
            toCloseWindow = self.show_image_and_check_for_exit_key_press(window_name, frame)

            if toCloseWindow:
                break

        cap.release()
        cv2.destroyAllWindows()

    def show_image_and_check_for_exit_key_press(self, window_name: str, frame) -> bool:
        """
        Displays the input image as the next frame in the displayed window.
        Furthermore, returns True if the ESC key was pressed, or False if not pressed.
        """

        cv2.imshow(window_name, frame)

        # Update the window after a short time.
        key = cv2.waitKey(1)

        # After, check for if the ESC key was pressed within the last wait period.
        if key & 0xFF == 27: # ESC key
            return True
        
        return False

# Execute when the file is run.
if __name__ == "__main__":
    RealtimePoseInferenceApp(
        path_to_direct_model=r"pose_model_pipeline\assets\classification_model\direct_model.pkl",
        path_to_pose_label_encoder=r"pose_model_pipeline\assets\classification_model\pose_label_encoder.pkl",
        frame_width=680,
        frame_height=480,
        window_name="Real-Time Pose App"
    )

