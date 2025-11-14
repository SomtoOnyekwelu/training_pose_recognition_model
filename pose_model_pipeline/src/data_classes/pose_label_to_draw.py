import cv2
import numpy as np
from pose_estimation_rough.train_pose_inference.src import data_defs as defs

class PoseLabelToDraw:
    """
    Contains the PoseLabel for a given person in an image, the probability fo the pose label, and the position on the image on which the pose label text should be drawn.
    """

    def __init__(self, pose_label: defs.PoseLabel, pose_probability: float, drawing_pos_on_source_frame: tuple[int, int]):
        """Instantiates the object.\n
        Assumes that the drawing position is with respect to the image to draw on. Also, assumes that it is normalized between 0 and 1."""

        self.pose_label = pose_label

        self.pose_probability = pose_probability

        self.xy_coord_to_draw_on_source_frame = drawing_pos_on_source_frame

    def draw(self, frame: np.ndarray) -> None:
        """Draw the pose label and probability near the person coordinate.
        The label is centered horizontally at the provided (x, y) and placed
        above that point if there's room, otherwise below. Works with pixel
        coords or normalized coords in [0..1]."""
        x_raw, y_raw = self.xy_coord_to_draw_on_source_frame

        h, w = frame.shape[:2]

        # If coords look normalized (0..1), convert to pixel coords
        if 0.0 <= x_raw <= 1.0 and 0.0 <= y_raw <= 1.0:
            x = int(x_raw * w)
            y = int(y_raw * h)
        else:
            x = int(x_raw)
            y = int(y_raw)

        # --- Text formatting ---
        display_text = f"{self.pose_label}: {self.pose_probability:.2f}"

        # --- Drawing parameters ---
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.6
        FONT_THICKNESS = 1
        TEXT_COLOR = (0, 0, 0)       # Black text (B,G,R)
        RECT_COLOR = (255, 255, 255) # White rectangle
        PADDING = 6                  # padding inside the rectangle
        GAP = 5                      # gap between the point and the rectangle

        (text_w, text_h), baseline = cv2.getTextSize(
            text=display_text,
            fontFace=FONT,
            fontScale=FONT_SCALE,
            thickness=FONT_THICKNESS
        )

        box_w = text_w + 2 * PADDING
        box_h = text_h + 2 * PADDING + baseline

        # Start with box centered horizontally on x and _above_ y
        box_x1 = int(x - box_w // 2)
        box_y2 = int(y - GAP)                 # bottom of box (just above the point)
        box_y1 = box_y2 - box_h               # top of box

        # If the box would go off the top of the frame, place it below the point instead
        if box_y1 < 0:
            box_y1 = int(y + GAP)
            box_y2 = box_y1 + box_h

        # Clamp horizontally to frame
        if box_x1 < 0:
            box_x1 = 0
        box_x2 = box_x1 + box_w
        if box_x2 > w:
            box_x2 = w
            box_x1 = box_x2 - box_w
            if box_x1 < 0:
                box_x1 = 0  # extreme small frames

        # Clamp vertically (safety)
        box_y1 = max(0, box_y1)
        box_y2 = min(h, box_y2)

        # Draw background rectangle
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), RECT_COLOR, thickness=cv2.FILLED)

        # Compute text origin: bottom-left corner of text baseline within the rectangle
        text_org = (box_x1 + PADDING, box_y1 + PADDING + text_h)

        # Draw text
        cv2.putText(
            frame,
            display_text,
            org=text_org,
            fontFace=FONT,
            fontScale=FONT_SCALE,
            color=TEXT_COLOR,
            thickness=FONT_THICKNESS,
            lineType=cv2.LINE_AA
        )

def draw_list_of_PoseLabelToDraw_on_frame(list_of_PoseLabelToDraw: list[PoseLabelToDraw], frame: np.ndarray) -> None:
    """Draws all the PoseLabelToDraw in the passed list on the frame.\n
    
    Returns None because the pose labels are drawn inplace."""

    for a_PoseLabelToDraw in list_of_PoseLabelToDraw:
        a_PoseLabelToDraw.draw(frame)
