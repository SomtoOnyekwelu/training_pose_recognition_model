from mpl_toolkits.mplot3d.art3d import Path3DCollection # For typing
from matplotlib.lines import Line2D

"""
Class and function triage
-----------
The classes here must:
- Define a custom data type.

All functions must:
- be methods of a class, and
- abstract away the internal data and provide APIs for the instances of the class. E.g: extract_hip_midpoint() in OneSetOfLandmarks.
"""

"""
A directory contains Second_level_Directory's or other and is one of:
    Empty, 
    length=1 
    length>1
A Second_level_Directory contains Files and is one of: 
    Empty, 
    One File, 
    File>1
A File is one of:
    Image
    unknown
An Image is one of:
    no person in image
    one person in image
    multiple persons in image
"""
type PoseLabel = str
"""
Represents one of the outputs of the model.
It is the label that identifies the pose someone in an image is performing.
"""

type LandmarkComponent = str
"""
Represents a subset of the Landmark Components in a OneSetOfLandmarks. 
It can be accessed using a string:
{
    'nose', 
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle',
    "left_foot_tip",
    "right_foot_tip"
}
"""

type OneSetOfLandmarks = list[float]
"""
Represents a flattened version of all the landmarks of a single person. It would be used in training the pose inference model.\n
Each component landmark is (x, y, z, and presence)\n
Each object should have a legth of 33*4=132, given that there are 33 landmark components (nose, eyes, etc).
"""
"""NOTE ON WHY I INCLUDED THE PRESENCE ATTRIBUTE BUT EXCLUDED THE VISIBILITY ATTRIBUTE:
-----
The MediaPipe model outputs both presence and visibility, where both measure that a Landmark component is located with a frame. However, visibility also accounts for if the component is occuluded (covered by another object).

I used presence because it would lead to a more accurate pose inference model, since it enables the pose inference model to learn to filter out low-confidence coordinates from the Pose Landmarker model. This would enable it account for the fact that the MediaPipe Pose Landmarker model generates low-confidence coordinates for component landmarks that it cannot see. <add example>

I did not use visibility because it would lead to a less robust pose inference model, for real-world use cases. This is because, for each pose class in the training dataset and in the real world, several images are different angles of the same person in the same pose. Hence, if I chose to use visibility, it would introduce noise that impedes the model learning since at different angles of the same pose, a component landmark may be visbile at one and blocked at the other. <add example>
"""

type AllLandmarks = list[OneSetOfLandmarks]
"""
Represents all the landmarks derived from all the valid photos of people in poses.\n
It is a List[OneSetOfLandmarks]
"""

type PoseLabels = list[str]
"""
Represents the corresponding pose labels (classes) of each member of AllLandmarks.
"""

type Dataset = tuple[AllLandmarks, PoseLabels]
"""
A Dataset represents the combination of the landmarks in the dataset and their corresponding pose labels (classes)\n
It is a (AllLandmarks, PoseLabels).
"""

type PlotTuple = tuple[Path3DCollection, list[Line2D]]
"""
A PlotTuple corresponds to an OneSetOfLandmarks. It is a two-element tuple:
- The first element is the scatter plot artist, which plots the coordinates of the key (i.e. in mapping) pose components.
- The second element is a list of line plot artists, where each line plot connects two pose components (e.g.: nose to shoulder midpoint to form the neck).
"""

type NormImgOneSetOfLandmarks = list[float]
"""This is unlike the OneSetOfLandmarks in the sense that the image dimensions is the frame of reference for each the outputed coordinates for the pose components.\n
While the OneSetOfLandmarks is a world landmark, that is the hip of the person is the frame of reference (=> hip midpoint coords = (0, 0)). one is a """