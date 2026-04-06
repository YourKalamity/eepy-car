from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def load_landmarker_model(model_path: str | Path):
    """Loads the MediaPipe FaceLandmarker model from a .task file.

    Args:
        model_path (str | Path): Path to the face_landmarker.task file.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.

    Returns:
        vision.FaceLandmarker: The loaded landmarker model ready for inference.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )

    return vision.FaceLandmarker.create_from_options(options)


def get_face_data(landmarker, frame_bgr: np.ndarray) -> list[tuple[float, float]] | None:
    """Runs face landmark detection on a single BGR frame.

    Converts the frame to RGB, runs the MediaPipe landmarker, and returns
    the pixel coordinates of all 478 landmarks for the first detected face.

    Args:
        landmarker (vision.FaceLandmarker): The loaded landmarker model.
        frame_bgr (np.ndarray): A single BGR frame from OpenCV.

    Returns:
        list[tuple[float, float]] | None: A list of (x, y) pixel coordinates or None if no face was detected.
    """
    height, width = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    return [
        (landmark.x * width, landmark.y * height)
        for landmark in landmarks
    ]
