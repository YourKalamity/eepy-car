from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import pytest

from eepy_car.drowsiness.face import load_landmarker_model, get_face_data
from eepy_car.drowsiness.ear import LEFT_EYE, ear, avg_ear
from eepy_car.drowsiness.mar import mar
import cv2

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"


@pytest.fixture(scope="class")
def landmarker():
    """Loads the landmarker model once for all tests in the class."""
    return load_landmarker_model(MODEL_PATH)


@pytest.fixture
def open_eye_landmarks():
    landmarks = [(0.0, 0.0)] * 478
    landmarks = list(landmarks)
    # the left eye indices are [362, 385, 387, 263, 373, 380]
    # the order being [left corner, top-left, top-right, right corner, bottom-right, bottom-left]
    landmarks[362] = (0.0, 0.0)
    landmarks[385] = (0.5, 1.0)
    landmarks[387] = (1.5, 1.0)
    landmarks[263] = (2.0, 0.0)
    landmarks[373] = (1.5, -1.0)
    landmarks[380] = (0.5, -1.0)
    return landmarks


@pytest.fixture
def closed_eye_landmarks():
    landmarks = [(0.0, 0.0)] * 478
    landmarks = list(landmarks)
    landmarks[362] = (0.0, 0.0)
    landmarks[385] = (0.5, 0.0)
    landmarks[387] = (1.5, 0.0)
    landmarks[263] = (2.0, 0.0)
    landmarks[373] = (1.5, 0.0)
    landmarks[380] = (0.5, 0.0)
    return landmarks


@pytest.fixture
def both_eyes_open_landmarks():
    landmarks = [(0.0, 0.0)] * 478
    landmarks = list(landmarks)

    # Left eye
    landmarks[362] = (0.0, 0.0)
    landmarks[385] = (0.5, 1.0)
    landmarks[387] = (1.5, 1.0)
    landmarks[263] = (2.0, 0.0)
    landmarks[373] = (1.5, -1.0)
    landmarks[380] = (0.5, -1.0)

    # Right eye
    landmarks[33] = (0.0, 0.0)
    landmarks[160] = (0.5, 1.0)
    landmarks[158] = (1.5, 1.0)
    landmarks[133] = (2.0, 0.0)
    landmarks[153] = (1.5, -1.0)
    landmarks[144] = (0.5, -1.0)

    return landmarks


@pytest.fixture
def open_mouth_landmarks():
    landmarks = [(0.0, 0.0)] * 478
    landmarks = list(landmarks)
    landmarks[78] = (0.0, 0.0)
    landmarks[308] = (2.0, 0.0)
    landmarks[13] = (1.0, 1.0)
    landmarks[14] = (1.0, -1.0)
    return landmarks


@pytest.fixture
def closed_mouth_landmarks():
    landmarks = [(0.0, 0.0)] * 478
    landmarks = list(landmarks)
    landmarks[78] = (0.0, 0.0)
    landmarks[308] = (2.0, 0.0)
    landmarks[13] = (1.0, 0.0)
    landmarks[14] = (1.0, 0.0)
    return landmarks


@pytest.fixture
def zero_width_mouth_landmarks():
    landmarks = [(0.0, 0.0)] * 478
    landmarks = list(landmarks)
    landmarks[78] = (1.0, 0.0)
    landmarks[308] = (1.0, 0.0)
    landmarks[13] = (1.0, 1.0)
    landmarks[14] = (1.0, -1.0)
    return landmarks


class TestFaceFunctions:

    def test_load_landmarker_model_successfully(self, landmarker):
        assert isinstance(landmarker, vision.FaceLandmarker)

    def test_load_raises_on_missing_file(self):
        """Should raise when file not found"""
        with pytest.raises(FileNotFoundError):
            load_landmarker_model("ashjdhashds")

    def test_get_landmarks_returns_none_on_blank_frame(self, landmarker):
        """Should return None when no face is present in the frame."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = get_face_data(landmarker, blank_frame)
        assert result == (None, None)

    def test_get_landmarks_from_image_file(self, landmarker):
        """Should load image file and return the landmarks and pose matrix"""
        image_path = Path(__file__).parent / "assets" / "images" / "face.png"

        image = cv2.imread(str(image_path))
        assert image is not None

        landmarks, pose_matrix = get_face_data(landmarker, image)
        assert landmarks is not None and len(landmarks) == 478
        assert pose_matrix is not None


class TestEAR:
    def test_ear_open_eye(self, open_eye_landmarks):
        """Should return EAR ~= 1.0 for an open eye"""
        result = ear(open_eye_landmarks, LEFT_EYE)
        assert pytest.approx(result, abs=1e-4) == 1.0

    def test_ear_closed_eye(self, closed_eye_landmarks):
        """Should return EAR ~= 0.0 for a closed eye"""
        result = ear(closed_eye_landmarks, LEFT_EYE)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_ear_returns_zero_when_horizontal_distance_is_zero(self, open_eye_landmarks):
        """Should return 0.0 when left and right corners overlap"""
        open_eye_landmarks[362] = (1.0, 0.0)
        open_eye_landmarks[263] = (1.0, 0.0)
        result = ear(open_eye_landmarks, LEFT_EYE)
        assert result == 0.0

    def test_avg_ear_both_eyes_open(self, both_eyes_open_landmarks):
        """Should return 1.0 when both eyes are fully open"""
        result = avg_ear(both_eyes_open_landmarks)
        assert result == pytest.approx(1.0, abs=1e-4)


class TestMAR:
    def test_mar_mouth_open(self, open_mouth_landmarks):
        """Should return 1.0 when mouth is open"""
        result = mar(open_mouth_landmarks)
        assert pytest.approx(result, abs=1e-4) == 1.0

    def test_mar_mouth_closed(self, closed_mouth_landmarks):
        """Should return 0.0 for a closed mouth"""
        result = mar(closed_mouth_landmarks)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_mar_zero_width(self, zero_width_mouth_landmarks):
        """Should return 0.0 when mouth is zero width"""
        result = mar(zero_width_mouth_landmarks)
        assert result == 0.0
