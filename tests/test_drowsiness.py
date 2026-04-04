from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import pytest

from eepy_car.drowsiness.face import load_landmarker_model, get_landmarks
from eepy_car.drowsiness.ear import LEFT_EYE, ear, avg_ear

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
    landmarks[33]  = (0.0, 0.0)
    landmarks[160] = (0.5, 1.0)
    landmarks[158] = (1.5, 1.0)
    landmarks[133] = (2.0, 0.0)
    landmarks[153] = (1.5, -1.0)
    landmarks[144] = (0.5, -1.0)

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
        result = get_landmarks(landmarker, blank_frame)
        assert result is None

    # TODO Add test cases for actual face scanning

    def test_ear_open_eye(self, open_eye_landmarks):
        """Should return EAR ~= 1.0 for an open eye"""
        result = ear(open_eye_landmarks, LEFT_EYE)
        assert pytest.approx(result, abs=1e-4) == 1.0

    def test_ear_closed_eye(self, closed_eye_landmarks):
        """Should return EAR ~= 0.0 for a closed eye"""
        result = ear(closed_eye_landmarks, LEFT_EYE)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_ear_returns_zero_when_horizontal_distance_is_zero(self, open_eye_landmarks):
        """Should return 0.0 when left and right corners overlap."""
        open_eye_landmarks[362] = (1.0, 0.0)
        open_eye_landmarks[263] = (1.0, 0.0)
        result = ear(open_eye_landmarks, LEFT_EYE)
        assert result == 0.0

    def test_avg_ear_both_eyes_open(self, both_eyes_open_landmarks):
        """Should return 1.0 when both eyes are fully open."""
        result = avg_ear(both_eyes_open_landmarks)
        assert result == pytest.approx(1.0, abs=1e-4)
