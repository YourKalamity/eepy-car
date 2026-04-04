from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import pytest

from eepy_car.drowsiness.face import load_landmarker_model, get_landmarks

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"


@pytest.fixture(scope="class")
def landmarker():
    """Loads the landmarker model once for all tests in the class."""
    return load_landmarker_model(MODEL_PATH)


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
