import cv2
import numpy as np
import pytest
from pathlib import Path

from eepy_car.distraction.apriltag import (
    APRILTAG_DICTS,
    build_detector,
    load_camera_calibration,
    detect_tags,
    tag_object_points,
)

CALIBRATION_PATH = Path(__file__).parent.parent / "calibration_output" / "calibration.npz"


@pytest.fixture
def detector():
    return build_detector("tag36h11")


@pytest.fixture
def calibration():
    return load_camera_calibration(CALIBRATION_PATH)


@pytest.fixture
def object_points():
    return tag_object_points(0.055)


class TestBuildDetector:

    def test_build_detector_valid_family(self, detector):
        """Should return a valid ArucoDetector for a known family."""
        assert isinstance(detector, cv2.aruco.ArucoDetector)

    def test_build_detector_invalid_family(self):
        """Should raise ValueError for an unknown family."""
        with pytest.raises(ValueError):
            build_detector("not_a_family")

    def test_build_detector_all_valid_families(self):
        """Should successfully build a detector for every supported family."""
        for family in APRILTAG_DICTS:
            assert isinstance(build_detector(family), cv2.aruco.ArucoDetector)


class TestLoadCameraCalibration:

    def test_loads_successfully(self):
        """Should return two numpy arrays when given a valid calibration file."""
        camera_matrix, dist = load_camera_calibration(CALIBRATION_PATH)
        assert isinstance(camera_matrix, np.ndarray)
        assert isinstance(dist, np.ndarray)

    def test_raises_on_missing_file(self, tmp_path):
        """Should raise FileNotFoundError if the calibration file does not exist"""
        with pytest.raises(FileNotFoundError):
            load_camera_calibration(tmp_path / "does_not_exist.npz")


class TestDetectTags:

    def test_returns_empty_dict_on_blank_frame(self, detector):
        """Should return empty dict when no tags are present."""
        blank_frame = np.zeros((480, 640), dtype=np.uint8)
        result = detect_tags(detector, blank_frame)
        assert result == {}

    def test_returns_dict(self, detector):
        """Should always return a dict."""
        blank_frame = np.zeros((480, 640), dtype=np.uint8)
        result = detect_tags(detector, blank_frame)
        assert isinstance(result, dict)
