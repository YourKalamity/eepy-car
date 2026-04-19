import cv2
import numpy as np
import pytest
from pathlib import Path

from eepy_car.distraction.apriltag import (
    APRILTAG_DICTS,
    build_tag_detector,
    load_camera_calibration,
    detect_tags,
    tag_object_points,
)

from eepy_car.distraction.gaze import (
        compute_gaze_and_pose_diff,
        gaze_offset_degrees,
        head_tag_distance,
)

CALIBRATION_PATH = Path(__file__).parent / "assets" / \
    "test_calibration_output" / "calibration.npz"


@pytest.fixture
def detector():
    return build_tag_detector("tag36h11")


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
            build_tag_detector("not_a_family")

    def test_build_detector_all_valid_families(self):
        """Should successfully build a detector for every supported family."""
        for family in APRILTAG_DICTS:
            assert isinstance(build_tag_detector(
                family), cv2.aruco.ArucoDetector)


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


class TestGaze:

    def test_gaze_offset_returns_none_on_zero_forward_vector(self):
        """Should return None when there is no forward vector"""
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 2] = np.array([0.0, 0.0, 0.0])
        rvec = np.zeros((3, 1), dtype=float)

        assert gaze_offset_degrees(pose_matrix, rvec) is None

    def test_gaze_offset_zero_when_aligned_with_tag_axes(self):
        """Should give a yaw and pitch of 0 when head aligned with tag"""
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 2] = np.array([0.0, 0.0, 1.0])
        rvec = np.zeros((3, 1), dtype=float)

        result = gaze_offset_degrees(pose_matrix, rvec)
        if result is not None:
            yaw, pitch = result
            assert yaw == pytest.approx(0.0, abs=1e-6)
            assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_gaze_offset_positive_yaw_when_forward_points_to_tag_right(self):
        """Should give a positive yaw when the head is facing right"""
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 2] = np.array([1.0, 0.0, 0.0])
        rvec = np.zeros((3, 1), dtype=float)

        result = gaze_offset_degrees(pose_matrix, rvec)
        if result is not None:
            yaw, pitch = result
            assert yaw == pytest.approx(90.0, abs=1e-6)
            assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_gaze_offset_negative_pitch_when_forward_points_to_tag_up(self):
        """Should give a negative pitch when head is facing up (down-is-positive convention)"""
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 2] = np.array([0.0, 1.0, 0.0])
        rvec = np.zeros((3, 1), dtype=float)

        result = gaze_offset_degrees(pose_matrix, rvec)
        if result is not None:
            yaw, pitch = result
            assert yaw == pytest.approx(0.0, abs=1e-6)
            assert pitch == pytest.approx(-90.0, abs=1e-6)

    def test_head_tag_distance_computes_euclidean_distance(self):
        """Should calculate the correct distance between head and tag"""
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 3] = np.array([1.0, 2.0, 3.0])
        tvec = np.array([1.0, 2.0, 6.0], dtype=float)

        distance = head_tag_distance(pose_matrix, tvec)

        assert distance == pytest.approx(3.0, abs=1e-7)

    def test_head_tag_distance_supports_opencv_tvec_shape(self):
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 3] = np.array([0.0, 0.0, 0.0])
        tvec = np.array([[[1.0, 2.0, 2.0]]], dtype=float)

        distance = head_tag_distance(pose_matrix, tvec)

        assert distance == pytest.approx(3.0, abs=1e-7)

    def test_compute_gaze_and_pose_diff_returns_expected_tuple(self):
        """Checks function returns proper tuple"""
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, 2] = np.array([0.0, 0.0, 1.0])
        pose_matrix[:3, 3] = np.array([0.0, 0.0, 0.0])

        rvec = np.zeros((3, 1), dtype=float)
        tvec = np.array([0.0, 0.0, 2.0], dtype=float)

        gaze_offset, distance = compute_gaze_and_pose_diff(
            pose_matrix, rvec, tvec)

        assert isinstance(gaze_offset, tuple)
        assert gaze_offset[0] == pytest.approx(0.0, abs=1e-7)
        assert gaze_offset[1] == pytest.approx(0.0, abs=1e-7)
        assert distance == pytest.approx(2.0, abs=1e-7)
