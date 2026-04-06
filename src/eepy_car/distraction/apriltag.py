import cv2
from pathlib import Path
import numpy as np


APRILTAG_DICTS = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5":  cv2.aruco.DICT_APRILTAG_16h5,
}


def build_detector(family: str) -> cv2.aruco.ArucoDetector:
    """Builds an ArucoDetector for the given AprilTag family

    Args:
        family (str): The family of AprilTag to look for

    Raises:
        ValueError: When unknown AprilTag family selected

    Returns:
        cv2.aruco.ArucoDetector: The ArucoDetector
    """
    if family not in APRILTAG_DICTS:
        raise ValueError(
            f"Unknown AprilTag family '{family}'. "
            f"Choose from: {list(APRILTAG_DICTS.keys())}"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[family])
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def load_camera_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load camera calibration outputs from file

    Args:
        path: The path to the calibration `.npz` file

    Returns:
        tuple: A tuple containing:
        - mtx(np.ndarray): Camera matrix as a float64 array
        - dist(np.ndarray): Distortion coefficients as a float64 array
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {path}\n"
            "Run the calibration script to generate it"
        )
    data = np.load(str(path))
    if "mtx" not in data or "dist" not in data:
        raise KeyError(
            f"Calibration file at {path} is missing expected keys\n"
            "Run the calibration script to regenerate it"
        )

    return data["mtx"].astype(np.float64), data["dist"].astype(np.float64)


def detect_tags(detector: cv2.aruco.ArucoDetector, gray: np.ndarray) -> dict[int, np.ndarray]:
    """Detects AprilTags in a grayscale frame.

    Args:
        detector (cv2.aruco.ArucoDetector): The ArucoDetector to run
        gray (np.ndarray): The grayscale image to run the detection on

    Returns:
        dict[int, np.ndarray]: A dictionary where the key is the tag ID and the value the coordinates
    """
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}
    ids = ids.flatten()
    return {int(tag_id): np.asarray(corners) for tag_id, corners in zip(ids, corners_list)}


def tag_object_points(tag_size: float) -> np.ndarray:
    """
    Returns the coordinates of the four corners of an AprilTag marker in 3D space,
    centered at the origin with the tag lying on the XY plane (Z=0).

    Args:
        tag_size (float): The size of the tag
    Returns:
        np.ndarray: A 4x3 array of float64 where each row represents the (x, y, z)
            coordinates of a corner. The corners are ordered as:
            - [-s, -s, 0]: bottom-left
            - [s, -s, 0]: bottom-right
            - [s, s, 0]: top-right
            - [-s, s, 0]: top-left
            where s = tag_size / 2.0
    """

    s = tag_size / 2.0
    return np.array([
        [-s, -s, 0.0],
        [s, -s, 0.0],
        [s,  s, 0.0],
        [-s,  s, 0.0],
    ], dtype=np.float64)


def estimate_pose(corners: np.ndarray,
                  object_points: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Runs solvePnP on a single tag's corners.

    Returns (rvec, tvec) or None if pose estimation fails.
    """
    img_points = corners.reshape(4, 2).astype(np.float64)

    ok_pnp, rvec, tvec = cv2.solvePnP(
        object_points, img_points, camera_matrix, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not ok_pnp:
        return None

    return rvec, tvec
