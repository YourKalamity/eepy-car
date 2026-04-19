import cv2
import numpy as np


def gaze_offset_degrees(pose_matrix: np.ndarray, rvec: np.ndarray,) -> tuple[float, float] | None:
    """Computes the yaw and pitch offset of the driver's head relative to the headrest AprilTag orientation

    Args:
        pose_matrix (np.ndarray): 4x4 head pose transformation matrix from MediaPipe.
        rvec (np.ndarray): Rotation vector of the headrest AprilTag

    Returns:
        tuple[float, float] | None: A (yaw, pitch) tuple in degrees
    """
    head_forward = pose_matrix[:3, 2]

    head_magnitude = np.linalg.norm(head_forward)
    if head_magnitude < 1e-6:
        return None

    head_forward_normalised = head_forward / head_magnitude

    tag_rotation, _ = cv2.Rodrigues(rvec)

    tag_right = tag_rotation[:3, 0]
    tag_up = tag_rotation[:3, 1]

    yaw = float(np.degrees(np.arcsin(np.clip(np.dot(head_forward_normalised, tag_right), -1.0, 1.0))))
    pitch = -float(np.degrees(np.arcsin(np.clip(np.dot(head_forward_normalised, tag_up), -1.0, 1.0))))

    return yaw, pitch


def head_tag_distance(pose_matrix: np.ndarray, tvec: np.ndarray,) -> float:
    """Euclidean distance in metres between the driver's head and the headrest AprilTag

    Args:
        pose_matrix (np.ndarray): 4x4 head pose transformation matrix from MediaPipe
        tvec (np.ndarray): Translation vector of the headrest AprilTag

    Returns:
        float: Distance in metres between the head and the tag.
    """
    head_position = pose_matrix[:3, 3]
    tag_position = tvec.reshape(3)

    return float(np.linalg.norm(head_position - tag_position))


def compute_gaze_and_pose_diff(pose_matrix: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> tuple[tuple[float, float] | None, float]:
    """Computes the full gaze and pose difference of the head and AprilTag

    Args:
        pose_matrix (np.ndarray): 4x4 head pose transformation matrix from MediaPipe
        rvec (np.ndarray): Rotation vector of the headrest AprilTag
        tvec (np.ndarray): Translation vector of the headrest AprilTag

    Returns:
        tuple[tuple[float, float] | None, float]:  A (yaw, pitch) tuple in degrees followed by the distance
    """
    gaze_offset = gaze_offset_degrees(pose_matrix, rvec)
    distance = head_tag_distance(pose_matrix, tvec)

    return gaze_offset, distance
