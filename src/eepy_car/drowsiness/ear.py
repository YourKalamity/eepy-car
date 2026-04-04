import numpy as np

# These are the landmark indicies used for EAR calculation
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def ear(landmarks: list[tuple[float, float]], indices: list[int]) -> float:
    """Calculates the Eye Aspect Ratio (EAR) for an eye

    Uses the formula from Soukupova & Cech (2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks (list[tuple[float, float]]): Full list of 478 face landmark pixel coordinates returned by get_landmarks
        indices (list[int]): Six landmark indices for the eye, in the order [left corner, top-left, top-right, right corner, bottom-right, bottom-left]

    Returns:
        float: The calculated EAR value
    """
    eye_points = [np.array(landmarks[i]) for i in indices]

    top_left_to_bottom_left = np.linalg.norm(eye_points[1] - eye_points[5])
    top_right_to_bottom_right = np.linalg.norm(eye_points[2] - eye_points[4])
    left_corner_to_right_corner = np.linalg.norm(eye_points[0] - eye_points[3])

    if left_corner_to_right_corner < 1e-6:
        return 0.0

    return float(
        (top_left_to_bottom_left + top_right_to_bottom_right) / (2.0 * left_corner_to_right_corner)
    )


def avg_ear(landmarks: list[tuple[float, float]]) -> float:
    """Returns the average EAR across both eyes

    Args:
        landmarks (list[tuple[float, float]]): Full list of 478 face landmark pixel coordinates returned by get_landmarks

    Returns:
        float: Mean of left and right EAR values
    """
    return (ear(landmarks, LEFT_EYE) + ear(landmarks, RIGHT_EYE)) / 2.0
