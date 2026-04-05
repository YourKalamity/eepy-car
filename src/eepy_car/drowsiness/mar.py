import numpy as np

MOUTH = [78, 308, 13, 14]


def mar(landmarks: list[tuple[float, float]]) -> float:
    """Calculates the Mouth Aspect Ratio (MAR)
    Calculated by finding ratio of vertical mouth opening to horizontal mouth width
    Higher values indicate a more open mouth, as seen during yawning

    Args:
        landmarks (list[tuple[float, float]]): Full list of 478 face landmark pixel coordinates returned by get_landmarks
    Returns:
        float: The calculated MAR value
    """
    left = np.array(landmarks[MOUTH[0]])
    right = np.array(landmarks[MOUTH[1]])
    top = np.array(landmarks[MOUTH[2]])
    bottom = np.array(landmarks[MOUTH[3]])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)
