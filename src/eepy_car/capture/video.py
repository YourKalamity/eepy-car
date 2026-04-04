import cv2


def start_capture(camera_index: int) -> cv2.VideoCapture:
    """Starts a cv2 Video Capture and returns the object

    Args:
        camera_index (int): The camera to capture from

    Raises:
        RuntimeError: Raised when camera cannot be opened

    Returns:
        cv2.VideoCapture: The VideoCapture object
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return cap


def release_capture(cap: cv2.VideoCapture) -> None:
    """Releases the VideoCapture object and closes any OpenCV windows.

    Args:
        cap (cv2.VideoCapture): The VideoCapture object to release
    """
    cap.release()
    cv2.destroyAllWindows()


def read_frame(cap: cv2.VideoCapture) -> tuple[bool, cv2.typing.MatLike]:
    """Reads a single frame from the VideoCapture object.

    Args:
        cap (cv2.VideoCapture): An open VideoCapture object

    Returns:
        tuple[bool, MatLike]: Success flag and the frame. Frame is None if unsuccessful.
    """
    return cap.read()
