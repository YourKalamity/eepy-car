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


class CaptureManager:
    def __init__(self, camera_index):
        """Initialize the video capture object
        Args:
            camera_index (int): Index of the camera device to use for video capture.
        """

        self.camera_index = camera_index

    def __enter__(self):
        """Open the camera capture stream and return the active capture object
        Initializes the camera capture using the configured ``camera_index``
        stores the resulting handle on ``self.cap``, and returns it for use
        within a context manager block
        Returns:
            Any: The initialized camera capture object returned by ``start_capture``.
        """
        self.cap = start_capture(self.camera_index)
        return self.cap

    def __exit__(self, exc_type, exc, tb):
        release_capture(self.cap)
