import argparse

import cv2
import numpy as np


APRILTAG_DICTS = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5":  cv2.aruco.DICT_APRILTAG_16h5,
}


def load_camera_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera calibration outputs from file

    Args:
        path: The path to the calibration `.npz` file

    Returns:
        tuple: A tuple containing:
        - mtx(np.ndarray): Camera matrix as a float64 array
        - dist(np.ndarray): Distortion coefficients as a float64 array
    """
    data = np.load(str(path))
    return data["mtx"].astype(np.float64), data["dist"].astype(np.float64)


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


def draw_axes(img: np.ndarray,
              K: np.ndarray,
              dist: np.ndarray,
              rvec: np.ndarray,
              tvec: np.ndarray,
              length: float) -> None:
    """Draws 3D coordinate axes on an image using camera projection.

    Args:
        img (np.ndarray): The input image on which axes will be drawn.
        K (np.ndarray): Camera intrinsic matrix (3x3).
        dist (np.ndarray): Distortion coefficients.
        rvec (np.ndarray): Rotation vector (3x1).
        tvec (np.ndarray): Translation vector (3x1).
        length (float): The length of each axis line to be drawn in 3D units.
    Returns:
        None
    """

    axis_pts = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ], dtype=np.float32)

    projected, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)

    pts = [tuple(map(int, p)) for p in projected]

    origin = pts[0]
    cv2.line(img, origin, pts[1], (0, 0, 255), 3)
    cv2.line(img, origin, pts[2], (0, 255, 0), 3)
    cv2.line(img, origin, pts[3], (255, 0, 0), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calibration_output/calibration.npz")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--tag-size", type=float, required=True,
                    help="Tag edge length in metres, e.g. 0.08")
    ap.add_argument("--family", default="tag36h11",
                    choices=APRILTAG_DICTS.keys())
    args = ap.parse_args()

    K, dist = load_camera_calibration(args.calib)
    objp = tag_object_points(args.tag_size)
    axis_len = args.tag_size * 0.6

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[args.family])
    detector = cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    print("Press Q or ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners_list,
                                          ids.reshape(-1, 1))

            for corners, tag_id in zip(corners_list, ids):
                imgp = corners.reshape(4, 2).astype(np.float32)

                ok_pnp, rvec, tvec = cv2.solvePnP(
                    objp, imgp, K.astype(np.float32), dist.astype(np.float32),
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok_pnp:
                    continue

                draw_axes(frame, K, dist, rvec, tvec, axis_len)

                t = tvec.reshape(3)
                cv2.putText(
                    frame,
                    f"id={tag_id} t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m",
                    (10, 30 + 25 * int(tag_id % 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

        cv2.imshow("AprilTag Pose Demo", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
