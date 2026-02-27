import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def build_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """Builds the 3D object points for a chessboard.

    Args:
        cols: The number of columns of the chessboard
        rows: The number of rows of the chessboard
        square_size: The physical length of a square side
    Returns:
        np.ndarray: The 3D coordinates for each corner of each square in the chessboard
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def find_corners(gray, pattern_size) -> tuple[bool, None | cv2.typing.MatLike]:
    """Attempts to find corners of a chessboard in a grayscale image

    Args:
        gray: The image to process
        pattern_size: Tuple of (columns, rows)

    Returns:
        Tuple with (False, None) if not found else (True, corners)
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not ret:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def run_calibration(objpoints: list[np.ndarray], imgpoints: list[np.ndarray], image_size: tuple[int, int]) -> dict:
    """Runs cv2.calibrateCamera and returns the results as a dict.

    Args:
        objpoints: List of 3D object point arrays, one per calibration image.
            Each array has shape (N, 3) where N is the number of corners.
        imgpoints: List of 2D image point arrays corresponding to objpoints.
            Each array has shape (N, 1, 2) as returned by cv2.cornerSubPix.
        image_size: Width and height of the calibration images as (w, h).

    Returns:
        A dictionary

    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size,
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
    )
    return {"rms": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}


def save_calibration(out_dir: str | Path, result: dict, image_size: tuple[int, int]) -> None:
    """Saves calibration to both .npz and .json formats

    Args:
        out_dir: Path to the output directory. Created if it doesn't exist.
        result: Dict returned by run_calibration, containing 'mtx' and 'dist'.
        image_size: Width and height of the calibration images as (w, h).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "calibration.npz",
        mtx=result["mtx"],
        dist=result["dist"],
        image_size=np.array(image_size),
    )

    payload = {
        "rms_reprojection_error": float(result["rms"]),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": result["mtx"].tolist(),
        "dist_coeffs": result["dist"].tolist(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved calibration to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cols", type=int, default=9,
                    help="Inner corners per row")
    ap.add_argument("--rows", type=int, default=6,
                    help="Inner corners per column")
    ap.add_argument("--square", type=float, default=0.025,
                    help="Square size in metres")
    ap.add_argument("--samples", type=int, default=10,
                    help="How many captures to take")
    ap.add_argument("--out", type=str, default="calibration_output")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square)
    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    objpoints, imgpoints = [], []
    count = 0
    last_msg = ""

    print(f"Press SPACE to capture. Need {args.samples} good captures.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size)

        vis = frame.copy()
        if found and corners is not None:
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)

        cv2.putText(vis, f"{count}/{args.samples} | found={found}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if last_msg:
            cv2.putText(vis, last_msg, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break

        if key == 32:  # space
            if not found:
                last_msg = "No chessboard found, try again"
                continue

            objpoints.append(objp.copy())
            imgpoints.append(corners)
            img_path = out_dir / "images" / f"calib_{count:02d}.jpg"
            cv2.imwrite(str(img_path), frame)
            count += 1
            last_msg = f"Captured {count}/{args.samples}"

            if count >= args.samples:
                h, w = frame.shape[:2]
                print("Running calibration...")
                result = run_calibration(objpoints, imgpoints, (w, h))
                print(f"RMS error: {result['rms']:.4f}")
                save_calibration(out_dir, result, (w, h))
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
