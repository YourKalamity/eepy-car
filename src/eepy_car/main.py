import concurrent.futures
import datetime as dt
import sys
import time

import cv2
import numpy as np

from eepy_car.alert import AlertLevel, AlertManager, DriverState
from eepy_car.capture import CaptureManager
from eepy_car.config import load_config
from eepy_car.distraction import (
    build_tag_detector,
    compute_gaze_and_pose_diff,
    detect_tags,
    estimate_tag_pose,
    load_camera_calibration,
    tag_object_points,
)
from eepy_car.drowsiness import avg_ear, get_face_data, load_landmarker_model, mar
from eepy_car.output.audio import play_alert
from eepy_car.output.logger import log_alert, setup_logger
from eepy_car.output.overlay import draw_overlay


def preflight_checks(config: dict) -> None:
    """Checks necessary files exist before starting

    Args:
        config (dict): Loaded configuration dictionary.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    from pathlib import Path

    required = {
        "Model": config["model"]["path"],
        "Calibration": config["calibration"]["path"],
    }

    for name, path in required.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} file not found at '{path}'.\n"
            )


def process_face_branch(face_landmarker,
                        frame: np.ndarray) -> tuple[list[tuple[float, float]] | None,
                                                    np.ndarray | None,
                                                    float | None,
                                                    float | None]:
    """Runs MediaPipe branch and extracts face landmarks, pose matrix and calculates EAR and MAR

    Args:
        face_landmarker (vision.FaceLandmarker): The loaded landmarker model.
        frame (np.ndarray): A single BGR frame from OpenCV.

    Returns:
        tuple[list[tuple[float, float]] | None, np.ndarray | None, float | None, float | None]: 
            landmarks, pose_matrix, ear_value, mar_value

    """
    landmarks, pose_matrix = get_face_data(face_landmarker, frame)
    ear_value = None
    mar_value = None

    if landmarks is not None:
        ear_value = avg_ear(landmarks)
        mar_value = mar(landmarks)

    return landmarks, pose_matrix, ear_value, mar_value


def process_tag_branch(tag_detector: cv2.aruco.ArucoDetector,
                       frame: np.ndarray,
                       tag_objp: np.ndarray,
                       camera_matrix: np.ndarray,
                       camera_dist: np.ndarray,
                       headrest_tag_id: int) -> tuple[dict[int, np.ndarray],
                                                      tuple[np.ndarray, np.ndarray] | None]:
    """Runs AprilTag branch and extarcts AprilTag coordinates as well as headrest tag pose

    Args:
        tag_detector (cv2.aruco.ArucoDetector): The ArucoDetector to run
        frame (np.ndarray): A single BGR frame from OpenCV.
        tag_objp (np.ndarray): The tag object points
        camera_matrix (np.ndarray): Calibration matrix
        camera_dist (np.ndarray): Calibration dist
        headrest_tag_id (int): Tag of headrest ID

    Returns:
        tuple[dict[int, np.ndarray], tuple[np.ndarray, np.ndarray] | None]: found_tags, tag_pose
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found_tags = detect_tags(tag_detector, gray)

    headrest_corners = found_tags.get(headrest_tag_id)
    tag_pose = None

    if headrest_corners is not None:
        tag_pose = estimate_tag_pose(
            headrest_corners, tag_objp, camera_matrix, camera_dist
        )

    return found_tags, tag_pose


def main() -> int:
    # Load config
    try:
        config = load_config("config.json")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Preflight checks
    try:
        preflight_checks(config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Set up logger
    logger = setup_logger(
        config["output"]["log_path"],
    )
    logger.info("eepy-car starting up")
    logger.info(f"Config values: {config}")

    # Initialise detection components
    logger.info("Loading face landmarker model...")
    face_landmarker = load_landmarker_model(config["model"]["path"])

    logger.info("Building AprilTag detector and object points...")
    tag_detector = build_tag_detector(config["apriltag"]["family"])
    tag_objp = tag_object_points(config["apriltag"]["tag_size_metres"])

    logger.info("Loading camera calibration...")
    camera_matrix, camera_dist = load_camera_calibration(config["calibration"]["path"])

    # Create alert system
    driver_state = DriverState(config)

    def on_alert(level: AlertLevel) -> None:
        log_alert(logger, level)
        if config["output"].get("audio_alert"):
            play_alert(level, config)

    alert_manager = AlertManager(config, on_alert=on_alert)

    # Runtime state
    last_known_rvec = None
    last_known_tvec = None
    tag_found = False
    logger.info("Opening camera...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            with CaptureManager(config["camera"]["index"]) as cap:

                logger.info("Pipeline running")
                logger.info("press Q or ESC to quit")
                logger.info("Looking for headrest tag...")
                logger.info("Warming up camera...")
                max_warmup_attempts = 30
                for attempt in range(max_warmup_attempts):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        logger.info("Camera ready")
                        break
                    logger.debug(f"Waiting for camera... attempt {attempt + 1}/{max_warmup_attempts}")
                    time.sleep(0.01)
                else:
                    logger.error(
                        "Camera failed to produce a frame after "
                        f"{max_warmup_attempts} attempts. "
                        "Check your camera connection and try again."
                    )
                    return 1
                
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        logger.error("Failed to read frame from camera")
                        break

                    now = dt.datetime.now()

                    # Run both detection branches concurrently
                    # Face Detection
                    future_face = executor.submit(process_face_branch, face_landmarker, frame)
                    # AprilTag Detection
                    future_tags = executor.submit(
                        process_tag_branch,
                        tag_detector,
                        frame,
                        tag_objp,
                        camera_matrix,
                        camera_dist,
                        config["apriltag"]["headrest_tag_id"]
                    )

                    # Wait for result
                    landmarks, pose_matrix, ear_value, mar_value = future_face.result()
                    found_tags, tag_pose = future_tags.result()

                    # Gaze Estimation
                    if tag_pose is not None:
                        last_known_rvec, last_known_tvec = tag_pose
                        if not tag_found:
                            tag_found = True
                            logger.info("Headrest tag found: Driver can now sit down.")

                    yaw_value = None
                    pitch_value = None

                    if (pose_matrix is not None and
                            last_known_rvec is not None and
                            last_known_tvec is not None):
                        gaze, distance = compute_gaze_and_pose_diff(
                            pose_matrix, last_known_rvec, last_known_tvec
                        )
                        if gaze is not None:
                            yaw_value, pitch_value = gaze

                    # Update scores
                    driver_state.update_scores(ear_value, mar_value, yaw_value, pitch_value, now)

                    # Alert level evaluation
                    alert_level = alert_manager.evaluate(driver_state, now)

                    # Overlay drawing
                    if config["output"].get("show_overlay", True):
                        draw_overlay(
                            frame=frame,
                            ear_value=ear_value,
                            mar_value=mar_value,
                            yaw_value=yaw_value,
                            pitch_value=pitch_value,
                            ear_score=driver_state.ear_score,
                            mar_score=driver_state.mar_score,
                            yaw_score=driver_state.yaw_score,
                            pitch_score=driver_state.pitch_score,
                            drowsiness_score=alert_manager._drowsiness_score(driver_state),
                            distraction_score=alert_manager._distraction_score(driver_state),
                            alert_level=alert_level,
                            landmarks=landmarks,
                            found_tags=found_tags,
                            headrest_tag_rvec=last_known_rvec,
                            headrest_tag_tvec=last_known_tvec,
                            camera_matrix=camera_matrix,
                            camera_dist=camera_dist,
                            config=config,
                        )

                    cv2.imshow("eepy-car", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

    except KeyboardInterrupt:
        pass
    finally:
        face_landmarker.close()
        logger.info("eepy-car shut down cleanly")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
