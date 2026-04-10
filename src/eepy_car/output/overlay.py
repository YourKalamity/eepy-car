import cv2
import numpy as np

from eepy_car.alert import AlertLevel
from eepy_car.drowsiness.ear import LEFT_EYE, RIGHT_EYE
from eepy_car.drowsiness.mar import MOUTH


def draw_overlay(frame: np.ndarray,
                 ear_value: float | None,
                 mar_value: float | None,
                 yaw_value: float | None,
                 pitch_value: float | None,
                 ear_score: float | None,
                 mar_score: float | None,
                 yaw_score: float | None,
                 pitch_score: float | None,
                 drowsiness_score: float | None,
                 distraction_score: float | None,
                 alert_level: AlertLevel | None,
                 landmarks: list[tuple[float, float]] | None,
                 found_tags: dict[int, np.ndarray] | None,
                 headrest_tag_rvec: np.ndarray | None,
                 headrest_tag_tvec: np.ndarray | None,
                 camera_matrix: np.ndarray | None,
                 camera_dist: np.ndarray | None,
                 config: dict) -> None:
    """Overlay drawer that calls helper functions depending on config values
    Handles None values by either not drawing or displaying N / A

    Args:
        frame (np.ndarray): The frame to draw on
        ear_value (float | None): The calculated EAR value
        mar_value (float | None): The calculated MAR value
        yaw_value (float | None): The yaw offset from headrest tag
        pitch_value (float | None): The pitch offset from headrest tag
        ear_score (float | None): The accumulated EAR score
        mar_score (float | None): The accumulated MAR score
        yaw_score (float | None): The accumulated yaw score
        pitch_score (float | None): The accumulated pitch score
        drowsiness_score (float | None): The drowsiness score
        distraction_score (float | None): The distraction score
        alert_level (AlertLevel | None): The current alert level
        landmarks (list[tuple[float, float]] | None): The facial landmark points
        found_tags (dict[int, np.ndarray] | None): The dictionary of all the found tags
        headrest_tag_rvec (np.ndarray | None): Headrest tags rotation vector
        headrest_tag_tvec (np.ndarray | None): Headrest tags translation vector
        camera_matrix (np.ndarray | None): Camera matrix from calibration
        camera_dist (np.ndarray | None): Camera dist from calibration
        config (dict): Configuration file dictionary
    """
    y_start = 30
    y_start = _draw_indicator_values(frame,
                                     y_start,
                                     ear_value,
                                     mar_value,
                                     yaw_value,
                                     pitch_value)

    overlay_cfg = config["output"]["overlay"]
    y_start = 30

    if overlay_cfg.get("show_indicator_values"):
        y_start = _draw_indicator_values(frame,
                                         y_start,
                                         ear_value,
                                         mar_value,
                                         yaw_value,
                                         pitch_value)

    if overlay_cfg.get("show_indicator_scores"):
        y_start = _draw_indicator_scores(frame,
                                         y_start,
                                         ear_score,
                                         mar_score,
                                         yaw_score,
                                         pitch_score)

    if overlay_cfg.get("show_combined_scores"):
        y_start = _draw_combined_scores(frame,
                                        y_start,
                                        drowsiness_score,
                                        distraction_score)

    if overlay_cfg.get("show_alert_level") and alert_level is not None:
        _draw_alert_level(frame, y_start, alert_level)

    if overlay_cfg.get("show_face_mesh") and landmarks is not None:
        _draw_face_landmarks(frame, landmarks)

    if overlay_cfg.get("show_ear_points") and landmarks is not None:
        _draw_ear_landmarks(frame, landmarks)

    if overlay_cfg.get("show_mar_points") and landmarks is not None:
        _draw_mar_landmarks(frame, landmarks)

    if overlay_cfg.get("show_apriltag") and found_tags:
        _draw_found_tags(frame, found_tags)

    if (overlay_cfg.get("show_apriltag_axes") and
            headrest_tag_rvec is not None and
            headrest_tag_tvec is not None and
            camera_matrix is not None and
            camera_dist is not None):
        _draw_headrest_tag_axes(frame,
                                headrest_tag_rvec,
                                headrest_tag_tvec,
                                camera_matrix,
                                camera_dist,
                                config)


def _draw_indicator_values(frame: np.ndarray,
                           y_start: int,
                           ear_value: float | None,
                           mar_value: float | None,
                           yaw_value: float | None,
                           pitch_value: float | None) -> int:
    """Helper function to draw indicator values if available

    Args:
        frame (np.ndarray): The frame to draw on
        y_start (int): The y position to start writing from
        ear_value (float | None): The calculated EAR value
        mar_value (float | None): The calculated MAR value
        yaw_value (float | None): The yaw offset from headrest tag
        pitch_value (float | None): The pitch offset from headrest tag

    Returns:
        int: The next y_start value to draw at
    """
    lines = []

    if ear_value is not None:
        lines.append(f"EAR:   {ear_value:.3f}")
    else:
        lines.append("EAR:   N / A")

    if mar_value is not None:
        lines.append(f"MAR:   {mar_value:.3f}")
    else:
        lines.append("MAR:   N / A")

    if yaw_value is not None:
        lines.append(f"Yaw:   {yaw_value:.1f}°")
    else:
        lines.append("Yaw:   N / A")

    if pitch_value is not None:
        lines.append(f"Pitch: {pitch_value:.1f}°")
    else:
        lines.append("Pitch: N / A")

    for i, line in enumerate(lines):
        cv2.putText(frame,
                    line,
                    (10, y_start + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                    )

    y_start += len(lines) * 28 + 10
    return y_start


def _draw_indicator_scores(frame: np.ndarray,
                           y_start: int,
                           ear_score: float | None,
                           mar_score: float | None,
                           yaw_score: float | None,
                           pitch_score: float | None) -> int:
    """Helper function to draw indicator scores if available

    Args:
        frame (np.ndarray): The frame to draw on
        y_start (int): The y position to start writing from
        ear_score (float | None): The accumulated EAR score
        mar_score (float | None): The accumulated MAR score
        yaw_score (float | None): The accumulated yaw score
        pitch_score (float | None): The accumulated pitch score

    Returns:
        int: The next y_start value to draw at
    """
    lines = []

    if ear_score is not None:
        lines.append(f"EAR score:   {ear_score:.3f}")
    else:
        lines.append("EAR score:   N / A")

    if mar_score is not None:
        lines.append(f"MAR score:   {mar_score:.3f}")
    else:
        lines.append("MAR score:   N / A")

    if yaw_score is not None:
        lines.append(f"Yaw score:   {yaw_score:.3f}")
    else:
        lines.append("Yaw score:   N / A")

    if pitch_score is not None:
        lines.append(f"Pitch score: {pitch_score:.3f}")
    else:
        lines.append("Pitch score: N / A")

    for i, line in enumerate(lines):
        cv2.putText(frame,
                    line,
                    (10, y_start + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                    )

    y_start += len(lines) * 28 + 10
    return y_start


def _draw_combined_scores(frame: np.ndarray,
                          y_start: int,
                          drowsiness_score: float | None,
                          distraction_score: float | None) -> int:
    """Helper function to draw combined scores if available

    Args:
        frame (np.ndarray): The frame to draw on
        y_start (int): The y position to start writing from
        drowsiness_score (float | None): The drowsiness score
        distraction_score (float | None): The distraction score

    Returns:
        int: The next y_start value to draw at
    """
    lines = []

    if drowsiness_score is not None:
        lines.append(f"Drowsiness score:   {drowsiness_score:.3f}")
    else:
        lines.append("Drowsiness score:   N / A")

    if distraction_score is not None:
        lines.append(f"Distraction score:   {distraction_score:.3f}")
    else:
        lines.append("Distraction score:   N / A")

    for i, line in enumerate(lines):
        cv2.putText(frame,
                    line,
                    (10, y_start + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                    )

    y_start += len(lines) * 28 + 10
    return y_start


def _draw_alert_level(frame: np.ndarray,
                      y_start: int,
                      alert_level: AlertLevel) -> int:
    """Helper function to draw the current AlertLevel

    Args:
        frame (np.ndarray): The frame to draw on
        y_start (int): The y position to start writing from
        alert_level (AlertLevel): The current alert level

    Returns:
        int: The next y_start value to draw at
    """
    if alert_level is None or alert_level == AlertLevel.NONE:
        return y_start

    label_map = {
        AlertLevel.DISTRACTION_WARNING: "DISTRACTION",
        AlertLevel.DROWSINESS_WARNING: "DROWSINESS",
        AlertLevel.CRITICAL_DISTRACTION: "CRITICAL DISTRACTION",
        AlertLevel.CRITICAL_DROWSINESS: "CRITICAL DROWSINESS",
    }

    color_map = {
        AlertLevel.DISTRACTION_WARNING: (0, 165, 255),
        AlertLevel.DROWSINESS_WARNING: (0, 165, 255),
        AlertLevel.CRITICAL_DISTRACTION: (0, 0, 255),
        AlertLevel.CRITICAL_DROWSINESS: (0, 0, 255),
    }

    text = label_map.get(alert_level)
    if text is None:
        return y_start

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    color = color_map[alert_level]

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, frame.shape[1] - text_w - 10)
    y = max(text_h + 10, y_start + text_h)

    cv2.putText(frame, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)

    return y + baseline + 10


def _draw_face_landmarks(frame: np.ndarray,
                         landmarks: list[tuple[float, float]]) -> None:
    """Helper function to draw all 478 face landmarks

    Args:
        frame (np.ndarray): The frame to draw on
        landmarks (list[tuple[float, float]]): The face landmarks
    """
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), -1)


def _draw_ear_landmarks(frame: np.ndarray,
                        landmarks: list[tuple[float, float]]) -> None:
    """Helper function to draw EAR landmarks

    Args:
        frame (np.ndarray): The frame to draw on
        landmarks (list[tuple[float, float]]): The face landmarks
    """
    for idx in LEFT_EYE + RIGHT_EYE:
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def _draw_mar_landmarks(frame: np.ndarray,
                        landmarks: list[tuple[float, float]]) -> None:
    """Helper function to draw MAR landmarks

    Args:
        frame (np.ndarray): The frame to draw on
        landmarks (list[tuple[float, float]]): The face landmarks
    """
    for idx in MOUTH:
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def _draw_found_tags(frame: np.ndarray,
                     found_tags: dict[int, np.ndarray]) -> None:
    """Helper function to draw overays for AprilTags detected as well as ID

    Args:
        frame (np.ndarray): The frame to draw on
        found_tags (dict[int, np.ndarray]): The dictionary with all the AprilTags found
    """
    for tag_id, corners in found_tags.items():
        pts = corners.reshape(4, 2).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID:{tag_id}", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# These points were defined outside of the function to avoid them being recreated constantly
_AXIS_PTS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
], dtype=np.float32)


def _draw_headrest_tag_axes(frame: np.ndarray,
                            headrest_tag_rvec: np.ndarray,
                            headrest_tag_tvec: np.ndarray,
                            camera_matrix: np.ndarray,
                            camera_dist: np.ndarray,
                            config: dict) -> None:
    """Helper function to draw 3D axes on specified AprilTag

    Args:
        frame (np.ndarray): The frame to draw on
        headrest_tag_rvec (np.ndarray | None): Headrest tags rotation vector
        headrest_tag_tvec (np.ndarray | None): Headrest tags translation vector
        camera_matrix (np.ndarray | None): Camera matrix from calibration
        camera_dist (np.ndarray | None): Camera dist from calibration
        config (dict): Configuration file dictionary
    """
    length = config["apriltag"]["tag_size_metres"] * 0.6

    projected, _ = cv2.projectPoints(_AXIS_PTS * length,
                                     headrest_tag_rvec,
                                     headrest_tag_tvec,
                                     camera_matrix,
                                     camera_dist)
    projected = projected.reshape(-1, 2)

    pts = [tuple(map(int, p)) for p in projected]

    origin = pts[0]
    cv2.line(frame, origin, pts[1], (0, 0, 255), 3)
    cv2.line(frame, origin, pts[2], (0, 255, 0), 3)
    cv2.line(frame, origin, pts[3], (255, 0, 0), 3)
