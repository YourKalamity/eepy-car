import cv2
import datetime as dt

from april_tag import draw_axes
from eepy_car.config import load_config
from eepy_car.capture import CaptureManager
from eepy_car.drowsiness import load_landmarker_model, get_face_data, avg_ear, mar
from eepy_car.distraction import build_tag_detector, detect_tags, estimate_tag_pose, tag_object_points, load_camera_calibration, compute_gaze_and_pose_diff
from eepy_car.alert import AlertLevel, AlertManager, DriverState

SHOW_FACE_MESH = False
SHOW_EAR_POINTS = False
SHOW_MAR_POINTS = False
SHOW_APRILTAG = False
SHOW_APRILTAG_AXES = False

if __name__ == "__main__":
    config = load_config("config.json")
    face_landmarker = load_landmarker_model("models/face_landmarker.task")
    april_tag_detector = build_tag_detector(config["apriltag"]["family"])
    tag_objp = tag_object_points(config["apriltag"]["tag_size_metres"])
    camera_matrix, camera_dist = load_camera_calibration(
        "calibration_output/calibration.npz")
    last_known_rvec = None
    last_known_tvec = None
    driver_state = DriverState(config)
    alert_manager = AlertManager(config, print)
    with CaptureManager(config["camera"]["index"]) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            landmarks, pose_matrix = get_face_data(face_landmarker, frame)
            if SHOW_FACE_MESH and landmarks is not None:
                for (x, y) in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)

            if SHOW_EAR_POINTS and landmarks is not None:
                from eepy_car.drowsiness.ear import LEFT_EYE, RIGHT_EYE
                for idx in LEFT_EYE + RIGHT_EYE:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            if SHOW_MAR_POINTS and landmarks is not None:
                from eepy_car.drowsiness.mar import MOUTH
                for idx in MOUTH:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
            ear_value = None
            mar_value = None
            if landmarks is not None and pose_matrix is not None:
                ear_value = avg_ear(landmarks)
                mar_value = mar(landmarks)

            found_tags = detect_tags(april_tag_detector, gray_frame)
            tag_pose = None
            headrest_corners = found_tags.get(config["apriltag"]["headrest_tag_id"])

            if SHOW_APRILTAG and found_tags:
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
            if headrest_corners is not None:
                tag_pose = estimate_tag_pose(found_tags[config["apriltag"]["headrest_tag_id"]],
                                             tag_objp,
                                             camera_matrix,
                                             camera_dist)
                if tag_pose is not None:
                    last_known_rvec, last_known_tvec = tag_pose

            if SHOW_APRILTAG_AXES and last_known_rvec is not None and last_known_tvec is not None:
                axis_len = config["apriltag"]["tag_size_metres"] * 0.6
                draw_axes(frame, camera_matrix, camera_dist, last_known_rvec, last_known_tvec, axis_len)
            gaze = None
            distance = None
            rvec = last_known_rvec
            tvec = last_known_tvec
            if pose_matrix is not None and rvec is not None and tvec is not None:
                gaze, distance = compute_gaze_and_pose_diff(pose_matrix, rvec, tvec)

            if ear_value is not None:
                cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if mar_value is not None:
                cv2.putText(frame, f"MAR: {mar_value:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            yaw = None
            pitch = None
            if gaze is not None:
                yaw, pitch = gaze
                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if distance is not None:
                cv2.putText(frame, f"Dist: {distance:.3f}m", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            now = dt.datetime.now()

            if ear_value is not None and mar_value is not None and yaw is not None and pitch is not None:
                driver_state.update_scores(ear_value, mar_value, yaw, pitch, now)
            alert = alert_manager.evaluate(driver_state, now)

            if alert != AlertLevel.NONE:
                cv2.putText(frame, f"ALERT: {alert.name}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            drowsiness = alert_manager._drowsiness_score(driver_state)
            distraction = alert_manager._distraction_score(driver_state)

            cv2.putText(frame, f"EAR score: {driver_state.ear_score:.3f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"MAR score: {driver_state.mar_score:.3f}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Yaw score: {driver_state.yaw_score:.3f}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Pitch score: {driver_state.pitch_score:.3f}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Drowsiness: {drowsiness:.3f}", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Distraction: {distraction:.3f}", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.imshow("eepy_car test harness", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
