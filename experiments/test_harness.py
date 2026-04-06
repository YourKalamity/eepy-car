import cv2

from eepy_car.config import load_config
from eepy_car.capture import CaptureManager
from eepy_car.drowsiness import load_landmarker_model, get_face_data, avg_ear, mar
from eepy_car.distraction import build_tag_detector, detect_tags, estimate_tag_pose, tag_object_points, load_camera_calibration, compute_gaze_and_pose_diff

if __name__ == "__main__":
    config = load_config("config.json")
    face_landmarker = load_landmarker_model("models/face_landmarker.task")
    april_tag_detector = build_tag_detector(config["apriltag"]["family"])
    tag_objp = tag_object_points(config["apriltag"]["tag_size_metres"])
    camera_matrix, camera_dist = load_camera_calibration(
        "calibration_output/calibration.npz")
    last_known_rvec = None
    last_known_tvec = None
    with CaptureManager(config["camera"]["index"]) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            landmarks, pose_matrix = get_face_data(face_landmarker, frame)
            ear_value = None
            mar_value = None
            if landmarks is not None and pose_matrix is not None:
                ear_value = avg_ear(landmarks)
                mar_value = mar(landmarks)

            found_tags = detect_tags(april_tag_detector, gray_frame)
            tag_pose = None
            headrest_corners = found_tags.get(config["apriltag"]["headrest_tag_id"])

            if headrest_corners is not None:
                tag_pose = estimate_tag_pose(found_tags[config["apriltag"]["headrest_tag_id"]],
                                             tag_objp,
                                             camera_matrix,
                                             camera_dist)
                if tag_pose is not None:
                    last_known_rvec, last_known_tvec = tag_pose

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

            if gaze is not None:
                yaw, pitch = gaze
                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if distance is not None:
                cv2.putText(frame, f"Dist: {distance:.3f}m", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("eepy_car test harness", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
