import os
import sys

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14]

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.60


def load_landmarker(model_path):
    """Loads the MediaPipe face landmarker model from a .task file."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_landmarks(landmarker, frame_bgr):
    """
    Returns pixel coords for all 478 landmarks as a list of (x, y) tuples.
    Returns None if no face detected.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    return [(lm[i].x * w, lm[i].y * h) for i in range(len(lm))]


def ear(landmarks, indices):
    """
    Eye Aspect Ratio from Soukupova & Cech (2016).
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Returns 0.0 if the eye width is basically zero (shouldn't happen normally).
    """
    p = [np.array(landmarks[i]) for i in indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def mar(landmarks, indices):
    """
    Mouth Aspect Ratio using 4 points.
    Vertical distance between inner lips divided by mouth width.
    Should be close to 0.0 when closed, rises clearly when yawning.
    """
    left = np.array(landmarks[indices[0]])
    right = np.array(landmarks[indices[1]])
    top = np.array(landmarks[indices[2]])
    bottom = np.array(landmarks[indices[3]])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def main():
    landmarker = load_landmarker(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            landmarks = get_landmarks(landmarker, frame)

            if landmarks is None:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                left = ear(landmarks, LEFT_EYE)
                right = ear(landmarks, RIGHT_EYE)
                avg_ear = (left + right) / 2.0
                avg_mar = mar(landmarks, MOUTH)

                ear_colour = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
                mar_colour = (0, 0, 255) if avg_mar > MAR_THRESHOLD else (0, 255, 0)

                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_colour, 2)
                cv2.putText(frame, f"MAR: {avg_mar:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, mar_colour, 2)
                cv2.putText(frame, f"L={left:.3f}  R={right:.3f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Draw the eye and mouth landmark points so I can see if the
                # indices are correct
                for idx in LEFT_EYE + RIGHT_EYE:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                for idx in MOUTH:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

            cv2.imshow("EAR / MAR test (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
