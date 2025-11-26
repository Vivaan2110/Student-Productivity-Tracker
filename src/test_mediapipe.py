# src/test_mediapipe.py
import cv2
import mediapipe as mp
from pathlib import Path

VIDEO = "data/raw/lab_video.mov"
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)
OUT_IMG = OUT / "test_output.jpg"

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()
if not ret:
    raise SystemExit("Can't read video: " + VIDEO)

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# face detection
with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
    face_res = face_detector.process(image_rgb)

# pose detection
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_detector:
    pose_res = pose_detector.process(image_rgb)

out = frame.copy()
h,w = out.shape[:2]

# draw faces
if face_res.detections:
    for det in face_res.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w); y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w); y2 = y1 + int(bbox.height * h)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)

# draw pose landmarks
if pose_res.pose_landmarks:
    mp_drawing.draw_landmarks(out, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cv2.imwrite(str(OUT_IMG), out)
print("Saved:", OUT_IMG)