import cv2
from facenet_pytorch import MTCNN
from PIL import Image

VIDEO = "/Users/Vivaan/Documents/Smart Productivity Tracker(student monitoring)/data/raw/lab_video.mov"

mtcnn = MTCNN(image_size=160, margin=2, device="cpu")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("Cannot open video")

frame_i = 0
while frame_i < 100:  # just first ~100 frames
    ret, frame = cap.read()
    if not ret:
        break
    frame_i += 1

    # no downscale for this test
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil)
    if boxes is not None:
        print(f"frame {frame_i}: {len(boxes)} faces, probs={probs}")
    else:
        print(f"frame {frame_i}: no faces")

cap.release()