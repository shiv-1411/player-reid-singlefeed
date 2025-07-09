import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

model_path = '../model/yolo.pt'
video_path = '../videos/15sec_input_720p.mp4'
output_folder = '../output/frames'
os.makedirs(output_folder, exist_ok=True)

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_detections = []

print("[INFO] Running detection on each frame...")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0 and conf > 0.4:
            frame_detections.append({
                'frame': frame_idx,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'conf': conf
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite(f"{output_folder}/frame_{frame_idx:04d}.jpg", frame)
    frame_idx += 1

cap.release()
print(f"[DONE] Processed {frame_idx} frames and saved to: {output_folder}")
