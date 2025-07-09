import os
import cv2
import numpy as np
from collections import deque

IOU_THRESHOLD = 0.5
MAX_LOST_FRAMES = 5

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

class PlayerTracker:
    def __init__(self):
        self.players = {}
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        updated_players = {}
        used_ids = set()
        for det in detections:
            best_iou = 0
            best_id = None
            for pid, pdata in self.players.items():
                iou = compute_iou(det, pdata['box'])
                if iou > best_iou and iou > IOU_THRESHOLD and pid not in used_ids:
                    best_iou = iou
                    best_id = pid
            if best_id is not None:
                updated_players[best_id] = {'box': det, 'lost': 0}
                self.tracks[best_id].append(get_centroid(det))
                used_ids.add(best_id)
            else:
                updated_players[self.next_id] = {'box': det, 'lost': 0}
                self.tracks[self.next_id] = deque([get_centroid(det)], maxlen=30)
                self.next_id += 1
        for pid, pdata in self.players.items():
            if pid not in updated_players:
                pdata['lost'] += 1
                if pdata['lost'] <= MAX_LOST_FRAMES:
                    updated_players[pid] = pdata
        self.players = updated_players
        return self.players

def run_tracking(frames_folder, output_video):
    tracker = PlayerTracker()
    frame_files = sorted(os.listdir(frames_folder))
    if not frame_files:
        print(f"No frames found in {frames_folder}")
        return
    example_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    if example_frame is None:
        print(f"Could not read example frame: {frame_files[0]}")
        return
    h, w, _ = example_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 25, (w, h))
    for f in frame_files:
        frame_path = os.path.join(frames_folder, f)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {f}, skipping.")
            continue
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(cv2.Canny(gray, 50, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w_, h_ = cv2.boundingRect(cnt)
            if w_ > 40 and h_ > 100:
                detections.append((x, y, x + w_, y + h_))
        tracked_players = tracker.update(detections)
        for pid, pdata in tracked_players.items():
            x1, y1, x2, y2 = pdata['box']
            color = (0, 255, pid * 17 % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {pid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out.write(frame)
    out.release()
    print(f"[DONE] Tracking complete. Video saved to: {output_video}")

if __name__ == "__main__":
    frames_folder = "../output/frames"
    output_video = "../output/tracked_output.mp4"
    run_tracking(frames_folder, output_video)
