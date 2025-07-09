# ⚽ Player Re-Identification in a Single Feed

**Assignment:** AI Intern @ Liat.ai  
**Task:** Option 2 – Re-Identification in a Single Feed  
**Author:** Shivam Bhardwaj

---

## 🏁 Problem Statement

Given a 15-second soccer video from a single camera, assign consistent `player_id`s to each player — even if they leave and re-enter the frame.

---

## 🎯 Objectives

- **Accurately detect** all players in each frame using a custom YOLOv11 model.
- **Track and re-identify** players across frames using IoU and centroid logic.
- **Maintain consistent IDs** for players, even after occlusions or re-entries.
- **Generate a final video** with stable `player_id`s displayed on screen.

---

## 🧰 Tech Stack

- **Python 3.8+**
- **Ultralytics YOLOv11** (custom fine-tuned)
- **OpenCV**
- **TQDM**

---

## 📂 Folder Structure

```
player-reid-singlefeed/
├── model/
│   └── yolo.pt                # Provided detection model
├── videos/
│   └── 15sec_input_720p.mp4   # Input video
├── src/
│   ├── detect.py              # Player detection using YOLO
│   ├── track.py               # Re-identification & tracking
├── output/
│   ├── frames/                # Frames with detection boxes
│   └── tracked_players.mp4    # Final output video
├── README.md
└── report.md                  # Approach, results & analysis
```

---

## 🚀 Quick Start

### 1️⃣ Install dependencies

```bash
pip install ultralytics opencv-python tqdm
```

### 2️⃣ Run player detection

```bash
cd src
python detect.py
```

- ✅ Detected bounding boxes saved as images in `/output/frames/`

### 3️⃣ Run re-identification & tracking

```bash
python track.py
```

- ✅ Final tracked video saved as `/output/tracked_players.mp4`

---

## 🧠 Approach

### 🔍 Detection

- Utilized the provided YOLOv11 model for robust player detection.
- Processed each frame individually, filtering detections by confidence (>0.4).
- Saved annotated frames for downstream tracking.

### 🔁 Tracking & Re-Identification

- Developed a lightweight tracking system:
  - Matched detections using **IoU (Intersection-over-Union)**
  - Used **centroid distance** for smoother tracking
  - Managed active `player_id`s with a custom Python class
  - IDs are retained for players re-entering the frame (unless unmatched for >5 frames)

---

## 🚧 Challenges

- Distinguishing between visually similar players
- Handling occlusions and missed detections
- Occasional identity switches in crowded scenes

---

## 🏆 Results

- Consistently retained `player_id`s throughout the video
- Output video `tracked_players.mp4` clearly displays tracked boxes and IDs
- IDs remain stable even when players leave and re-enter the frame

---

## 🚀 Potential Improvements

- Integrate **Deep SORT** or embedding-based ReID models (e.g., ResNet, ArcFace)
- Add **jersey number detection** using OCR for more robust ID assignment
- Enhance occlusion handling with **Kalman Filters** or advanced motion models

---

## 📝 Conclusion

A simple yet effective solution was implemented, demonstrating the core principles of re-identification and tracking in sports analytics.  
All results are reproducible, and the workflow is fully documented for further