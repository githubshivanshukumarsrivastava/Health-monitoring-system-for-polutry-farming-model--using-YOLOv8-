# fake_injector.py
import os
import cv2
import random
import pandas as pd
from datetime import datetime
import argparse
import time

# -----------------------
# Config (you can change)
# -----------------------
OUTPUT_LOG = "mortality_log.csv"
OUTPUT_DET = "detections.csv"
FRAMES_DIR = "saved_frames"
SAMPLE_IMAGE = None   # set to "sample.jpg" to use a static image instead of webcam
NUM_FRAMES = 200      # how many frames to simulate
DETECTION_PROB = 0.15 # probability per frame to create at least one detection (0.0-1.0)
DEATH_PROB = 0.3      # if detection occurs, probability that it's 'death_chicken'
MAX_DETECTIONS_PER_FRAME = 3
MIN_BOX_SIZE = 30     # min bbox width/height in pixels
MAX_BOX_SIZE_RATIO = 0.5  # bbox size relative to frame (max fraction)

# -----------------------
# Setup folders & CSVs
# -----------------------
os.makedirs(FRAMES_DIR, exist_ok=True)

# load or init CSVs
if os.path.exists(OUTPUT_LOG):
    df_log = pd.read_csv(OUTPUT_LOG)
else:
    df_log = pd.DataFrame(columns=["timestamp", "death_count", "healthy_count"])

if os.path.exists(OUTPUT_DET):
    df_det = pd.read_csv(OUTPUT_DET)
else:
    df_det = pd.DataFrame(columns=["timestamp", "frame_id", "label", "x1", "y1", "x2", "y2", "image_file"])

# -----------------------
# Video source
# -----------------------
if SAMPLE_IMAGE and os.path.exists(SAMPLE_IMAGE):
    use_image = True
    img = cv2.imread(SAMPLE_IMAGE)
    h, w = img.shape[:2]
else:
    use_image = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. If no camera available, set SAMPLE_IMAGE to a valid image path.")
        raise SystemExit
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        cap.release()
        raise SystemExit
    h, w = frame.shape[:2]

print(f"Frame size: {w}x{h}. OUTPUT_LOG={OUTPUT_LOG}, OUTPUT_DET={OUTPUT_DET}, FRAMES_DIR={FRAMES_DIR}")

# -----------------------
# Helper: random bbox generator
# -----------------------
def random_bbox(w, h):
    max_box_w = int(w * MAX_BOX_SIZE_RATIO)
    max_box_h = int(h * MAX_BOX_SIZE_RATIO)
    bw = random.randint(MIN_BOX_SIZE, max(MIN_BOX_SIZE, max_box_w))
    bh = random.randint(MIN_BOX_SIZE, max(MIN_BOX_SIZE, max_box_h))
    x1 = random.randint(0, max(0, w - bw))
    y1 = random.randint(0, max(0, h - bh))
    x2 = x1 + bw
    y2 = y1 + bh
    return x1, y1, x2, y2

# -----------------------
# Main loop: simulate frames
# -----------------------
frame_id = 0
try:
    for i in range(NUM_FRAMES):
        frame_id += 1
        if use_image:
            frame = img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed, stopping.")
                break

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        death_count = 0
        healthy_count = 0
        det_rows = []

        # Decide whether to generate detections in this frame
        if random.random() < DETECTION_PROB:
            num_dets = random.randint(1, MAX_DETECTIONS_PER_FRAME)
            for di in range(num_dets):
                x1, y1, x2, y2 = random_bbox(w, h)
                is_death = random.random() < DEATH_PROB
                label = "death_chicken" if is_death else "healthy_chicken"
                color = (0,0,255) if is_death else (0,255,0)

                # draw bbox on frame for visualization
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # crop & save detection image
                crop = frame[y1:y2, x1:x2]
                fname = os.path.join(FRAMES_DIR, f"frame{frame_id}_{label}_{i}.jpg")
                cv2.imwrite(fname, crop)

                # append detection row
                det_rows.append({
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "label": label,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "image_file": fname
                })

                if is_death:
                    death_count += 1
                else:
                    healthy_count += 1
        else:
            # no detection this frame
            pass

        # Append to detections CSV
        if det_rows:
            df_det = pd.concat([df_det, pd.DataFrame(det_rows)], ignore_index=True)
            df_det.to_csv(OUTPUT_DET, index=False)

        # Append to log CSV (one row per frame)
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "death_count": int(death_count),
            "healthy_count": int(healthy_count)
        }])
        df_log = pd.concat([df_log, new_row], ignore_index=True)
        df_log.to_csv(OUTPUT_LOG, index=False)

        # show progress in console
        print(f"[{timestamp}] Frame {frame_id} | Deaths: {death_count} | Healthy: {healthy_count} | Total detections: {len(det_rows)}")

        # small pause
        time.sleep(0.08)  # adjust to simulate fps

    print("Simulation finished.")

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if not use_image:
        cap.release()
    print("Final saved:", OUTPUT_LOG, OUTPUT_DET, "images in", FRAMES_DIR)
