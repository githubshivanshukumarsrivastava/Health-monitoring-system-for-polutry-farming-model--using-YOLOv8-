import os
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import time
import platform
import shutil
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------
MODEL_PATH = r"H:\Requin\10 sep\detect\train4\weights\best.pt"
LOG_FILE = "mortality1_log.csv"
DETECTIONS_FILE = "detections1.csv"
SAVED_FRAMES_DIR = "saved_frames1"
CONF_THRESHOLD = 0.2
CAMERA_ID = 0
IMG_SIZE = 640

# -----------------------
# Reset logs on startup
# -----------------------
for file in [LOG_FILE, DETECTIONS_FILE]:
    if os.path.exists(file):
        os.remove(file)
if os.path.exists(SAVED_FRAMES_DIR):
    shutil.rmtree(SAVED_FRAMES_DIR)
os.makedirs(SAVED_FRAMES_DIR, exist_ok=True)

# -----------------------
# Load YOLO model
# -----------------------
print("✅ Loading YOLO model...")
model = YOLO(MODEL_PATH)

# -----------------------
# Initialize logs
# -----------------------
df_log = pd.DataFrame(columns=["timestamp", "death_count", "healthy_count"])
df_det = pd.DataFrame(columns=["timestamp", "frame_id", "label", "x1", "y1", "x2", "y2", "image_file"])

# -----------------------
# Open camera
# -----------------------
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

frame_id = 0

# -----------------------
# Matplotlib plot setup
# -----------------------
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("Cumulative Chicken Mortality")
ax.set_xlabel("Time")
ax.set_ylabel("Count")
line_death, = ax.plot([], [], 'r-o', label="Deaths")
line_healthy, = ax.plot([], [], 'g-x', label="Healthy")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# -----------------------
# Sound alert
# -----------------------
def sound_alert():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)

# -----------------------
# Monitoring flag
# -----------------------
monitoring = False

# -----------------------
# Update matplotlib plot inside Tkinter
# -----------------------
def update_plot(df):
    if not df.empty:
        times = pd.to_datetime(df["timestamp"])
        line_death.set_data(times, df["death_count"].cumsum())
        line_healthy.set_data(times, df["healthy_count"].cumsum())
        ax.relim()
        ax.autoscale_view()
        canvas.draw()

# -----------------------
# Monitoring function using Tkinter after()
# -----------------------
def monitor_frame():
    global frame_id, df_log, df_det
    if not monitoring:
        return

    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        root.after(50, monitor_frame)
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    death_count = 0
    healthy_count = 0
    detection_rows = []

    results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False, stream=False)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        labels = [model.names[int(cls)] for cls in r.boxes.cls]

        for (x1, y1, x2, y2), label in zip(boxes, labels):
            if label == "death_chicken":
                death_count += 1
                color = (0, 0, 255)
            else:
                healthy_count += 1
                color = (0, 255, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            fname = os.path.join(SAVED_FRAMES_DIR, f"frame{frame_id}_{label}.jpg")
            cv2.imwrite(fname, crop)

            detection_rows.append({
                "timestamp": timestamp,
                "frame_id": frame_id,
                "label": label,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "image_file": fname
            })

    if detection_rows:
        df_det = pd.concat([df_det, pd.DataFrame(detection_rows)], ignore_index=True)
        df_det.to_csv(DETECTIONS_FILE, index=False)

        new_log_row = pd.DataFrame([{
            "timestamp": timestamp,
            "death_count": death_count,
            "healthy_count": healthy_count
        }])
        df_log = pd.concat([df_log, new_log_row], ignore_index=True)
        df_log.to_csv(LOG_FILE, index=False)

        # -------------------
        # GUI Updates
        # -------------------
        death_count_label.config(text=f"Deaths: {death_count}")
        healthy_count_label.config(text=f"Healthy: {healthy_count}")

        print(f"[{timestamp}] Frame {frame_id} | Deaths: {death_count} | Healthy: {healthy_count}")
        if death_count > 0:
            print(f"⚠️ ALERT! {death_count} dead chicken(s) detected!")
            sound_alert()

        update_plot(df_log)

    cv2.imshow("Mortality Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_monitoring()
        return

    root.after(50, monitor_frame)

# -----------------------
# Start/Stop functions
# -----------------------
def start_monitoring():
    global monitoring
    if monitoring:
        return
    monitoring = True
    status_label.config(text="Status: Running", foreground="green")
    monitor_frame()

def stop_monitoring():
    global monitoring
    monitoring = False
    status_label.config(text="Status: Stopped", foreground="red")
    print("🛑 Monitoring stopped")

# -----------------------
# Tkinter GUI
# -----------------------
root = tk.Tk()
root.title("🐔 Chicken Mortality Monitoring 🐔")
root.geometry("900x600")
root.configure(bg="#f0f0f0")  # Light background color

# Frame for buttons and status
control_frame = tk.Frame(root, bg="#20a0f5", bd=2, relief=tk.RIDGE, padx=20, pady=20)
control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Start button
start_btn = ttk.Button(control_frame, text="▶ Start Monitoring", command=start_monitoring)
start_btn.pack(side=tk.LEFT, padx=10, pady=5)

# Stop button
stop_btn = ttk.Button(control_frame, text="⏹ Stop Monitoring", command=stop_monitoring)
stop_btn.pack(side=tk.LEFT, padx=10, pady=5)

# Status label
status_label = ttk.Label(control_frame, text="Status: Stopped", foreground="red",
                         font=("Helvetica", 12, "bold"))
status_label.pack(side=tk.LEFT, padx=20)

# Current frame counts
death_count_label = ttk.Label(control_frame, text="Deaths: 0", foreground="red",
                              font=("Helvetica", 12, "bold"))
death_count_label.pack(side=tk.LEFT, padx=10)

healthy_count_label = ttk.Label(control_frame, text="Healthy: 0", foreground="green",
                                font=("Helvetica", 12, "bold"))
healthy_count_label.pack(side=tk.LEFT, padx=10)

# Embed matplotlib plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# -----------------------
# Graceful exit
# -----------------------
def on_closing():
    stop_monitoring()
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Data saved to {LOG_FILE}")
    print(f"✅ Detection file saved to {DETECTIONS_FILE}")
    print(f"✅ Cropped frames saved in {SAVED_FRAMES_DIR}")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
