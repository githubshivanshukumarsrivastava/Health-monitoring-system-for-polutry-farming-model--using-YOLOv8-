from ultralytics import YOLO

# apne model ka path do
model = YOLO("H:/Requin/19 sep/detect/train4/weights/best.pt")

# ek dead chicken image ka path do
results = model.predict(r"H:\Requin\19 sep\new_chichken_dataset-1\train\images\dead-chicken-poultry-farm-selective-focus-31186371_jpg.rf.98b2e4fe66ecd14403f7e4c86f54e20e.jpg", conf=0.05, save=True, show=True)

# detections print karne ke liye
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: {model.names[cls]} with confidence {conf:.2f}")
