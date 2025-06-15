import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import time
import mss
import numpy as np

# === Load Model ===
body_model = YOLO("yolov8n.pt")  # yolov8n.pt for faster but less accurate

# === Create output folder ===
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# === Screen capture setup ===
sct = mss.mss()
monitor = sct.monitors[1]  # Full screen capture (primary monitor)

print("[INFO] Starting screen detection. Press 'q' to quit.")

last_saved_time = 0
save_interval = 5  # seconds

while True:
    # === Capture the screen as frame ===
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    frame_resized = cv2.resize(frame, (1000, 780))

    # === Run YOLOv8 for person detection ===
    results = body_model(frame_resized, verbose=False)[0]
    body_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = body_model.names[cls]
        if label == "person":
            body_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    current_time = time.time()

    # === Save snapshot every 5 seconds if person detected ===
    if body_detected and (current_time - last_saved_time) >= save_interval:
        now = datetime.now()
        filename_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_path = os.path.join(output_folder, f"pic_{filename_time}.jpg")
        cv2.imwrite(snapshot_path, frame_resized)
        print(f"[SAVED] Human detected. Snapshot saved: {snapshot_path}")
        last_saved_time = current_time

    # === Display the video feed ===
    cv2.imshow("Screen Human Detection", frame_resized)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cv2.destroyAllWindows()
