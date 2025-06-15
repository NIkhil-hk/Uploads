import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import mss
import numpy as np

# Load YOLO model
body_model = YOLO("yolov8n.pt")

# Create output folder
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# Define the LEFT HALF of screen to capture (video playing area)
monitor = {"top": 0, "left": 0, "width": 960, "height": 1080}
sct = mss.mss()

print("[INFO] Detection started. Press 'q' to quit.")
last_saved_time = 0
save_interval = 3  # seconds

# Set up OpenCV window on RIGHT HALF
cv2.namedWindow("Human Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Human Detection", 960, 1080)
cv2.moveWindow("Human Detection", 960, 0)

while True:
    # Capture left half of the screen (video area)
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    frame_resized = cv2.resize(frame, (640, 480))

    # Run YOLO on the captured frame
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

    # Save snapshot if person detected
    current_time = time.time()
    if body_detected and (current_time - last_saved_time) >= save_interval:
        now = datetime.now()
        filename_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_path = os.path.join(output_folder, f"pic_{filename_time}.jpg")
        cv2.imwrite(snapshot_path, frame_resized)
        print(f"[SAVED] Snapshot saved: {snapshot_path}")
        last_saved_time = current_time

    # Show detection results on right half
    cv2.imshow("Human Detection", frame_resized)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cv2.destroyAllWindows()
