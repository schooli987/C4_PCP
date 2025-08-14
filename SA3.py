import cv2
import numpy as np
import time
import os

# -------------------------
# Load Face Detector
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Filter path and name
# -------------------------
filter_path = r"C:/Users/LENOVO/Desktop/AI/C4/sunglasses.png"
filter_name = "Sunglasses"

# Load filter
overlay_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
if overlay_img is None:
    print(f"Error: Could not load {filter_path}")
    exit()

# -------------------------
# Helper to overlay filter
# -------------------------
def add_filter(frame, overlay_img, x, y, w, h):
    overlay_img = cv2.resize(overlay_img, (w, h))
    b, g, r, a = cv2.split(overlay_img)
    mask = a / 255.0
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (1 - mask) * frame[y:y+h, x:x+w, c] + mask * overlay_img[:, :, c]
    return frame

# -------------------------
# Setup camera
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 7)

    for (x, y, w, h) in faces:
        if filter_name == "Sunglasses":
            
            fw, fh = w, int(h * 0.3)
            fx, fy = x, y + int(h * 0.3)

        frame = add_filter(frame, overlay_img, fx, fy, fw, fh)
        break

    cv2.putText(frame, filter_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Fun Filters", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
