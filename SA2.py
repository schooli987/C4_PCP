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

    cv2.imshow("Fun Filters", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
