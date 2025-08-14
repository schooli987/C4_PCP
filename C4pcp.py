import cv2
import random
import time
import os

# -----------------------------
# Config
# -----------------------------
adjectives = ["Intelligent", "Fantastic", "Awesome", "Brilliant", "Amazing"]

change_interval = 0.1  # seconds between adjective changes


# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

start_time = time.time()
last_change = time.time()
current_adjective = random.choice(adjectives)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)

    # Change adjective every few seconds
    if time.time() - last_change > change_interval:
        current_adjective = random.choice(adjectives)
        last_change = time.time()

    # Draw rectangle and adjective
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, current_adjective, (x+10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Detection with Adjectives", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save snapshot when 's' is pressed
        filename =  f"{random.randint(1000, 9000)}.png"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

  
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
