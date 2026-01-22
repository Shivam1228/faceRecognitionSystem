import cv2
import numpy as np

from face_detection import detect_faces
from face_recognition import get_embedding, cosine_similarity
from liveness_blink import is_blinking
from depth_midas import estimate_depth

# Dummy registered face embedding
registered_embedding = None

cap = cv2.VideoCapture(0)

print("[INFO] Press 'r' to register face.")
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        embedding = get_embedding(face_img)

        if registered_embedding is not None:
            similarity = cosine_similarity(embedding, registered_embedding)
        else:
            similarity = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blink = is_blinking(gray)

        depth_map, depth_var = estimate_depth(face_img)

        # Decision logic
        status = "SPOOF"
        if similarity > 0.6 and blink and depth_var > 0.002:
            status = "LIVE"

        # Draw UI
        color = (0, 255, 0) if status == "LIVE" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{status}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AI Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and faces:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        registered_embedding = get_embedding(face_img)
        print("[INFO] Face registered.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
