from mtcnn import MTCNN
import cv2

detector = MTCNN()

def detect_faces(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    boxes = []
    for face in faces:
        x, y, w, h = face['box']
        boxes.append((x, y, w, h))
    return boxes
