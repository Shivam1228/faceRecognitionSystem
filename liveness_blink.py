import dlib
import cv2
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def is_blinking(gray):
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = coords[36:42]
        right_eye = coords[42:48]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < 0.21:
            return True
    return False
