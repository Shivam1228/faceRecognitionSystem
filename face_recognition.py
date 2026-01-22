from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(face_img):
    face = cv2.resize(face_img, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        embedding = model(face)
    return embedding.numpy()[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
