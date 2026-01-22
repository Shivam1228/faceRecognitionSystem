import torch
import cv2
import numpy as np

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def estimate_depth(face_img):
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_batch)

    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    depth_variance = np.var(depth_map)
    return depth_map, depth_variance
