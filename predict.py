import torch
import cv2
import numpy as np
from model.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

def predict(image):
    img = cv2.resize(image, (256, 256)) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)[0][0].cpu().numpy()

    return pred