import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.unet import UNet

class CamoDataset(Dataset):
    def __init__(self, img_dir, mask_dir, limit=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_files = os.listdir(img_dir)

        if limit:
            self.img_files = self.img_files[:limit]

        print("Total images:", len(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            return self.__getitem__((idx + 1) % len(self))

        img = cv2.resize(img, (256, 256)) / 255.0
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(float)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )

# Dataset
dataset = CamoDataset("dataset/images", "dataset/masks", limit=500)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Debug
for imgs, masks in loader:
    print("Batch:", imgs.shape)
    break

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()

# Training
for epoch in range(10):
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")