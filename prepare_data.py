import os
import shutil

src_img = "COD10K-v3/Train/Image"
src_mask = "COD10K-v3/Train/GT_Object"

dst_img = "dataset/images"
dst_mask = "dataset/masks"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_mask, exist_ok=True)

for file in os.listdir(src_img):
    shutil.copy(os.path.join(src_img, file), dst_img)

for file in os.listdir(src_mask):
    shutil.copy(os.path.join(src_mask, file), dst_mask)

print("Dataset prepared!")