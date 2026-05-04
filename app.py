import streamlit as st
import numpy as np
import cv2
from PIL import Image
from predict import predict

st.title("Camouflaged Object Detection")

file = st.file_uploader("Upload Image", type=["jpg", "png"])

if file:
    image = np.array(Image.open(file))
    st.image(image, caption="Original Image")

    mask = predict(image)

    # Show raw mask
    st.image(mask, caption="Predicted Mask")

    # 🔥 FIX: Resize mask to match original image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert to 3 channel
    mask3 = np.stack([mask_resized]*3, axis=-1)

    # Overlay
    overlay = image.copy().astype(float)
    overlay = overlay * (1 - mask3)
    overlay = overlay.astype(np.uint8)

    st.image(overlay, caption="Detected Camouflage")




# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from predict import predict

# st.title("Camouflaged Object Detection")

# file = st.file_uploader("Upload Image", type=["jpg", "png"])

# if file:
#     image = np.array(Image.open(file))
#     st.image(image, caption="Original Image")

#     mask = predict(image)

#     # 🔥 Normalize mask (important)
#     mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

#     # 🔥 Apply threshold (this makes it visible)
#     threshold = 0.3
#     binary_mask = (mask > threshold).astype(np.uint8)

#     # Show binary mask
#     st.image(binary_mask * 255, caption="Binary Mask")

#     # Resize to original size
#     mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

#     # Convert to 3-channel
#     mask3 = np.stack([mask_resized]*3, axis=-1)

#     # 🔥 Better overlay (RED highlight instead of black)
#     overlay = image.copy()
#     overlay[mask_resized == 1] = [255, 0, 0]  # red highlight

#     st.image(overlay, caption="Detected Camouflage")