import cv2
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
from PIL import Image

image_path = 'DoLR/merged256_new_nov.tif'  # Set your image path here

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to a CuPy array
image_cp = cp.asarray(image)

# Perform thresholding
_, image_cp = cv2.threshold(cp.asnumpy(image_cp), 127, 255, cv2.THRESH_BINARY)

# Convert back to CuPy array after thresholding
image_cp = cp.asarray(image_cp)

# Define the kernel
kernel = cp.ones((10, 10), cp.uint8)

# Create the negative of the image
negative_cp = 255 - image_cp

# Perform dilation
dilated_image_cp = cv2.dilate(cp.asnumpy(negative_cp), cp.asnumpy(kernel), iterations=1)

# Convert back to CuPy array after dilation
dilated_image_cp = cp.asarray(dilated_image_cp)

# Convert the dilated image to a PIL image
broad = Image.fromarray(cp.asnumpy(dilated_image_cp))

# Save the result
split_list=image_path.split('.')
name=split_list[0]
save_path=name+'_thick.tif'
broad.save(save_path)
