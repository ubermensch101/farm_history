import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Function to convert an image to HSV and return the HSV image
def convert_to_hsv(image_path):
    # Read the image using OpenCV
    image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_cv is None:
        print("Error: Could not open or find the image.")
        return None
    
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    return hsv_image

# Define the image paths
image_path1 = "monthly/hunsa/2022/1/response.tiff"
image_path2 = "quads/hunsa/global_monthly_2022_06_mosaic/1466-1132_quad.tif"

# Convert images to HSV
hsv_image1 = convert_to_hsv(image_path1)
hsv_image2 = convert_to_hsv(image_path2)

# Convert original images to BGR for enhancement and histogram matching
image_cv1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
image_cv2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

# Enhancement techniques
# Method 1: Brightness adjustment by adding a matrix
brightness_matrix = np.ones(image_cv1.shape, dtype="uint8") * 50
brightened_image_cv = cv2.add(image_cv1, brightness_matrix)
cv2.imwrite("brightened_image_matrix.tiff", brightened_image_cv)

# Method 2: Brightness adjustment using PIL
image_pil = Image.open(image_path1)
enhancer = ImageEnhance.Brightness(image_pil)
image_pil_enhanced = enhancer.enhance(1.5)
image_pil_enhanced.save("brightened_image_pil.tiff")

# Convert the PIL image to OpenCV format for displaying
image_pil_enhanced_cv = cv2.cvtColor(np.array(image_pil_enhanced), cv2.COLOR_RGB2BGR)

# Method 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
lab_image = cv2.cvtColor(image_cv1, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_image)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
enhanced_image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imwrite("enhanced_image_clahe.tiff", enhanced_image_clahe)

# Method 4: Contrast Enhancement using PIL
enhancer_contrast = ImageEnhance.Contrast(image_pil)
image_pil_contrast = enhancer_contrast.enhance(1.5)
image_pil_contrast.save("contrast_image_pil.tiff")

# Convert the PIL contrast-enhanced image to OpenCV format for displaying
image_pil_contrast_cv = cv2.cvtColor(np.array(image_pil_contrast), cv2.COLOR_RGB2BGR)

# Method 5: Auto Gamma Correction
def automatic_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma_corrected_image_cv = automatic_gamma_correction(image_cv1, gamma=1.5)
cv2.imwrite("gamma_corrected_image_cv.tiff", gamma_corrected_image_cv)

# Histogram Matching
# Check if images are multi-channel (color) or single-channel (grayscale)
if len(image_cv1.shape) == 3 and len(image_cv2.shape) == 3:
    matched_image = np.zeros_like(image_cv1)
    for i in range(image_cv1.shape[2]):
        matched_image[:, :, i] = match_histograms(image_cv1[:, :, i], image_cv2[:, :, i])
else:
    matched_image = match_histograms(image_cv1, image_cv2)

cv2.imwrite("matched_image.tiff", matched_image)

# Display the images using Matplotlib
images = [
    ("Original Image 1", image_cv1),
    ("Original Image 2", image_cv2),
    ("Brightened Image (Matrix Addition)", brightened_image_cv),
    ("Brightened Image (PIL Enhancement)", image_pil_enhanced_cv),
    ("Enhanced Image (CLAHE)", enhanced_image_clahe),
    ("Contrast Enhanced Image (PIL)", image_pil_contrast_cv),
    ("Gamma Corrected Image", gamma_corrected_image_cv),
    ("Histogram Matched Image", matched_image)
]

plt.figure(figsize=(18, 12))

for i, (title, img) in enumerate(images, 1):
    plt.subplot(2, 4, i)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
