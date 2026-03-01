# -------------------------------------------------
# PROJECT: Medical Imaging Boundary Detection using Convex Hull
# Developed by: [Your Name]
# -------------------------------------------------

import cv2
import numpy as np
import os
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# ----------- STEP 1: SET FOLDER PATH -----------
folder_path = "img"  # Folder containing your medical images

# ----------- STEP 2: CREATE OUTPUT FOLDER -----------
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

# ----------- STEP 3: FUNCTION TO PROCESS IMAGE -----------
def process_image(image_path):
    print(f"Processing: {image_path}")

    # Read the image (grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image:", image_path)
        return

    # STEP 1: Denoise using Gaussian Blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # STEP 2: Threshold using Otsu’s method
    thresh_val = threshold_otsu(blur)
    _, binary = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    # STEP 3: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # STEP 4: Create color version of original image for drawing
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # STEP 5: Apply convex hull to each contour
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # ignore tiny regions
            hull = cv2.convexHull(cnt)
            cv2.drawContours(img_color, [hull], -1, (0, 255, 0), 2)

    # STEP 6: Save and display the results
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_folder, f"hull_{filename}")
    cv2.imwrite(save_path, img_color)

    # Show before and after using matplotlib
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Convex Hull Boundary")
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"✅ Saved result to: {save_path}\n")


# ----------- STEP 4: LOOP THROUGH ALL IMAGES -----------
for file in os.listdir(folder_path):
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        image_path = os.path.join(folder_path, file)
        process_image(image_path)

print("🎉 All images processed successfully!")
