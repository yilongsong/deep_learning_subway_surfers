

# preprocessdata.py

# yilong song
# Apr 6, 2023

import os
import cv2

# Directory path containing the image files
directory_path = "dataset"

# Loop through the directory
for folder in os.listdir('dataset'):
    if folder == '.DS_Store':
        continue
    for file in os.listdir('dataset/'+folder):
        if file.endswith(".png"):
            print(file)
            # Load the original image
            img_path = os.path.join(directory_path, folder, file)
            print(img_path)
            img = cv2.imread(img_path)

            # Apply Gaussian blur to the image
            downsampled_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

            # Define the new filename for the blurred image
            new_filename = "downsampled_" + file

            # Save the blurred image with the new filename
            new_img_path = os.path.join(directory_path, folder, new_filename)
            cv2.imwrite(new_img_path, downsampled_img)

            print(f"Processed {file} -> Saved {new_filename}")

print("Gaussian blur applied to all images in the directory.")