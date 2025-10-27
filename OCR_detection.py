import easyocr
import cv2
from collections import Counter
import os
import shutil


test = [1,1,1,2,2,3,3,3,3]

plate = Counter(test).most_common(1)[0][0] #gets majority vote
print(plate)



dir_name = 'test'

# Step 1: Check if the directory exists.
if os.path.exists(dir_name):
    # If it exists, remove the directory and everything inside it.
    print(f"Directory '{dir_name}' already exists. Removing it.")
    shutil.rmtree(dir_name)

# Step 2: Create a new, empty directory.
os.mkdir(dir_name)
print(f"Successfully created clean directory: '{dir_name}'")

# Verify it's there
print(os.listdir())