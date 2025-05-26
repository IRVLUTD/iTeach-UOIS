#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------


import os
import re
import sys
import glob
import json
from PIL import Image
from tqdm import tqdm

# Base directory containing scene folders
base_dir =sys.argv[1]  # Change this to your actual path

# Iterate through all scene directories
for scene in tqdm(sorted(os.listdir(base_dir))):
    scene_path = os.path.join(base_dir, scene)
    rgb_dir = os.path.join(scene_path, "rgb")
    jpg_dir = os.path.join(scene_path, "jpg")
    print(rgb_dir)
    mapper_file = os.path.join(scene_path, "mapper.json")

    # Check if the RGB directory exists
    if not os.path.isdir(rgb_dir):
        continue

    # Create jpg directory if it doesn't exist
    os.makedirs(jpg_dir, exist_ok=True)


    # Collect all image files and sort in reverse order
    img_files = sorted(glob.glob(os.path.join(rgb_dir, "*")), reverse=True)


    # Dictionary to store mappings
    mapper = {}
# 
    print(img_files)

    # Process images
    for i, img_path in enumerate(img_files):
        img_name = os.path.basename(img_path)
        new_name = f"{i:06d}.jpg"
        new_path = os.path.join(jpg_dir, new_name)

        # Convert and save as JPG
        with Image.open(img_path) as img:
            img.convert("RGB").save(new_path, "JPEG")

        # Store mapping
        mapper[img_name] = new_name

    # Save mapping to a JSON file
    with open(mapper_file, "w") as f:
        json.dump(mapper, f, indent=4)

    print(f"Processed {scene} - {len(img_files)} images converted.")
