import json
import os
import shutil

# Load the JSON data
with open('training/image_sarcasm.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

target_dir = 'training/image-classification/analyzing/analyze-image/'
# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Iterate over the items in the JSON data
for key, value in data.items():
    image_name = value['image']
    source_path = os.path.join('train-images/', image_name)
    target_path = os.path.join(target_dir, f"{key}.jpg")
    
    # Copy the image to the new folder with the new name
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")
    else:
        print(f"Image {source_path} does not exist")

print("Image copying completed.")