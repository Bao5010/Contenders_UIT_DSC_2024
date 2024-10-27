import json

# Load the data from both JSON files
with open('statistics/results_image.json', 'r', encoding='utf-8') as f:
    image_data = json.load(f)

with open('statistics/best_results_multi.json', 'r', encoding='utf-8') as f:
    multi_data = json.load(f)

# Copy the data labeled as 'image-sarcasm' from image_data to multi_data
for key, value in image_data['results'].items():
    if value == 'image-sarcasm' and multi_data['results'][key] != 'multi-sarcasm':
        multi_data['results'][key] = value

# Save the updated multi_data back to the JSON file
with open('statistics/results_ovr.json', 'w', encoding='utf-8') as f:
    json.dump(multi_data, f, ensure_ascii=False, indent=4)

print("Data labeled as 'image-sarcasm' has been copied successfully.")