import json

# Load the data from both JSON files
with open('scoring/dev.json', 'r', encoding='utf-8') as f:
    image_data = json.load(f)

with open('statistics/test_scoring_system.json', 'r', encoding='utf-8') as f:
    multi_data = json.load(f)

# Copy the data labeled as 'image-sarcasm' from image_data to multi_data
for key, value in image_data['results'].items():
    if value == 'multi-sarcasm':
        multi_data['results'][key] = value

# Save the updated multi_data back to the JSON file
with open('statistics/test_scoring_system.json', 'w', encoding='utf-8') as f:
    json.dump(multi_data, f, ensure_ascii=False, indent=4)

print("Data labeled as 'image-sarcasm' has been copied successfully.")