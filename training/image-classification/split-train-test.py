import json
import random

# Load the JSON data from the file
with open('training/image-classification/vimmsd-train-image.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Collect items with the specified labels
image_sarcasm_items = []
not_sarcasm_items = []

for key, value in data.items():
    if value['label'] == 'image-sarcasm':
        image_sarcasm_items.append({key: value})
    elif value['label'] == 'not-sarcasm':
        not_sarcasm_items.append({key: value})

# Select 28 items labeled "image-sarcasm" and 442 items labeled "not-sarcasm"
selected_image_sarcasm = random.sample(image_sarcasm_items, 28)
selected_not_sarcasm = random.sample(not_sarcasm_items, 442)

# Combine the selected items
selected_items = {**{k: v for item in selected_image_sarcasm for k, v in item.items()},
                  **{k: v for item in selected_not_sarcasm for k, v in item.items()}}

# Remove selected items from the original data
for key in selected_items.keys():
    del data[key]

# Reset keys in the remaining training data to start from 0 incrementally
updated_train_data = {str(index): value for index, (key, value) in enumerate(data.items())}

# Save the updated training data back to the file
with open('training/image-classification/vimmsd-train-image.json', 'w', encoding='utf-8') as file:
    json.dump(updated_train_data, file, ensure_ascii=False, indent=4)

# Shuffle the selected items
items = list(selected_items.items())
random.shuffle(items)

# Rearrange keys starting from 0 incrementally for the test data
shuffled_data = {str(index): value for index, (key, value) in enumerate(items)}

# Save the selected items into a new JSON file
with open('training/image-classification/vimmsd-test-image.json', 'w', encoding='utf-8') as file:
    json.dump(shuffled_data, file, ensure_ascii=False, indent=4)

with open('training/image-classification/vimmsd-test-image.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create the results dictionary
results = {key: value['label'] for key, value in data.items()}

# Create the output dictionary
output = {
    "results": results,
    "phase": "dev"
}

# Write the output JSON file
with open('training/image-classification/vimmsd-test-image-results.json', 'w', encoding='utf-8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)