import json

# Load the image_sarcasm data
with open('training/image_sarcasm.json', 'r', encoding='utf-8') as file:
    image_sarcasm_data = json.load(file)

# Load the multi_sarcasm data
with open('training/multi_sarcasm.json', 'r', encoding='utf-8') as file:
    multi_sarcasm_data = json.load(file)

# Get the first 100 entries from multi_sarcasm and rename the label
multi_sarcasm_subset = {k: v for k, v in list(multi_sarcasm_data.items())[:300]}
for item in multi_sarcasm_subset.values():
    item['label'] = 'image-sarcasm'

# Append the multi_sarcasm_subset to image_sarcasm_data
image_sarcasm_data.update(multi_sarcasm_subset)

# Save the updated image_sarcasm data back to the file
with open('training/image_sarcasm_add_multi_300.json', 'w', encoding='utf-8') as file:
    json.dump(image_sarcasm_data, file, ensure_ascii=False, indent=4)

print("Data appended successfully.")