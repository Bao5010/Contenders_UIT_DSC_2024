import json
import random

# Step 1: Read the JSON files
with open('training/split_multi_sarcasm.json', 'r', encoding='utf-8') as file:
    sarcasm_data = json.load(file)

with open('training/split_not_sarcasm.json', 'r', encoding='utf-8') as file:
    not_sarcasm_data = json.load(file)

# Step 2: Merge the data
merged_data = list(sarcasm_data.values()) + list(not_sarcasm_data.values())

# Step 3: Shuffle the data
random.shuffle(merged_data)

# Step 4: Reindex the keys
reindexed_data = {str(index): item for index, item in enumerate(merged_data)}

# Step 5: Write to a new file
with open('training/multi-classification/split_multi_not_sarcasm.json', 'w', encoding='utf-8') as file:
    json.dump(reindexed_data, file, ensure_ascii=False, indent=4)