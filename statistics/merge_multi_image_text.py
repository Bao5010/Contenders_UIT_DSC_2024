import json

# Load the data from both JSON files
with open('image_results_recall_0.5.json', 'r', encoding='utf-8') as f:
    image_data = json.load(f)

with open('best_results_multi_not.json', 'r', encoding='utf-8') as f:
    multi_data = json.load(f)

with open('results_text_not.json', 'r', encoding='utf-8') as f:
    text_data = json.load(f)

for key,value in image_data['results'].items():
    if value == 'image-sarcasm':
        if text_data['results'][key] == 'text-sarcasm':
            multi_data['results'][key] = 'multi-sarcasm'
        elif multi_data['results'][key] != 'multi-sarcasm':
            text_data['results'][key] = 'image-sarcasm'

for key,value in text_data['results'].items():
    if value == 'text-sarcasm':
        if multi_data['results'][key] != 'multi-sarcasm':
            multi_data['results'][key] = 'text-sarcasm'

# Save the updated preprocess_data back to the JSON file
with open('results_preprocess_datas.json', 'w', encoding='utf-8') as f:
    json.dump(multi_data, f, ensure_ascii=False, indent=4)

print("Data labeled as 'image-sarcasm' has been copied successfully.")