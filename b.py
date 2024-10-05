from transformers import Qwen2VLForConditionalGeneration, DefaultDataCollator, AutoProcessor, TrainingArguments, Trainer
import torch
import json
from PIL import Image
from datasets import Dataset

# Load the dataset
with open('vimmsd-warmup.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
        
# Initialize the processor with the correct model name
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Update to the correct model name
processor = AutoProcessor.from_pretrained(model_name)

# Create a mapping from label strings to numerical values
label_mapping = {
    "multi-sarcasm": 0,
    "not-sarcasm": 1,
    "text-sarcasm": 2,
    "image-sarcasm": 3,
}

def preprocess_data(dataset):
    images = []
    captions = []
    labels = []
    
    for key, value in dataset.items():
        image_path = 'warmup-images\\' + value['image']
        caption = value['caption']
        label = label_mapping[value['label']]  # Convert label to numerical value
        
        # Load the image
        image = Image.open(image_path)
        
        # Append the data
        images.append(image)
        captions.append(caption)
        labels.append(label)
    
    # Preprocess the images and captions
    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True)
    
    return {
        'input_ids': inputs['input_ids'],
        'labels': torch.tensor(labels)
    }

# Example usage
preprocessed_data = preprocess_data(data)

# Convert to Dataset object
preprocessed_dataset = Dataset.from_dict(preprocessed_data)

data_collator = DefaultDataCollator()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

repo_id = "my_model"  # Update to the correct model name

training_args = TrainingArguments(
    output_dir=repo_id,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=preprocessed_dataset,
    tokenizer=processor,
)

trainer.train()  # Training the model

# Evaluate
results = trainer.evaluate()
print(results)