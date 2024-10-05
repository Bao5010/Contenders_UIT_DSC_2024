from transformers import Qwen2VLForConditionalGeneration, DefaultDataCollator, AutoProcessor, TrainingArguments, Trainer
import torch
import json
from PIL import Image

# Load the dataset
with open('vimmsd-warmup.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
        
# Initialize the processor with the correct model name
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Update to the correct model name
processor = AutoProcessor.from_pretrained(model_name)

# Create a mapping from label strings to numerical values
label_mapping = {
    "multi-sarcasm": 0,
    # Add other labels if necessary
}

def preprocess_data(dataset):
    preprocessed_data = []
    for key, value in dataset.items():
        image_path = 'D:\\UIT_DSC\\warmup-images\\' + value['image']
        caption = value['caption']
        label = label_mapping[value['label']]  # Convert label to numerical value
        
        # Load the image
        image = Image.open(image_path)
        
        # Preprocess the image and caption
        inputs = processor(images=image, text=caption, return_tensors="pt")
        
        # Append the preprocessed data
        preprocessed_data.append({
            'inputs': inputs,
            'label': label
        })
    
    return preprocessed_data

# Example usage
preprocessed_dataset = preprocess_data(data)

data_collator = DefaultDataCollator()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

repo_id = "Qwen/Qwen2-VL-2B-Instruct"  # Update to the correct model name

training_args = TrainingArguments(
    output_dir=repo_id,
    per_device_train_batch_size=4,
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