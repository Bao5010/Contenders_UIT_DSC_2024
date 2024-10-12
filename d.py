import torch
from transformers import Qwen2VLForConditionalGeneration, DefaultDataCollator, AutoProcessor, TrainingArguments, Trainer
import json
from PIL import Image
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle

# Load the dataset
with open('a.json', 'r', encoding='utf-8') as f:
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
        image_path = 'warmup-images\\' + value['image']  # Ensure correct path handling
        caption = value['caption']
        label = label_mapping[value['label']]  # Convert label to numerical value
    
        # Load the image
        image = Image.open(image_path).convert("RGB")
        print(f"Original image size: {image.size}")  # Print original image size

        # Append the data
        images.append(image)
        captions.append(caption)
        labels.append(label)

    # Preprocess the images and captions
    # Use the processor for image-text modality, apply padding and truncation to handle different input lengths
    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True)
     # Convert input_ids to LongTensor
    inputs['input_ids'] = inputs['input_ids'].long()

    # Pad the labels to match the length of the input_ids
    max_length = inputs['input_ids'].shape[1]
    padded_labels = pad_sequence([torch.tensor([label] * max_length) for label in labels], batch_first=True)

    # Pad the pixel values to match the length of the input_ids
    padded_pixel_values = pad_sequence([torch.tensor(pixel_value) for pixel_value in inputs['pixel_values']], batch_first=True)

    # Check the shapes before returning
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"attention_mask shape: {inputs['attention_mask'].shape}")
    print(f"pixel_values shape: {padded_pixel_values.shape}")
    print(f"labels shape: {padded_labels.shape}")

    print(f"input_ids: {inputs['input_ids']}")
    print(f"attention_mask: {inputs['attention_mask']}")
    print(f"pixel_values: {padded_pixel_values}")
    print(f"labels: {padded_labels}")

    # Return the dictionary with all processed data
    return {
        'input_ids': inputs['input_ids'],  # shape: [batch_size, seq_length]
        'attention_mask': inputs['attention_mask'],  # shape: [batch_size, seq_length]
        'pixel_values': padded_pixel_values,  # shape should be [batch_size, channels, height, width]
        'labels': padded_labels  # Ensure labels match the number of examples
    }


# Example usage
preprocessed_data = preprocess_data(data)

# Convert to Dataset object (Huggingface Dataset)
preprocessed_dataset = Dataset.from_dict({
    'input_ids': preprocessed_data['input_ids'],
    'attention_mask': preprocessed_data['attention_mask'],
    'labels': preprocessed_data['labels']
})
data_collator = DefaultDataCollator()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

repo_id = "my_model"  # Update to the correct repo ID

training_args = TrainingArguments(
    output_dir=repo_id,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    per_device_train_batch_size=1,  
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=preprocessed_dataset,
    tokenizer=processor
)

trainer.train()  # Start training the model
# save the model
trainer.save_model(repo_id)