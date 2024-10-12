import torch
from transformers import Qwen2VLForConditionalGeneration, DefaultDataCollator, AutoProcessor, TrainingArguments, Trainer, AutoImageProcessor
import json
from PIL import Image
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

# Load the dataset

# Initialize the processor with the correct model name
model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Update to the correct model name
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Create a mapping from label strings to numerical values

size = (224, 224)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

data = load_dataset("json", data_files="a.json")
# Access the specific split
train_data = data['train']


def transforms(examples):
    print(examples["image"])
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples
train_data = train_data.with_transform(transforms)

labels = {key: value['label'] for key, value in train_data.features.items()}

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

data_collator = DefaultDataCollator()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map="auto", 
    num_labels = len(labels),
    id2label=id2label,
    label2id=label2id
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
    train_dataset=train_data,
    tokenizer=image_processor
)

trainer.train()  # Start training the model