import torch
from transformers import Qwen2VLForConditionalGeneration, DefaultDataCollator, AutoProcessor, TrainingArguments, Trainer, AutoImageProcessor
import json
from PIL import Image
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

# Initialize the processor with the correct model name
model_name = "Qwen/Qwen2-VL-2B-Instruct"
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Create image transforms
size = (224, 224)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

# Load and prepare dataset
with open('a.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
data_list = [value for key, value in sorted(data.items(), key=lambda item: int(item[0]))]
train_data = Dataset.from_list(data_list)
# train_data = train_data.train_test_split(test_size=0.1)

# Create label mappings
labels = train_data.unique("label")
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

def transforms(example):
    for i, img in enumerate(example["image"]):
        tmp = "warmup-images\\" + img
        example["image"][i] = Image.open(tmp)
    example["pixel_values"] = [_transforms(img.convert("RGB")) for img in example["image"]]
    # Convert label to a single integer (not a list)
    del example["image"]
    print(example)
    return example

train_data = train_data.with_transform(transforms)
print(train_data)

# Initialize model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map="auto", 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="my_model",
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
    data_collator=DefaultDataCollator(),
    train_dataset=train_data,
    tokenizer=image_processor
)

trainer.train()