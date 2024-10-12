import torch
from transformers import Qwen2VLForConditionalGeneration, DefaultDataCollator, AutoProcessor, TrainingArguments, Trainer
import json
from PIL import Image
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import gc
import os

# # Enable memory efficient attention if available
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Load the dataset
# with open('a.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# with open('a.json', 'r') as file:
#     data = json.load(file)

# # Convert the JSON object with numeric keys into a list of objects
# data_list = [value for key, value in sorted(data.items(), key=lambda item: int(item[0]))]
# dataset = Dataset.from_list(data_list)

# # Access the first entry
# print(dataset)


food = load_dataset("food101", split="train[:10]")
print(food)