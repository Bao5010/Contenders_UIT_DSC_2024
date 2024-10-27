import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load the dataset
with open('training/text-classification/text_not_sarcasm_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare the data
texts = [entry['caption'] for entry in data.values()]
labels = [0 if entry['label'] == 'not-sarcasm' else 1 for entry in data.values()]

# Create a Dataset object
dataset = Dataset.from_dict({'text': texts, 'label': labels})

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
model = AutoModelForSequenceClassification.from_pretrained('vinai/phobert-base-v2', num_labels=2)
print(model)


# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./text-sarc',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    logging_steps=100,
    learning_rate=5e-5,
    remove_unused_columns=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()
# Save the model
model.save_pretrained('./text-sarcasm-model')
tokenizer.save_pretrained('./text-sarcasm-model')