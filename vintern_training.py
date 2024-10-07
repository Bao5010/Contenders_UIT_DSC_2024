from transformers import AutoModel, AutoTokenizer, DefaultDataCollator, TrainingArguments, Trainer
import torch
import json
from PIL import Image
from datasets import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Load the dataset
with open('vimmsd-warmup.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# Initialize the model and tokenizer
model_name = "5CD-AI/Vintern-1B-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

def build_transform(input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Use the first ratio for simplicity in training
    target_ratio = next(iter(target_ratios))
    
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

# Create a mapping from label strings to numerical values
label_mapping = {
    "multi-sarcasm": 0,
    "not-sarcasm": 1,
    "text-sarcasm": 2,
    "image-sarcasm": 3,
}

def preprocess_data(dataset):
    transform = build_transform()
    pixel_values_list = []
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for key, value in dataset.items():
        image_path = 'warmup-images\\' + value['image']
        caption = value['caption']
        label = label_mapping[value['label']]
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        processed_images = dynamic_preprocess(image)
        pixel_values = torch.stack([transform(img) for img in processed_images])
        
        # Process text
        encoded_text = tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        pixel_values_list.append(pixel_values)
        input_ids_list.append(encoded_text['input_ids'].squeeze())
        attention_mask_list.append(encoded_text['attention_mask'].squeeze())
        labels_list.append(label)
    
    return {
        'pixel_values': pixel_values_list,
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': torch.tensor(labels_list)
    }

# Preprocess the data
preprocessed_data = preprocess_data(data)

# Convert to Dataset object
preprocessed_dataset = Dataset.from_dict(preprocessed_data)

data_collator = DefaultDataCollator()

# Initialize the model
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)

repo_id = "vintern_sarcasm_detector"

training_args = TrainingArguments(
    output_dir=repo_id,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=preprocessed_dataset,
    tokenizer=tokenizer,
)

trainer.train()  # Training the model

# Evaluate
results = trainer.evaluate()
print(results)