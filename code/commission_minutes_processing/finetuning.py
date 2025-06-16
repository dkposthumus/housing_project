from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch

# Step 1: Load your hand-labelled JSONL training file
# Format: each line is {"prompt": "...raw block...", "completion": "...target JSON..."}
train_dataset = load_dataset('json', data_files='data/structured_minutes.jsonl', split='train')

# Step 2: Tokenizer and Preprocessing
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_input_length = 512
max_target_length = 256

def preprocess_fn(batch):
    inputs = ["extract project fields:\n" + x for x in batch["prompt"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["completion"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_data = train_dataset.map(preprocess_fn, batched=True, remove_columns=["prompt", "completion"])

# Step 3: Load base model and apply LoRA
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(base_model, peft_config)

# Step 4: Training setup
training_args = Seq2SeqTrainingArguments(
    output_dir="out/minutes_extractor",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    save_steps=500,
    fp16=True,
    logging_steps=100,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer
)

# Step 5: Train
trainer.train()
model.save_pretrained("out/minutes_extractor")
tokenizer.save_pretrained("out/minutes_extractor")

# Step 6: Inference on unseen example
inference_model = AutoModelForSeq2SeqLM.from_pretrained("out/minutes_extractor", device_map="auto", load_in_8bit=True)
inference_tokenizer = AutoTokenizer.from_pretrained("out/minutes_extractor")

# Replace with path to an unseen .txt file (e.g., minutes_raw/1999/19990610.txt)
unseen_text = Path("data/meeting_minutes/raw/1999/19990604.txt").read_text(encoding="utf-8", errors="ignore")
prompt = "extract project fields:\n" + unseen_text

inputs = inference_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(inference_model.device)
outputs = inference_model.generate(**inputs, max_new_tokens=max_target_length)
result = inference_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Extracted JSON:", result)
