import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# ── Schema-driving prompt ───────────────────────────────────────────────
PROMPT_INSTRUCTION = """You are a strict JSON extractor. Given a block of raw meeting text, extract and return exactly one valid JSON object with the following keys (in any order). 

Only output the JSON — do not include explanations, formatting, or commentary. If a field is missing or not found, set it to an empty string ("").

Required keys:
- case_number              
- project_address          
- lot_number                
- assessor_block             
- project_descr              
- type_district             
- type_district_descr        
- speakers                   
- action                     
- modifications              
- ayes                       
- noes                       
- absent                     
- vote                       
- action_name  (should include 'resolution' or 'motion' if present)

Below is the raw block of text, exactly as it appeared in the minutes:
"""

def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except:
        return False

def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    home      = Path.home()
    work_dir  = home / "housing_project"
    data_dir  = work_dir / "data" / "meeting_minutes"
    raw_dir   = data_dir / "raw"
    model_dir = data_dir / "processed" / "minutes_extractor"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load hand-labelled JSONL ───────────────────────────────────
    ds = load_dataset(
        "json",
        data_files={ "train": str(raw_dir / "training.txt") },
    )["train"]
    ds = ds.train_test_split(test_size=0.1, seed=42)

    # ── Step 2: Tokenizer & Base Model ─────────────────────────────────────
    MODEL_NAME = "google/flan-t5-base"
    tokenizer  = T5Tokenizer.from_pretrained(MODEL_NAME)
    model      = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    max_input_length  = 512
    max_target_length = 256

    # ── Step 3: Preprocessing ───────────────────────────────────────────────
    def preprocess_fn(examples):
        inputs = [PROMPT_INSTRUCTION + raw for raw in examples["prompt"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["completion"],
                max_length=max_target_length,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    # ── Step 4: Data collator ───────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    # ── Step 5: Training arguments ─────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_dir),
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir=str(model_dir / "logs"),
        logging_steps=100,
        fp16=False,
        metric_for_best_model="valid_json_ratio",
        greater_is_better=True,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # replace -100 in the labels so we can decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        # add our JSON-validity metric
        valid_json_count = sum(is_valid_json(p) for p in decoded_preds)
        result["valid_json_ratio"] = valid_json_count / len(decoded_preds)
        # round everything
        return {k: round(v, 4) for k, v in result.items()}

    # ── Step 6: Trainer ────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Step 7: Train & save ───────────────────────────────────────────────
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"✔ Training complete — model saved to {model_dir}")

if __name__ == "__main__":
    main()