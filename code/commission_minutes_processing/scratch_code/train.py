#!/usr/bin/env python3
import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    home       = Path.home()
    work_dir   = home / "housing_project"
    data       = work_dir / "data"
    minutes    = data / "meeting_minutes"
    raw_dir    = minutes / "raw"
    model_dir  = minutes / "processed" / "minutes_extractor"
    jsonl_path = raw_dir / "training.jsonl"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load hand-labelled JSONL ──────────────────────────────────
    # expects lines: {"prompt":"...", "completion":"{...JSON...}"}
    train_dataset = load_dataset(
        "json", data_files=str(jsonl_path), split="train"
    )

    # ── Step 2: Tokenizer + Preprocessing ───────────────────────────────
    model_name = "google/flan-t5-base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    max_input_length  = 512
    max_target_length = 256

    def preprocess_fn(batch):
        inputs = ["extract project fields:\n" + x for x in batch["prompt"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["completion"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_data = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    # ── Step 3: Load base model + apply LoRA ─────────────────────────────
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=None)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, peft_config)

    # ── Step 4: Training arguments ────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=3e-4,
        save_steps=500,
        logging_steps=100,
        fp16=False,                 # disable mixed-precision on macOS
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer
    )

    # ── Step 5: Train & save ──────────────────────────────────────────────
    trainer.train()
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Training complete — model saved to {model_dir}")

if __name__ == "__main__":
    main()
