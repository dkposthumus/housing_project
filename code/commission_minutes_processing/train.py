#!/usr/bin/env python3
"""
train.py
"""

import json, numpy as np, torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

# ───────────────────────────────── prompt ────────────────────────────────
PROMPT_INSTRUCTION = """You are a strict JSON extractor. Given a block of raw meeting text, extract and return exactly one valid double-quoted JSON object with the keys below (any order). 
Only output the JSON—no commentary. Missing fields → empty string "".

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
- action_name

Raw block:
"""

EOJ_TOKEN = "<extra_id_0>"  
# ───────────────────────────────── helpers ───────────────────────────────
def is_valid_json(txt: str) -> bool:
    try:
        json.loads(txt)
        return True
    except Exception:
        return False

# ───────────────────────────────── main ──────────────────────────────────
def main():
    # ── paths ────────────────────────────────────────────────────────────
    home = Path.home()
    work = home / "housing_project"
    data = work / "data" / "meeting_minutes"
    raw  = data / "tagged"
    out  = data / "processed" / "minutes_extractor"
    out.mkdir(parents=True, exist_ok=True)

    # ── dataset ──────────────────────────────────────────────────────────
    ds = load_dataset("json", data_files={"train": str(raw / "training.txt")})["train"]
    ds = ds.train_test_split(test_size=0.1, seed=42)

    # ── model / tokenizer ────────────────────────────────────────────────
    MODEL_NAME = "google/flan-t5-small"
    tokenizer  = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.gradient_checkpointing_enable()

    max_in, max_out = 512, 256

    # ── preprocessing ────────────────────────────────────────────────────
    def preprocess(ex):
        inputs = [PROMPT_INSTRUCTION + txt for txt in ex["prompt"]]
        model_in = tokenizer(inputs, max_length=max_in, truncation=True)
        labels = tokenizer(
            text_target=ex["completion"],          # no extra token added
            max_length=max_out,
        truncation=True,
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    tokenized = ds.map(preprocess, batched=True, remove_columns=["prompt", "completion"])

    # ── collator ─────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ── training args ────────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        learning_rate=3e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="valid_json_ratio",
        greater_is_better=True,
        generation_max_length=600,
        generation_num_beams=1,
        seed=42,
        fp16=False,                              # ignored on M-series
    )

    # ── metric fn ────────────────────────────────────────────────────────
    def compute_metrics(eval_preds):
        preds, _ = eval_preds
        # make sure we have `int32` (some back-ends give int64 or float32)
        preds = np.asarray(preds, dtype=np.int32)
        # strip any IDs that are outside the SentencePiece vocab
        vocab_size = tokenizer.vocab_size
        cleaned = [[tok for tok in seq if 0 <= tok < vocab_size] for seq in preds]
        decoded = tokenizer.batch_decode(cleaned, skip_special_tokens=True)        
        #  ── DEBUG: show one sample every epoch ───────────────────────────────
        print("\n── sample pred ─────────────────────────────────────────")
        print(decoded[0][:500])
        print("────────────────────────────────────────────────────────\n")
        decoded = [d.split(EOJ_TOKEN)[0] for d in decoded]   # safety strip
        valid   = sum(is_valid_json(d) for d in decoded)
        return {"valid_json_ratio": valid / len(decoded)}

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print("✓ training complete – model saved to", out)

if __name__ == "__main__":
    main()
