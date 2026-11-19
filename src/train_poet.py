"""Fine-tune the poem model on our image -> poem pairs with LoRA (supervised).

Input  (x): "Write a short 8-line poem about: <caption>. Mood: <mood>. ..."
Target (y): the poem.  Loss = cross-entropy on the poem tokens only.

LoRA = freeze the 494M base weights, train small low-rank matrices
(A·B, rank 16) added to every attention + MLP layer (~1.8% of weights).

Run:  python -m src.train_poet
Out:  models/poet_lora/ (adapter), reports/poet_loss.csv, reports/poet_loss.png
"""
import csv
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.poem import BASE_NAME, PROMPT
from src.utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, best_device

POEMS = DATA_DIR / "poems"
LORA_DIR = MODELS_DIR / "poet_lora"
EPOCHS, BATCH, ACCUM, LR, SEED = 3, 4, 2, 2e-4, 42  # effective batch = 4 x 2 = 8
DEVICE = best_device()


def load_jsonl(name):
    return [json.loads(l) for l in open(POEMS / f"{name}.jsonl")]


def encode(tok, row):
    """Token ids + labels. Prompt tokens get label -100 (= ignored by the loss)."""
    msgs = [{"role": "user", "content": PROMPT.format(caption=row["caption"], mood=row["mood"])}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt)["input_ids"]
    y_ids = tok(row["poem"] + "<|im_end|>")["input_ids"]
    return {"input_ids": p_ids + y_ids, "labels": [-100] * len(p_ids) + y_ids}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)


@torch.no_grad()
def eval_loss(model, data, pad_id):
    model.eval()
    losses = [model(input_ids=i, attention_mask=m, labels=l).loss.item()
              for i, m, l in (collate(data[k:k + BATCH], pad_id) for k in range(0, len(data), BATCH))]
    model.train()
    return sum(losses) / len(losses)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_NAME)
    train = [encode(tok, r) for r in load_jsonl("train")]
    val = [encode(tok, r) for r in load_jsonl("val")]
    print(f"train pairs: {len(train)}   val pairs: {len(val)}   device: {DEVICE}")

    model = AutoModelForCausalLM.from_pretrained(BASE_NAME, torch_dtype=torch.float32).to(DEVICE)
    # Save memory: recompute activations in the backward pass instead of storing them
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} of {total:,} ({trainable / total:.2%})")

    steps = EPOCHS * math.ceil(len(train) / (BATCH * ACCUM))  # optimizer steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)

    log = [{"step": 0, "epoch": 0, "train_loss": None, "val_loss": eval_loss(model, val, tok.pad_token_id)}]
    print(f"step 0  val_loss {log[0]['val_loss']:.3f}  (before training)")
    step, t0 = 0, time.perf_counter()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train)
        batches = [train[k:k + BATCH] for k in range(0, len(train), BATCH)]
        for b in range(0, len(batches), ACCUM):
            # gradient accumulation: ACCUM small batches -> one optimizer step
            group = batches[b:b + ACCUM]
            total_loss = 0.0
            for mb in group:
                ids, mask, labels = collate(mb, tok.pad_token_id)
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / len(group)
                loss.backward()
                total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            step += 1
            row = {"step": step, "epoch": round(step / (steps / EPOCHS), 2), "train_loss": total_loss, "val_loss": None}
            if step % 10 == 0 or step == steps:
                row["val_loss"] = eval_loss(model, val, tok.pad_token_id)
                print(f"step {step}/{steps}  train {total_loss:.3f}  val {row['val_loss']:.3f}  "
                      f"({time.perf_counter() - t0:.0f}s)")
            log.append(row)

    print(f"[time] training: {time.perf_counter() - t0:.1f}s")
    model.save_pretrained(LORA_DIR)
    json.dump({"base": BASE_NAME, "train_pairs": len(train), "val_pairs": len(val), "epochs": EPOCHS,
               "batch": BATCH * ACCUM, "micro_batch": BATCH, "grad_accum": ACCUM, "lr": LR, "lora_r": 16, "trainable_params": trainable,
               "total_params": total, "train_seconds": round(time.perf_counter() - t0, 1),
               "device": DEVICE}, open(LORA_DIR / "train_info.json", "w"), indent=2)
    with open(REPORTS_DIR / "poet_loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "val_loss"])
        w.writeheader(); w.writerows(log)
    print(f"Saved adapter -> {LORA_DIR}")
    from src.poet_plots import plot_loss
    plot_loss()


if __name__ == "__main__":
    main()
