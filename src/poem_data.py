"""Build the image -> poem training set.

For every image in data/<mood>/:
  1. caption it (BLIP) - mood label comes from the folder (our labels)
  2. the base language model writes 8 candidate poems with a long, strict
     "teacher" prompt (rules + example)
  3. score each candidate (src/poem_metrics.py) and keep the best ones
Training input is the SHORT prompt only -> after fine-tuning the model writes
good 8-line rhyming poems without needing the long rules.
(This is called prompt distillation / rejection-sampling fine-tuning.)

Run:  python -m src.poem_data
Out:  data/poems/{train,val,test}.jsonl  (split by IMAGE, so no leakage)
"""
import json
import random

import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from src.caption import caption_image
from src.features import list_dataset
from src.poem import BASE_NAME, PROMPT
from src.poem_metrics import format_8, score_poem
from src.utils import DATA_DIR, best_device, timer

OUT = DATA_DIR / "poems"
N_CANDIDATES = 8
KEEP_PER_IMAGE = 2
MIN_SCORE = 0.45

TEACHER_SYSTEM = """You are a careful poet. Rules:
- Exactly 8 lines: two verses of 4 lines, blank line between them.
- Rhyme scheme AABB in each verse (line 1 rhymes with 2, line 3 with 4).
- Simple words, vivid pictures, max 9 words per line.
- Talk about the things in the scene. Match the mood.
- Output only the poem. No title, no notes."""

EXAMPLE_USER = PROMPT.format(caption="a boat on the sea at sunset", mood="calm")
EXAMPLE_POEM = """The sun sinks low upon the sea,
A golden path for you and me.
The little boat rocks soft and slow,
While gentle winds begin to blow.

The waves hum quiet songs to rest,
The sky glows pink along the west.
The first stars wake, so small and bright,
And calm comes drifting with the night."""


def device():
    return best_device()


def load_teacher():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_NAME, torch_dtype=torch.float32)
    return model.to(device()).eval(), tok


@torch.no_grad()
def candidates(model, tok, caption, mood, n=N_CANDIDATES):
    messages = [{"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": EXAMPLE_USER},
                {"role": "assistant", "content": EXAMPLE_POEM},
                {"role": "user", "content": PROMPT.format(caption=caption, mood=mood)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.9,
                         top_p=0.95, repetition_penalty=1.1, num_return_sequences=n,
                         pad_token_id=tok.eos_token_id)
    start = inputs["input_ids"].shape[1]
    return [tok.decode(o[start:], skip_special_tokens=True).strip() for o in out]


def main():
    random.seed(42)
    torch.manual_seed(42)
    paths, labels = list_dataset()
    # split IMAGES 80/10/10, stratified by mood
    idx = list(range(len(paths)))
    tr, rest = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    va, te = train_test_split(rest, test_size=0.5, stratify=[labels[i] for i in rest], random_state=42)
    split_of = {**{i: "train" for i in tr}, **{i: "val" for i in va}, **{i: "test" for i in te}}

    model, tok = load_teacher()
    print(f"Teacher on {model.device}; {len(paths)} images")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}
    kept_scores, all_scores = [], []

    with timer("build poem dataset"):
        for k, i in enumerate(idx):
            img = Image.open(paths[i]).convert("RGB")
            caption, mood, split = caption_image(img), labels[i], split_of[i]
            base = {"image": str(paths[i].relative_to(DATA_DIR)), "caption": caption, "mood": mood}
            if split == "test":                  # test images: no poems, model must write them
                rows["test"].append(base)
                continue
            scored = []
            for c in candidates(model, tok, caption, mood):
                poem = format_8(c)
                s = score_poem(poem, caption, mood)
                all_scores.append(s["total"])
                if s["n_lines"] == 8:
                    scored.append((s["total"], poem))
            scored.sort(reverse=True)
            keep = KEEP_PER_IMAGE if split == "train" else 1
            for total, poem in scored[:keep]:
                if total >= MIN_SCORE:
                    rows[split].append({**base, "poem": poem, "score": round(total, 3)})
                    kept_scores.append(total)
            if k % 20 == 0:
                print(f"  {k}/{len(idx)}  train={len(rows['train'])} val={len(rows['val'])}")

    for split, items in rows.items():
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
    json.dump({"candidate_scores": all_scores, "kept_scores": kept_scores},
              open(OUT / "scores.json", "w"))
    print({k: len(v) for k, v in rows.items()}, "-> data/poems/")


if __name__ == "__main__":
    main()
