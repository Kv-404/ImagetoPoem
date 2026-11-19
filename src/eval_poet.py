"""Compare base vs fine-tuned poem model on TEST images (never seen in training).

Raw model output is scored (no trimming), 3 poems per image per model.
Run:  python -m src.eval_poet
Out:  reports/poet_eval.csv, poet_eval_summary.csv, poet_*.png
"""
import json

import pandas as pd
import torch
from PIL import Image

from src.features import load_clip
from src.poem import PROMPT, load_base, load_poet
from src.poem_metrics import score_poem
from src.poet_plots import plot_eval
from src.utils import DATA_DIR, REPORTS_DIR, best_device, timer

N_PER_IMAGE = 3
DEVICE = best_device()


@torch.no_grad()
def generate(model, tok, caption, mood, n=N_PER_IMAGE):
    msgs = [{"role": "user", "content": PROMPT.format(caption=caption, mood=mood)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=160, do_sample=True, temperature=0.9, top_p=0.9,
                         repetition_penalty=1.1, num_return_sequences=n, pad_token_id=tok.eos_token_id)
    start = inputs["input_ids"].shape[1]
    return [tok.decode(o[start:], skip_special_tokens=True).strip() for o in out]


@torch.no_grad()
def clip_score(image, poem):
    model, proc = load_clip()
    inp = proc(text=[poem], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    out = model(**inp)
    a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((a * b).sum())


def main():
    torch.manual_seed(0)
    test = [json.loads(l) for l in open(DATA_DIR / "poems" / "test.jsonl")]
    rows = []
    for name, loader in (("base", load_base), ("fine-tuned", load_poet)):
        model, tok = loader()
        model.to(DEVICE)
        with timer(f"{name}: {len(test) * N_PER_IMAGE} poems"):
            for r in test:
                img = Image.open(DATA_DIR / r["image"]).convert("RGB")
                for poem in generate(model, tok, r["caption"], r["mood"]):
                    s = score_poem(poem, r["caption"], r["mood"])
                    rows.append({"model": name, "image": r["image"], "caption": r["caption"], "mood_label": r["mood"],
                                 "poem": poem, "clip": clip_score(img, poem), **s})
        model.to("cpu")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "poet_eval.csv", index=False)
    summary = df.groupby("model")[["lines", "rhyme", "keyword", "mood", "clip", "total"]].mean().round(3)
    summary.to_csv(REPORTS_DIR / "poet_eval_summary.csv")
    print(summary.to_string())
    plot_eval()
    print("Saved reports/poet_eval*.csv and poet_*.png")


if __name__ == "__main__":
    main()
