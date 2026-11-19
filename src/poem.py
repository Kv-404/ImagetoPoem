"""Poem model: our LoRA fine-tuned poet (falls back to the base model if not trained)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.poem_metrics import format_8, score_poem
from src.utils import MODELS_DIR, timer

BASE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # base model we fine-tune
LORA_DIR = MODELS_DIR / "poet_lora"         # our trained adapter (src/train_poet.py)
PROMPT = ("Write a short 8-line poem about: {caption}. Mood: {mood}. "
          "Use vivid imagery, simple words, and a gentle rhyme.")
_poet = None


def load_base():
    tok = AutoTokenizer.from_pretrained(BASE_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_NAME, torch_dtype=torch.float32).eval()
    return model, tok


def load_poet():
    """Base + our LoRA adapter, merged into one model for fast inference."""
    global _poet
    if _poet is None:
        with timer("load poem model"):
            model, tok = load_base()
            if LORA_DIR.exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, LORA_DIR).merge_and_unload().eval()
                print("Poem model: fine-tuned (LoRA adapter from models/poet_lora)")
            else:
                print("Poem model: base (not fine-tuned yet - run python -m src.train_poet)")
            _poet = (model, tok)
    return _poet


load_qwen = load_poet  # old name


def poet_name():
    return "fine-tuned poet (LoRA)" if LORA_DIR.exists() else "base poet"


N_CANDIDATES = 3   # best-of-N: write 3 poems in one batch, keep the one my scorer likes most


@torch.no_grad()
def write_poem(caption, mood, temperature=0.9, n=N_CANDIDATES):
    """Write n poems at once (one batched generate call) and return the best one.

    "Best" = highest total from src/poem_metrics.score_poem (rhyme, caption words,
    mood words, minus penalties). This is called re-ranking: the model is unchanged,
    we just pick the strongest of several samples.
    """
    model, tok = load_poet()
    messages = [{"role": "user", "content": PROMPT.format(caption=caption, mood=mood)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True, num_return_sequences=n,
                         temperature=temperature, top_p=0.9, repetition_penalty=1.1,
                         pad_token_id=tok.eos_token_id)
    start = inputs["input_ids"].shape[1]                 # drop the prompt part
    poems = [format_8(tok.decode(o[start:], skip_special_tokens=True).strip()) for o in out]
    return max(poems, key=lambda p: score_poem(p, caption, mood)["total"])


if __name__ == "__main__":
    print(write_poem("a dog running on a beach at sunset", "energetic"))
