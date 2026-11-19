"""Image caption with BLIP (pretrained, frozen)."""
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.utils import timer

BLIP_NAME = "Salesforce/blip-image-captioning-base"
_blip = None


def load_blip():
    global _blip
    if _blip is None:
        with timer("load BLIP"):
            # NVIDIA GPU if present (Colab), else CPU (laptop demo)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _blip = (BlipForConditionalGeneration.from_pretrained(BLIP_NAME).to(dev).eval(),
                     BlipProcessor.from_pretrained(BLIP_NAME))
    return _blip


@torch.no_grad()
def caption_image(image):
    """PIL image -> short English caption."""
    model, proc = load_blip()
    inputs = proc(images=image.convert("RGB"), return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=30, num_beams=3)
    return proc.decode(ids[0], skip_special_tokens=True).strip()
