"""Demo server: website + chat API + Gradio UI, all on one port.

Run:  python app.py
  http://127.0.0.1:7860/         website with chat (drop image -> poem)
  http://127.0.0.1:7860/gradio   classic Gradio UI
  POST /api/poem (image file)    JSON: caption, mood, poem, timing
"""
import io
import sys
import threading
import time

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.caption import caption_image, load_blip
from src.mood import load_mood_model, predict_mood
from src.poem import load_poet, poet_name, write_poem
from src.utils import MOODS, ROOT

# --- load everything once at start (so the demo itself is fast) ---
try:
    PCA, CLF, MODEL_NAME = load_mood_model()
except FileNotFoundError as e:
    print(f"\nERROR: {e}\n")
    sys.exit(1)
load_blip()
load_poet()
POET = poet_name()
LOCK = threading.Lock()  # one request at a time on CPU (keeps timings honest)


def make_poem(image):
    """Full pipeline for one PIL image -> dict with results + timing."""
    times = {}
    with LOCK:
        t = time.perf_counter()
        caption = caption_image(image)
        times["caption (BLIP)"] = time.perf_counter() - t

        t = time.perf_counter()
        mood, conf, probs = predict_mood(image, PCA, CLF)
        times[f"mood (CLIP + {MODEL_NAME})"] = time.perf_counter() - t

        t = time.perf_counter()
        poem = write_poem(caption, mood)
        times[f"poem ({POET})"] = time.perf_counter() - t
    print("[time]", " | ".join(f"{k}: {v:.2f}s" for k, v in times.items()))
    return {"caption": caption, "mood": mood, "confidence": conf, "probs": probs,
            "poem": poem, "timing": {k: round(v, 2) for k, v in times.items()}}


def new_poem(caption, mood):
    with LOCK:
        t = time.perf_counter()
        poem = write_poem(caption, mood)
        dt = time.perf_counter() - t
    print(f"[time] regenerate poem: {dt:.2f}s")
    return poem, round(dt, 2)


# ---------------- JSON API (used by the website chat) ----------------
api = FastAPI(title="Image to Poem")


@api.post("/api/poem")
def api_poem(image: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not an image.")
    return make_poem(img)


@api.post("/api/regenerate")
def api_regenerate(caption: str = Form(...), mood: str = Form(...)):
    if mood not in MOODS:
        raise HTTPException(400, f"mood must be one of {MOODS}")
    poem, dt = new_poem(caption, mood)
    return {"poem": poem, "timing": {f"poem ({POET})": dt}}


@api.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


# ---------------- Gradio UI (at /gradio) ----------------
def run(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    r = make_poem(image)
    timing = "\n".join(f"{k}: {v:.2f}s" for k, v in r["timing"].items())
    return r["caption"], r["probs"], r["poem"], timing, r["caption"], r["mood"]


def regenerate(caption, mood):
    if not caption:
        raise gr.Error("Generate a poem first.")
    poem, dt = new_poem(caption, mood)
    return poem, f"poem ({POET}): {dt:.2f}s"


with gr.Blocks(title="Image to Poem") as demo:
    gr.Markdown(f"# Image to Poem\nBLIP caption · CLIP + **{MODEL_NAME}** mood · **{POET}**")
    cap_state, mood_state = gr.State(""), gr.State("")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Upload image")
            go = gr.Button("Make poem", variant="primary")
            again = gr.Button("Regenerate poem")
        with gr.Column():
            caption_box = gr.Textbox(label="Caption")
            mood_box = gr.Label(label="Predicted mood (confidence)", num_top_classes=4)
            poem_box = gr.Textbox(label="Poem", lines=10)
            time_box = gr.Textbox(label="Timing", lines=3)

    go.click(run, img, [caption_box, mood_box, poem_box, time_box, cap_state, mood_state])
    again.click(regenerate, [cap_state, mood_state], [poem_box, time_box])

api = gr.mount_gradio_app(api, demo, path="/gradio")
# Website last: "/" catches everything not matched above.
api.mount("/", StaticFiles(directory=ROOT / "site", html=True), name="site")

if __name__ == "__main__":
    print("Website + chat:  http://127.0.0.1:7860/")
    print("Gradio UI:       http://127.0.0.1:7860/gradio")
    uvicorn.run(api, host="127.0.0.1", port=7860, log_level="warning")
