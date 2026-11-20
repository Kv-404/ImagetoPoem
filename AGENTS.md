# AGENTS.md — Image to Poem

Guide for AI coding agents (Grok, Claude, Cursor…) working in this repo.
Read this first, then `README.md` for the user-facing version.

## What this is

A college ML mini-project (scikit-learn syllabus). User uploads a photo and gets:
1. **caption**: BLIP `Salesforce/blip-image-captioning-base` (pretrained, frozen)
2. **mood** (happy / sad / calm / energetic): frozen CLIP `openai/clip-vit-base-patch32`
   embedding (512-d) → StandardScaler → PCA (90 % var) → best of 5 sklearn models
   (**trained by the student**)
3. **poem** (8 lines): `Qwen/Qwen2.5-0.5B-Instruct` + **LoRA adapter fine-tuned by the
   student** on their own image→poem dataset (`models/poet_lora/`)

Runs on a laptop (MacBook, Apple Silicon, 16 GB RAM), offline after first model download.
Demo: website with a "Studio" chat + Gradio UI, both served by `app.py`.

### Honesty rule (important)
Always describe the models truthfully: BLIP and CLIP are pretrained/frozen. The poem
model is a **LoRA fine-tune of Qwen2.5-0.5B-Instruct**, not trained from scratch. The mood
classifier is trained from scratch. Never hide or misstate what is pretrained in the code,
README, notebook or site. This is for a graded course.

## Environment

- Python **3.11** (Homebrew `/opt/homebrew/bin/python3.11`). The system python is 3.9, don't use it.
- venv at `./venv`. Always run with `venv/bin/python` or `source venv/bin/activate`.
- Run modules from the project root: `python -m src.<module>`.
- Set `HF_HUB_OFFLINE=1` (models are cached in `~/.cache/huggingface`).
- Shell is **zsh**: unmatched globs abort the whole command (`mv ./* ./.[!.]*` failed once
  this way). Prefer `find` or explicit names.
- Pinned versions are in `requirements.txt`. **Keep `torch==2.5.1`.** Installing new packages
  (e.g. `peft`) can silently upgrade torch. Re-pin after any install and run `pip check`.
- `pronouncing` needs `setuptools<81` (`pkg_resources`).

## Layout

```
app.py                 FastAPI server: /api/poem, /api/regenerate, /api/health,
                       Gradio UI at /gradio, static website (site/) at /
src/utils.py           paths (ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR), MOODS, timer(), best_device()
src/build_dataset.py   Openverse CC image downloader -> data/<mood>/*.jpg (~60 per mood)
src/features.py        CLIP embeddings -> models/features.npz (X, y)
src/train_mood.py      5 models (LogReg, KNN, DT, RF, MLP), Pipeline(scaler, PCA, clf),
                       stratified 80/20 + 5-fold CV, saves models/pca.joblib + mood_model.joblib
src/evaluate.py        metrics, confusion/PCA/elbow plots, clustering_report()
src/mood.py            load_mood_model() (clear FileNotFoundError), predict_mood()
src/caption.py         BLIP caption (CUDA if present, else CPU)
src/poem_metrics.py    explainable poem scores: lines, rhyme (CMU dict), keyword, mood, total
src/poem_data.py       builds image->poem pairs: 8 teacher-prompt candidates/image, keep best
                       -> data/poems/{train,val,test}.jsonl + scores.json (split by IMAGE)
src/train_poet.py      LoRA SFT (r=16, all attn+MLP proj), loss on poem tokens only,
                       batch 4 x grad-accum 2, grad checkpointing, 3 epochs, lr 2e-4
                       -> models/poet_lora/ + reports/poet_loss.csv
src/eval_poet.py       base vs fine-tuned on test images (3 poems each) -> reports/poet_eval*.csv
src/poet_plots.py      poem-model PNG plots
src/poem.py            load_poet() = base + LoRA merged (falls back to base), write_poem(),
                       PROMPT, BASE_NAME, poet_name()
src/make_samples.py    1 sample per mood with timings -> site/samples.json
src/site_data.py       every number/chart point for the site -> site/data.json + site/thumbs/
site/build_site.py     generates site/index.html (data inlined, SVG charts in vanilla JS)
notebook/ImageToPoem.ipynb     presentation notebook (executed, 17 graphs; RETRAIN_* flags)
notebook/Train_on_Colab.ipynb  same poem pipeline on a free Colab T4 (uses colab_bundle.zip)
data/<mood>/           251 images (not in git)
reports/               CSVs + PNG plots
```

## Common commands

```bash
# demo (website + chat at http://127.0.0.1:7860, Gradio at /gradio)
python app.py

# mood classifier
python -m src.build_dataset --per-mood 60
python -m src.features
python -m src.train_mood

# poem model (heavy: GPU. Stop app.py first to free RAM)
python -m src.poem_data      # ~1.5 h on Mac MPS, ~10 min Colab T4
python -m src.train_poet     # ~6 min on Mac MPS
python -m src.eval_poet      # ~5 min

# website refresh after any retrain
python -m src.make_samples && python -m src.site_data && python site/build_site.py

# notebook: execute in place so outputs are saved
cd notebook && ../venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 ImageToPoem.ipynb
```

API: `curl -F image=@photo.jpg http://127.0.0.1:7860/api/poem` →
`{caption, mood, confidence, probs{mood:p}, poem, timing{step:secs}}`.

## Current results (keep README/site in sync if these change)

- Data: 251 images (happy 60, sad 65, calm 65, energetic 61).
- Mood: best **KNN** CV 0.875 ± 0.05 (test acc 0.843). LogReg 0.870, RF 0.835, MLP 0.835,
  DT 0.680. PCA 512 → 98 comps (90 %). K-Means k=4 silhouette 0.05, ARI 0.28.
- Poem data: 270 train / 19 val pairs, 26 test images; only 8 % of targets rhyme.
- LoRA: 8.8 M trainable / 503 M (1.75 %). Val loss 2.13 → 1.89 (step 30) → 1.97 (overfits).
- Eval on 78 test poems: 8-line rate 0.28 → **0.99**, rhyme 0.04 → 0.03, keyword 0.60 → 0.48,
  mood 0.87 → 0.78, CLIP 0.251 → 0.255, 42 % faster. The model learned the format, not rhyme
  (data quality limit). Next improvement: rebuild the data keeping only rhyming candidates.
- App speed (CPU): ~7 s per image. `write_poem` is **best-of-3**: one batched generate with
  `num_return_sequences=3`, keep the poem with the highest `score_poem` total
  (`N_CANDIDATES` in `src/poem.py`). 8-poem check: total 0.36 → 0.40, rhyme 0.00 → 0.06,
  poem time 2.4 s → 5.4 s. Eval (`eval_poet.py`) still scores single raw samples.

## Gotchas learned the hard way

- **RAM:** app.py holds ~3 GB of models. Running training at the same time caused 6 GB of swap
  and an MPS OOM. Stop the server before `poem_data` / `train_poet` / `eval_poet`.
- **MPS OOM** with batch 8 → fixed with batch 4 × accum 2 + gradient checkpointing.
  Don't raise the batch size.
- `poem_data.py` writes files **only at the end** and walks moods in folder order. Killing it
  early loses everything and skews the moods. Pause with `kill -STOP`, resume with `-CONT`.
- When chaining commands, use `set -o pipefail`. A `| grep` once hid a crash and eval ran
  base-vs-base.
- `eval_poet.py` stores the mood label as `mood_label`. `mood` is the mood **score**.
- Qwen often ignores "8 lines". `write_poem` trims via `poem_metrics.format_8`, but eval
  scores **raw** output.
- After moving the project folder, rewrite the absolute paths in `venv/bin/*` and
  `venv/pyvenv.cfg` (or recreate the venv).
- `nbconvert` must be ≥ 7.16.6 (mistune 3.1 incompatibility).

## Website design rules

`site/build_site.py` builds a page in the **same style as the user's portfolio**
(`~/Desktop/everything/Portfolio`, Next.js, Apple-like strict monochrome). The skin lives in the
"portfolio skin" CSS block at the end of `CSS` and overrides the older rules above it.
No gradients, no emoji, no glow.

- **Fonts:** Geist (400/500/600) for text, Geist Mono (400/500) for labels and numbers (copied from
  the portfolio's `geist` package, OFL, `site/fonts/GEIST-LICENSE.txt`), IBM Plex Serif for poems.
  Self-hosted woff2 in `site/fonts/` so the page works offline. Don't add Google Fonts or a CDN.
- **Tokens (portfolio greys):** light paper `#fbfbfd` / surface `#fff` / line `#e5e5ea` /
  ink `#1d1d1f` / muted `#6e6e73`; dark true black `#000` / surface `#0d0d0e` /
  line `rgba(255,255,255,.09)` / ink `#f5f5f7` / muted `#a1a1a6`. Soft layered `--shadow`, glass nav.
- **Look:** floating glass pill nav (blur 20px, links as pills, active = filled pill);
  section masthead = mono `02 — LABEL` on a hairline, then a big tight title (letter-spacing
  -0.045em) ending in a grey period; mono uppercase tracked labels; 20px rounded cards;
  pill buttons (primary = ink fill, ghost = hairline); pill tabs and toggles.
- **Accent = ink** (`--accent` = foreground). "Mine" (trained / fine-tuned / chosen) is shown by
  solid ink: filled "Trained by me" tags, solid card borders, ink series in charts, bright
  confusion diagonal. Pretrained = dashed border. Base model in charts = grey `--base`.
  Results that got worse stay `--energetic` red (rhyme, caption words) so the page stays honest.
- **Mood colours** appear only where mood matters, always with a text label and a shape
  (happy circle, sad square, calm triangle, energetic diamond). Validated palette, don't change
  without re-validating CVD: light `#eda100 #6250d6 #1baf7a #e34948`; dark
  `#c98500 #9085e9 #199e70 #e34948`.
- **Layout (presentation order, the user's choice):** product first, then explanation.
  Section links sit in the floating glass nav (scrollspy; a second, sideways-scrolling row below
  1100px) with the Live/Offline dot (polls `/api/health` every 15 s) and theme toggle.
  9 sections: `try` (hero = the Studio: drop box left, result/example poem right, KPIs),
  `models` (6 always-open cards: BLIP, CLIP, mood classifier, Qwen, LoRA adapter, poem scorer),
  `dataset`, `mood` (5 compact classifier cards + charts), `unsupervised`, `poet-training`,
  `poet-eval`, `gallery`, `more` (tabs: Viva Q&A, The app, Syllabus, Limits, Glossary, Run it).
- **Charts sit side by side** in `.grid2` pairs (one column below 980px) so they can be
  explained together. Keep pages short: no long prose, one-line `.kick` under each heading.
  Glossary links (`#g-…`) open the Glossary tab first.
- The Data section id is `#dataset`, **not** `#data`, because `<script id="data">` holds the
  inlined JSON and the ids clashed.
- **Charts:** 14 inline SVG charts in vanilla JS, each with tooltips, a "Show table" toggle
  (tables scroll inside `.tbl-wrap`) and "How to read it / What it shows" captions.
- Numbers come from `data.json`. Two are typed in by hand because `data.json` lacks them:
  "42 % faster" and "8 % of targets rhyme". Update both if the poem model is retrained.
- Must work at 375px with no horizontal scroll, in light and dark. Keep the skip link and
  visible focus rings. Escape all model text.

The slide deck (claude.ai Slides artifact) still uses the older IBM Plex fonts and stone palette.
Never edit `site/index.html` by hand. Change `build_site.py` and rebuild.

## Code style

Simple, commented code a student can explain in a viva. Print timings with `timer()`.
Clear error messages for missing model files. Don't add heavy dependencies.

## Exact numbers (from result files)

Mood classifier, `reports/model_comparison.csv`:

| Model | CV acc | CV std | Test acc | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| KNN | 0.875 | 0.052 | 0.843 | 0.842 | 0.840 | 0.838 |
| Logistic Regression | 0.870 | 0.046 | 0.863 | 0.866 | 0.861 | 0.860 |
| Random Forest | 0.835 | 0.046 | 0.863 | 0.876 | 0.859 | 0.859 |
| MLP | 0.835 | 0.049 | 0.882 | 0.901 | 0.880 | 0.882 |
| Decision Tree | 0.680 | 0.033 | 0.706 | 0.715 | 0.704 | 0.704 |

Poem model, `reports/poet_eval_summary.csv` (mean over 78 poems on 26 test images):

| Model | 8 lines | Rhyme | Keyword | Mood | CLIP | Total |
|---|---|---|---|---|---|---|
| base | 0.282 | 0.042 | 0.599 | 0.865 | 0.251 | 0.409 |
| fine-tuned | 0.987 | 0.026 | 0.478 | 0.776 | 0.255 | 0.371 |

LoRA training, `models/poet_lora/train_info.json`:

```json
{
  "base": "Qwen/Qwen2.5-0.5B-Instruct",
  "train_pairs": 270,
  "val_pairs": 19,
  "epochs": 3,
  "batch": 8,
  "micro_batch": 4,
  "grad_accum": 2,
  "lr": 0.0002,
  "lora_r": 16,
  "trainable_params": 8798208,
  "total_params": 502830976,
  "train_seconds": 349.3,
  "device": "mps"
}
```

Sample outputs (`site/samples.json`, fine-tuned poet, best-of-3):

- **happy** (100%): "a group of people walking across a street holding balloons". Total 7.3 s {'caption': 1.33, 'mood': 0.21, 'poem': 5.79}
- **sad** (100%): "an apartment building with a tree in front of it". Total 6.0 s {'caption': 0.99, 'mood': 0.15, 'poem': 4.88}
- **calm** (100%): "the sun is setting over the ocean at sunset". Total 7.4 s {'caption': 1.04, 'mood': 0.12, 'poem': 6.25}
- **energetic** (100%): "a group of people sitting on the grass". Total 6.2 s {'caption': 0.89, 'mood': 0.16, 'poem': 5.18}
