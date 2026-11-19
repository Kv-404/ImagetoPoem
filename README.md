# Image to Poem 🖼️ → 📝

Upload a photo. The app writes a **caption**, predicts the **mood**
(happy / sad / calm / energetic) with a classifier *I trained* in scikit-learn,
then a poem model *I fine-tuned* on my own image → poem dataset writes an
8-line **poem**. Everything runs on a laptop, offline after the first download.

```
image ──► BLIP (pretrained) ────────────────────────► caption ─┐
   │                                                           ├─► my fine-tuned poet ─► poem
   └───► CLIP (frozen) ─► scale ─► PCA ─► my mood classifier ─► mood ─┘   (LoRA on Qwen2.5-0.5B-Instruct)
```

| Part | Pretrained or mine? |
|---|---|
| Caption — BLIP base | pretrained, frozen |
| Image features — CLIP ViT-B/32 | pretrained, frozen (feature extractor) |
| Mood classifier — scaler + PCA + 5 compared models | **trained by me from scratch** (scikit-learn) |
| Poem model — base Qwen2.5-0.5B-Instruct | **fine-tuned by me** with LoRA on 270 image→poem pairs |

## 1. Setup (Python 3.11)

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads models (~2.5 GB) to `~/.cache/huggingface`.
After that, work fully offline: `export HF_HUB_OFFLINE=1`.

## 2. Mood classifier (supervised + unsupervised)

```bash
python -m src.build_dataset --per-mood 60   # CC images from Openverse -> data/<mood>/
python -m src.features                      # CLIP embeddings -> models/features.npz
python -m src.train_mood                    # compare 5 models, save best, plots
```

**Check the images by hand** (delete ones that don't fit the mood), then re-run
the last two commands.

## 3. Poem model (supervised fine-tuning, LoRA)

```bash
python -m src.poem_data    # build image -> poem pairs  (~1.5 h on Mac GPU, ~10 min on Colab T4)
python -m src.train_poet   # LoRA fine-tune             (~6 min on Mac GPU)
python -m src.eval_poet    # base vs fine-tuned on unseen test images
```

- **Dataset** (`src/poem_data.py`): each image is captioned; its mood comes
  from my folder labels. The base model writes 8 candidate poems with a long,
  strict "teacher" prompt (rules + example). My scorer
  (`src/poem_metrics.py`: rhyme, caption words, mood words, penalties) keeps the
  best → 270 train / 19 val pairs. 26 test images get no poems. The split is by
  image, so nothing leaks. This is called **rejection-sampling / prompt distillation**.
- **Training** (`src/train_poet.py`): input = short prompt (caption + mood),
  target = poem, cross-entropy on poem tokens only. LoRA rank 16 on all attention
  + MLP layers = 8.8 M trainable of 503 M (1.75 %). Effective batch 8 (4 × 2
  gradient accumulation), gradient checkpointing, 3 epochs, lr 2e-4, cosine schedule.
- **No GPU at home / Mac too hot?** `notebook/Train_on_Colab.ipynb` runs the
  same three steps on a free Colab T4. Upload `colab_bundle.zip`, then unzip the
  downloaded `colab_results.zip` into this folder.

## 4. Run the demo

```bash
python app.py
```

| URL | What |
|---|---|
| http://127.0.0.1:7860/ | **website + Studio chat**: drop an image → caption, mood, poem, timing |
| http://127.0.0.1:7860/gradio | classic Gradio UI with **Regenerate poem** |
| `POST /api/poem` (field `image`) | JSON API: `curl -F image=@photo.jpg http://127.0.0.1:7860/api/poem` |

If a model file is missing, the app stops with a clear message saying what to run.

## 5. Notebook + website

- **Notebook:** `notebook/ImageToPoem.ipynb` covers the whole project with 17
  graphs, already executed. To re-run it live: `jupyter notebook notebook/ImageToPoem.ipynb`.
  It loads the saved results, so it runs in a few minutes. Set the `RETRAIN_*` flags
  at the top to redo the heavy steps.
- **Website:** after training, run
  ```bash
  python -m src.make_samples   # 1 sample per mood (with timings) -> site/samples.json
  python -m src.site_data      # all chart data -> site/data.json
  python site/build_site.py    # -> site/index.html (interactive charts, light/dark)
  ```

## Results

**Mood classifier** (251 images, stratified 80/20 split + 5-fold CV):

| Model | 5-fold CV acc | Test acc | Test F1 |
|---|---|---|---|
| **KNN (best by CV)** | **0.875 ± 0.05** | 0.843 | 0.838 |
| Logistic Regression | 0.870 ± 0.05 | 0.863 | 0.860 |
| Random Forest | 0.835 ± 0.05 | 0.863 | 0.859 |
| MLP | 0.835 ± 0.05 | 0.882 | 0.882 |
| Decision Tree | 0.680 ± 0.03 | 0.706 | 0.704 |

PCA: 512 → 98 components keep 90 % of the variance. K-Means k=4: silhouette 0.05,
ARI vs moods 0.28. Moods are not natural clusters, which is why supervised learning is needed.

**Poem model**, 78 poems per model on 26 unseen test images:

| Metric | Base | Fine-tuned |
|---|---|---|
| Exactly 8 lines | 0.28 | **0.99** |
| Rhyming pairs | 0.04 | 0.03 |
| Uses caption words | 0.60 | 0.48 |
| Mood words | 0.87 | 0.78 |
| CLIP image↔poem similarity | 0.251 | 0.255 |
| Time for 78 poems (Mac GPU) | 194 s | **112 s** |

Val loss went 2.13 → 1.89 (best, step 30, about epoch 1), then slowly rose to 1.97 while train
loss kept falling to ~1.0. That is **overfitting** on a small dataset (270 pairs).

What it learned: the **format**. Two verses, exactly 8 lines, shorter and faster
output. What it didn't learn: **rhyme**. Only 8 % of the training poems rhymed,
so the model never saw enough of it. The model can only be as good as its data.
Next step: rebuild the data and keep only rhyming candidates.

**Best-of-3 at run time.** The app writes 3 poems in one batch and shows the one my
poem scorer rates highest (`N_CANDIDATES` in `src/poem.py`). The model is unchanged;
this is re-ranking. On 8 test poems it raised the mean score from 0.36 to 0.40 and
rhyme from 0.00 to 0.06: a small gain, at about twice the poem time.

**Speed** (16 GB Apple Silicon, CPU, app): caption ~1 s, mood ~0.1–0.3 s, poem ~5.5 s
(best of 3; ~2.5 s with `n=1`) → **~7 s per image**.

## Syllabus map

| Syllabus module | Where | What to say in viva |
|---|---|---|
| Preprocessing | `features.py`, `train_mood.py` | CLIP turns image → 512 numbers; L2-normalise; `StandardScaler`. |
| PCA | `train_mood.py`, `site_data.py` | Keep 90 % variance (512 → 98). Fewer features = less overfitting. |
| Classification | `train_mood.py` | Logistic Regression, KNN, Decision Tree on the same pipeline. |
| Ensembles | `train_mood.py` | Random Forest = many trees on bootstrap samples, majority vote. |
| Neural network | `train_mood.py`, `train_poet.py` | `MLPClassifier` (128 ReLU units); transformer fine-tuning. |
| Evaluation | `evaluate.py`, `eval_poet.py` | Stratified split, 5-fold CV, acc/precision/recall/F1, confusion matrix; poem metrics on held-out images. |
| Clustering | `evaluate.py`, `site_data.py` | K-Means elbow + silhouette, ARI vs moods, PCA / t-SNE maps. |
| Transfer learning / fine-tuning | `poem_data.py`, `train_poet.py` | Supervised LoRA fine-tuning (W + B·A, rank 16), loss masking, train vs val loss. |
| Model saving / deployment | `train_mood.py`, `train_poet.py`, `app.py` | `joblib` + LoRA adapter; FastAPI + Gradio app; website chat. |

## Viva notes

- **Is the poem model trained from scratch?** No. I fine-tuned a pretrained
  0.5 B model with LoRA. Training a language model from scratch needs billions of
  words and many GPUs. Fine-tuning 1.75 % of the weights takes 6 minutes on a laptop.
- **Why a Pipeline?** Scaler and PCA are fit *inside* each CV fold, so test folds never leak.
- **Why pick by CV, not test?** The test set stays an honest final check.
- **Why split poems by image?** So test images are never seen in training.
- **Why did rhyme not improve?** The training targets barely rhymed (8 %).
  Data quality limits the model.

## Files

```
app.py                demo server: website + chat API + Gradio UI
src/utils.py          paths, moods, timer, device
src/build_dataset.py  Openverse downloader
src/features.py       CLIP embeddings
src/train_mood.py     train + compare + save mood classifier
src/evaluate.py       metrics, plots, clustering
src/mood.py           load model + predict mood
src/caption.py        BLIP caption
src/poem_metrics.py   rhyme / keyword / mood / line scores
src/poem_data.py      build image -> poem dataset
src/train_poet.py     LoRA fine-tuning
src/eval_poet.py      base vs fine-tuned evaluation
src/poet_plots.py     poem-model plots
src/poem.py           load fine-tuned poet + write poem
src/make_samples.py   sample poems for the website
src/site_data.py      all chart data for the website
notebook/             ImageToPoem.ipynb (presentation) · Train_on_Colab.ipynb
site/                 build_site.py -> index.html
```
