"""Build the project website: site/index.html (stdlib only).

Run from the project root:
    python site/build_site.py                                  # site/data.json -> site/index.html
    python site/build_site.py --data other.json --out x.html   # custom input / output

site/data.json comes from `python -m src.site_data`. It is inlined into the page
as <script type="application/json" id="data">, so index.html works on file://
and when served by app.py (http://127.0.0.1:7860/). All charts are drawn in the
browser as inline SVG with plain JavaScript (no libraries, no CDN).
Fonts (Geist + IBM Plex Serif) are self-hosted in site/fonts/ so the page works offline.

Design: monochrome and editorial. Ink = "trained by me / fine-tuned",
mid grey = "pretrained / base". Mood colours only where the mood itself matters.
"""
import argparse
import datetime
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- fonts
FONTS = [  # (family, weight, style, file in site/fonts/). Geist = same font as my portfolio (OFL).
    ("Geist", 400, "normal", "geist-400.woff2"),
    ("Geist", 500, "normal", "geist-500.woff2"),
    ("Geist", 600, "normal", "geist-600.woff2"),
    ("Geist Mono", 400, "normal", "geist-mono-400.woff2"),
    ("Geist Mono", 500, "normal", "geist-mono-500.woff2"),
    ("IBM Plex Serif", 400, "normal", "ibm-plex-serif-latin-400-normal.woff2"),
    ("IBM Plex Serif", 400, "italic", "ibm-plex-serif-latin-400-italic.woff2"),
    ("IBM Plex Serif", 600, "normal", "ibm-plex-serif-latin-600-normal.woff2"),
]


def font_css():
    return "\n".join(
        f'@font-face{{font-family:"{fam}";src:url(fonts/{f}) format("woff2");'
        f"font-weight:{w};font-style:{s};font-display:swap}}"
        for fam, w, s, f in FONTS if (SITE / "fonts" / f).exists())


# ---------------------------------------------------------------- sections (TOC)
# Order = the order I present in class: product first, then how it works, then proof.
SECTIONS = [
    ("try", "Try it"), ("models", "The 6 models"), ("dataset", "Mood data"),
    ("mood", "Mood results"), ("unsupervised", "Clustering"), ("poet-training", "Poem training"),
    ("poet-eval", "Poem results"), ("gallery", "Samples"), ("more", "Viva & more"),
]

# ---------------------------------------------------------------- glossary
# id -> (term, plain-English definition). Used for the glossary section AND for
# inline terms written as [[label|id]] in the HTML below (title = definition).
GLOSSARY = {
    "pretrained": ("Pretrained / frozen", "A model whose weights someone else already learned on a large dataset. Frozen means I use those weights as they are and never change them."),
    "transfer": ("Transfer learning", "Reusing what a model learned on a big task as the starting point for a small one. I reuse CLIP's image features instead of learning vision from 251 photos."),
    "embedding": ("Embedding", "A list of numbers that describes something, here a photo. Images that look or mean alike get similar numbers. CLIP gives each image 512 numbers."),
    "scaler": ("Standardisation", "Shifting and stretching each feature to mean 0 and standard deviation 1, so no single feature dominates distances."),
    "pca": ("PCA", "Principal Component Analysis. Rotates the data so the first new axes hold most of the variation; the rest can be dropped."),
    "knn": ("k-nearest neighbours", "Labels a new point by a vote of the k most similar training points."),
    "cv": ("Cross-validation", "Split the training data into k parts, train on k-1 and test on the one left out, repeat k times and average. A fairer score than one split."),
    "stratified": ("Stratified split", "A split that keeps the same share of each class in every part."),
    "leakage": ("Data leakage", "When information from test data sneaks into training (for example fitting PCA on all images), so the score looks better than it really is."),
    "overfit": ("Overfitting", "The model memorises the training data and gets worse on new data. Visible as training loss falling while validation loss rises."),
    "accuracy": ("Accuracy", "The share of predictions that are correct."),
    "confusion": ("Confusion matrix", "A table of true class (rows) against predicted class (columns). The diagonal holds the correct answers."),
    "prf": ("Precision, recall, F1", "Precision: of the images I called sad, how many were sad. Recall: of the sad images, how many I found. F1: the harmonic mean of the two."),
    "finetune": ("Fine-tuning", "Continuing to train a pretrained model on my own small dataset so it adapts to my task."),
    "lora": ("LoRA", "Low-Rank Adaptation. Freeze the big weight matrices and learn two small matrices whose product is added to them. Cheap fine-tuning."),
    "token": ("Token", "The pieces a language model reads and writes: whole words or parts of words."),
    "loss": ("Loss", "The number training tries to make small. For text it is cross-entropy: how surprised the model is by the correct next token."),
    "epoch": ("Epoch", "One full pass over the training data."),
    "batch": ("Batch", "The group of examples used for one weight update."),
    "accum": ("Gradient accumulation", "Add up gradients from several small batches and update once. Acts like a bigger batch with less memory."),
    "sampling": ("Temperature and top-p", "How the model picks the next token. Temperature below 1 makes choices safer, above 1 more random. Top-p keeps only the most likely tokens that together make up p of the probability."),
    "kmeans": ("K-Means", "Clustering that splits data into k groups by repeatedly assigning points to the nearest centre and moving each centre to the mean of its points."),
    "silhouette": ("Silhouette score", "Cluster quality from -1 to 1: how much closer each point is to its own cluster than to the next one. Near 0 means the clusters overlap."),
    "ari": ("ARI", "Adjusted Rand Index. Agreement between two groupings (here clusters and moods), corrected for chance: 0 = random, 1 = identical."),
    "tsne": ("t-SNE", "A method that places points on a 2-D map so close neighbours stay close. Good for looking, not for measuring distances."),
}

# ---------------------------------------------------------------- CSS
DARK = """
  --paper:#000000;--surface:#0d0d0e;--line:rgba(255,255,255,.09);--ink:#f5f5f7;--ink-2:#d2d2d7;--muted:#a1a1a6;
  --soft:#161617;--hover:#1c1c1e;--base:#48484a;--grid:rgba(255,255,255,.07);--on-ink:#000000;
  --glass:rgba(22,22,23,.62);--glass-line:rgba(255,255,255,.08);
  --shadow:0 1px 0 rgba(255,255,255,.04) inset,0 24px 64px -24px rgba(0,0,0,.8);
  --happy:#c98500;--sad:#9085e9;--calm:#199e70;--energetic:#e34948;
  --accent:#f5f5f7;--on-accent:#000000;--accent-soft:rgba(255,255,255,.06);--tuned:#f5f5f7;
  color-scheme:dark;"""

CSS = r"""
/*FONTS*/
:root{
  --paper:#fbfbfd;--surface:#ffffff;--line:#e5e5ea;--ink:#1d1d1f;--ink-2:#424245;--muted:#6e6e73;
  --soft:#f5f5f7;--hover:#f0f0f2;--base:#c7c7cc;--grid:#ececf0;--on-ink:#fbfbfd;
  --glass:rgba(251,251,253,.72);--glass-line:rgba(29,29,31,.08);
  --shadow:0 1px 1px rgba(0,0,0,.03),0 8px 24px -8px rgba(0,0,0,.08),0 24px 64px -24px rgba(0,0,0,.12);
  --happy:#eda100;--sad:#6250d6;--calm:#1baf7a;--energetic:#e34948;
  --accent:#1d1d1f;--on-accent:#fbfbfd;--accent-soft:rgba(29,29,31,.05);--tuned:#1d1d1f;
  --ease:cubic-bezier(.16,1,.3,1);
  --sans:"Geist",-apple-system,BlinkMacSystemFont,"SF Pro Display",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  --mono:"Geist Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --serif:"IBM Plex Serif",Georgia,"Times New Roman",serif;
  --top:56px;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){/*DARK*/} }
:root[data-theme="dark"]{/*DARK*/}

*,*::before,*::after{box-sizing:border-box}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%;scroll-padding-top:calc(var(--top) + 16px)}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}*,*::before,*::after{animation:none!important;transition:none!important}}
body{margin:0;background:var(--paper);color:var(--ink);font:400 17px/1.65 var(--sans);font-variant-numeric:tabular-nums;overflow-x:hidden;-webkit-font-smoothing:antialiased}
img{max-width:100%;display:block}
a{color:var(--ink);text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;text-decoration-color:var(--muted)}
a:hover{text-decoration-color:var(--ink)}
button{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--ink);outline-offset:2px;border-radius:2px}
code,pre,kbd,.mono{font-family:var(--mono)}
code{font-size:.86em;background:var(--soft);border:1px solid var(--line);border-radius:4px;padding:0 4px}
kbd{font-size:.82em;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--surface)}
abbr[title]{text-decoration:underline dotted;text-decoration-color:var(--muted);text-underline-offset:3px;cursor:help}
a.term{text-decoration:none}
a.term abbr{text-decoration:underline dotted;text-decoration-color:var(--muted)}
a.term:hover abbr{text-decoration-style:solid;text-decoration-color:var(--ink)}
.sr{position:absolute!important;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

.skip{position:absolute;left:16px;top:-48px;z-index:50;background:var(--ink);color:var(--on-ink);padding:8px 14px;border-radius:6px;text-decoration:none;font-weight:500}
.skip:focus{top:8px}

/* ---------- top bar ---------- */
.topbar{position:sticky;top:0;z-index:30;background:var(--paper);border-bottom:1px solid var(--line)}
.topbar-in{max-width:1240px;margin:0 auto;padding:0 32px;height:var(--top);display:flex;align-items:center;gap:16px}
.brand{font-weight:600;font-size:16px;letter-spacing:-.01em;text-decoration:none;white-space:nowrap}
.brand-sub{font:400 13px var(--mono);color:var(--muted);white-space:nowrap}
.top-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.status{display:inline-flex;align-items:center;gap:8px;font:500 13px var(--mono);color:var(--ink-2);padding:0 8px;min-height:32px;white-space:nowrap}
.status .dot{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--muted);flex:none}
.status.live .dot{background:var(--ink);border-color:var(--ink);box-shadow:0 0 0 0 var(--muted);animation:pulse 2.4s ease-out infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ink) 35%,transparent)}70%,100%{box-shadow:0 0 0 6px transparent}}
.icon-btn{min-width:36px;height:36px;border:1px solid var(--line);background:var(--surface);border-radius:6px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;gap:6px}
.icon-btn:hover{border-color:var(--muted)}
.icon-btn svg{width:16px;height:16px}

/* ---------- layout: one wide column, section links in the top bar ---------- */
.shell{max-width:1240px;margin:0 auto;padding:0 32px}
main{min-width:0;padding:0 0 96px}
.toc{min-width:0;overflow-x:auto;scrollbar-width:none}
.toc::-webkit-scrollbar{display:none}
.toc ol{list-style:none;margin:0;padding:0;display:flex;gap:2px}
.toc li{margin:0;max-width:none}
.topbar .brand-sub{display:none}
.toc{margin-left:8px}
.toc a{display:flex;align-items:center;height:var(--top);padding:0 9px;text-decoration:none;color:var(--muted);font-size:14px;white-space:nowrap;border-bottom:2px solid transparent}
.toc a .n{display:none}
.toc a:hover{color:var(--ink)}
.toc a[aria-current="true"]{color:var(--ink);border-bottom-color:var(--ink);font-weight:500}
@media (max-width:1100px){
  .topbar-in{flex-wrap:wrap;height:auto;padding-top:8px;gap:0 8px}
  .topbar-in .toc{order:3;flex-basis:100%;margin:0 -8px}
  .toc a{height:40px;font-size:14px}
  :root{--top:96px}
}
@media (max-width:900px){
  .topbar-in{padding:8px 16px 0}
  .brand-sub{display:none}
  .shell{padding:0 16px}
}
@media (max-width:380px){ .status{padding:0 2px} }

/* ---------- type ---------- */
section{padding-top:72px}
section + section{margin-top:64px;border-top:1px solid var(--line)}
.sec-head{display:flex;align-items:baseline;gap:16px;margin:0 0 20px}
.sec-num{font:500 13px var(--mono);color:var(--muted);min-width:22px}
h1{font:600 44px/1.1 var(--sans);letter-spacing:-.025em;margin:0 0 20px}
h2{font:600 32px/1.2 var(--sans);letter-spacing:-.02em;margin:0}
h3{font:600 21px/1.3 var(--sans);letter-spacing:-.01em;margin:56px 0 12px}
h4{font:600 17px/1.35 var(--sans);margin:0}
p,ul,ol{margin:0 0 16px}
p,li{max-width:68ch}
ul,ol{padding-left:22px}
li{margin:4px 0}
li::marker{color:var(--muted)}
strong{font-weight:600}
.lead{font-size:20px;line-height:1.55;color:var(--ink-2)}
.short{margin:0 0 28px;padding:0 0 0 18px;border-left:2px solid var(--ink)}
.short .lab{display:block;margin-bottom:2px}
.short p{font-size:19px;line-height:1.55;margin:0;color:var(--ink)}
.lab{font:500 12.5px/1.5 var(--mono);color:var(--muted)}
.muted{color:var(--muted)}
.small{font-size:15px}
.note{font-size:15px;color:var(--ink-2)}
@media (max-width:640px){
  section{padding-top:48px} section + section{margin-top:40px}
  h1{font-size:34px} h2{font-size:26px} h3{font-size:20px;margin-top:44px}
  .lead{font-size:18px} .short p{font-size:17.5px}
}

/* ---------- tags (only on model cards / pipeline) ---------- */
.tag{display:inline-block;font:500 12px/1.5 var(--mono);padding:1px 7px;border:1px solid var(--muted);border-radius:4px;color:var(--ink-2);white-space:nowrap;vertical-align:1px}
.tag.pre{border-style:dashed}
.tag.mine{border-color:var(--ink);color:var(--ink);background:var(--surface)}
.tag.algo{border-color:var(--line)}

/* ---------- hero ---------- */
.hero{padding-top:56px}
.hero .lab{margin-bottom:12px;display:block}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:40px 0 0}
.kpi{padding:18px 20px 18px 0}
.kpi + .kpi{padding-left:20px;border-left:1px solid var(--line)}
.kpi .v{font:600 28px/1.15 var(--sans);letter-spacing:-.02em;white-space:nowrap}
@media (max-width:1100px){.kpi .v{font-size:24px}}
.kpi .v small{font-size:15px;font-weight:500;color:var(--ink-2);letter-spacing:0}
.kpi .k{font-size:14.5px;color:var(--ink-2);margin-top:4px;line-height:1.4}
.kpi .s{font:400 12px/1.4 var(--mono);color:var(--muted);margin-top:6px}
@media (max-width:720px){
  .kpis{grid-template-columns:1fr 1fr}
  .kpi:nth-child(3){padding-left:0;border-left:0}
  .kpi:nth-child(n+3){border-top:1px solid var(--line)}
}

/* ---------- pipeline diagram ---------- */
.pipe{display:grid;grid-template-columns:minmax(0,.8fr) 28px minmax(0,1.6fr) 28px minmax(0,1.2fr);align-items:center;margin:32px 0 12px}
.pipe .stack{display:grid;gap:12px}
.node{border:1px solid var(--ink);background:var(--surface);border-radius:8px;padding:14px 16px}
.node.pre{border:1px dashed var(--muted)}
.node.in{border-color:var(--line);background:var(--soft)}
.node .lab{display:block}
.node b{display:block;font-weight:600;font-size:16px;line-height:1.35;margin:2px 0}
.node span.d{display:block;font-size:14px;color:var(--ink-2);line-height:1.45}
.arrow{display:flex;align-items:center;justify-content:center;color:var(--muted)}
.arrow svg{width:24px;height:12px}
.pipe-legend{display:flex;flex-wrap:wrap;gap:8px 24px;font-size:14px;color:var(--ink-2);margin:4px 0 0}
.pipe-legend span{display:inline-flex;align-items:center;gap:8px}
.pipe-legend i{display:inline-block;width:22px;height:14px;border-radius:3px;border:1px solid var(--ink)}
.pipe-legend i.pre{border:1px dashed var(--muted)}
@media (max-width:720px){
  .pipe{grid-template-columns:1fr;gap:0}
  .arrow{height:28px} .arrow svg{transform:rotate(90deg)}
}
.example{display:grid;grid-template-columns:150px minmax(0,1fr);gap:20px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:16px;margin:24px 0 0}
.example img{width:150px;height:150px;object-fit:cover;border-radius:4px}
.example dl{margin:0;display:grid;grid-template-columns:72px minmax(0,1fr);gap:6px 12px;font-size:15px}
.example dt{font:500 12.5px/1.9 var(--mono);color:var(--muted)}
.example dd{margin:0}
@media (max-width:560px){ .example{grid-template-columns:1fr} .example img{width:100%;height:180px} }

/* ---------- poem text ---------- */
.poem{font:400 18px/1.7 var(--serif);white-space:pre-line;margin:0;color:var(--ink)}
.poem.sm{font-size:16px;line-height:1.65}
.cap{font-style:normal;color:var(--ink-2)}
.chip{display:inline-flex;align-items:center;gap:7px;font-weight:500;font-size:15px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;flex:none}

/* ---------- studio ---------- */
.studio{border:1px solid var(--line);background:var(--surface);border-radius:8px;padding:20px;margin-top:8px}
.offline{border:1px solid var(--ink);border-radius:6px;padding:14px 16px;margin:0 0 16px;font-size:15px}
.offline p{margin:0 0 10px}
.drop{border:1px dashed var(--muted);border-radius:6px;padding:28px 16px;text-align:center;cursor:pointer;background:var(--paper);transition:background .15s,border-color .15s}
.drop:hover,.studio.over .drop{border-color:var(--ink);background:var(--hover)}
.studio.over .drop{border-style:solid}
.drop p{margin:0 auto;max-width:none}
.drop .big{font-weight:600;font-size:17px}
.drop .u{text-decoration:underline;text-underline-offset:3px}
.drop svg{width:22px;height:22px;margin:0 auto 8px;color:var(--ink-2)}
.picks{display:flex;align-items:center;flex-wrap:wrap;gap:8px 12px;margin:16px 0 0}
.picks .lab{margin-right:4px}
@media (max-width:600px){ .picks .lab{flex-basis:100%} }
.pick{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);background:var(--surface);border-radius:6px;padding:4px 10px 4px 4px;cursor:pointer;font-size:14.5px;min-height:40px}
.pick:hover{border-color:var(--muted)}
.pick img{width:30px;height:30px;object-fit:cover;border-radius:3px}
.log{display:grid;gap:16px;margin-top:20px}
.log:empty{display:none}
.result{border-top:1px solid var(--line);padding-top:20px;display:grid;grid-template-columns:168px minmax(0,1fr);gap:24px}
.result .thumb img{width:168px;height:168px;object-fit:cover;border-radius:4px;border:1px solid var(--line)}
.result .thumb .lab{margin-top:6px}
@media (max-width:600px){ .result{grid-template-columns:1fr} .result .thumb img{width:100%;height:auto;max-height:260px} }
.steps{list-style:none;padding:0;margin:0 0 4px;display:grid;gap:6px;font-size:15px}
.steps li{display:grid;grid-template-columns:20px minmax(0,1fr) auto;align-items:center;gap:10px;color:var(--muted);margin:0;max-width:none}
.steps li .m{width:12px;height:12px;border-radius:50%;border:1.5px solid var(--line);justify-self:center}
.steps li.active{color:var(--ink)}
.steps li.active .m{border-color:var(--ink);border-right-color:transparent;animation:spin .8s linear infinite}
.steps li.done{color:var(--ink-2)}
.steps li.done .m{background:var(--ink);border-color:var(--ink)}
.steps li.fail .m{border-color:var(--ink);background:repeating-linear-gradient(45deg,var(--ink) 0 2px,transparent 2px 4px)}
.steps .t{font:400 13px var(--mono);color:var(--muted)}
@keyframes spin{to{transform:rotate(360deg)}}
.elapsed{font:400 13px var(--mono);color:var(--muted);margin-top:8px}
.r-head{display:flex;align-items:baseline;flex-wrap:wrap;gap:6px 14px;margin-bottom:12px}
.r-head .chip{font-size:20px;font-weight:600}
.r-head .chip .sw{width:12px;height:12px}
.probs{display:grid;gap:4px;margin:0 0 16px;max-width:420px}
.prow{display:grid;grid-template-columns:84px minmax(0,1fr) 48px;align-items:center;gap:10px;font-size:14px}
.prow .pt{height:8px;background:var(--soft);border-radius:2px;overflow:hidden;border:1px solid var(--line)}
.prow .pb{display:block;height:100%}
.prow .pv{font:400 13px var(--mono);text-align:right;color:var(--ink-2)}
.result .cap{margin:0 0 14px;font-size:15.5px}
.result .poem{margin:0 0 18px}
.timing{display:flex;flex-wrap:wrap;gap:4px 18px;font:400 13px var(--mono);color:var(--muted);margin:0 0 14px}
.timing b{color:var(--ink);font-weight:500}
.actions{display:flex;flex-wrap:wrap;gap:8px}
.btn{min-height:36px;padding:0 14px;border-radius:6px;border:1px solid var(--ink);background:var(--ink);color:var(--on-ink);font-weight:500;font-size:14.5px;cursor:pointer}
.btn:hover{opacity:.88}
.btn.ghost{background:var(--surface);color:var(--ink);border-color:var(--line)}
.btn.ghost:hover{border-color:var(--muted);opacity:1}
.btn:disabled{opacity:.5;cursor:progress}
.err{border-top:1px solid var(--line);padding-top:16px;font-size:15px}
.err strong{display:block}

/* ---------- code ---------- */
.code{position:relative;border:1px solid var(--line);background:var(--soft);border-radius:6px;margin:0 0 16px}
.code pre{margin:0;padding:14px 16px;overflow-x:auto;font:13.5px/1.65 var(--mono);color:var(--ink)}
.code pre code{background:none;border:0;padding:0;font-size:inherit}
.code .copy{position:absolute;top:6px;right:6px;min-height:28px;min-width:56px;padding:0 10px;font:500 12px var(--mono);border:1px solid var(--line);background:var(--surface);border-radius:4px;cursor:pointer;color:var(--ink-2)}
.code .copy:hover{color:var(--ink);border-color:var(--muted)}
.code .cmt{color:var(--muted)}

/* ---------- tables ---------- */
.tbl-wrap{overflow-x:auto;margin:0 0 20px}
table{width:100%;border-collapse:collapse;font-size:15px}
caption{text-align:left;font:500 12.5px var(--mono);color:var(--muted);padding:0 0 8px}
th,td{text-align:left;padding:9px 12px 9px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:500;color:var(--ink-2);font-size:14px}
thead th{border-bottom-color:var(--muted)}
td.n,th.n{text-align:right;font-family:var(--mono);font-size:14px;white-space:nowrap}
tr.hl td{font-weight:600}
td code{white-space:nowrap}

/* ---------- figures ---------- */
figure.fig{margin:28px 0;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:20px}
.fig-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin:0 0 4px}
.fig-head h4{max-width:52ch}
.fig .sub{font-size:14.5px;color:var(--muted);margin:2px 0 0}
.fig .controls{display:flex;flex-wrap:wrap;gap:8px 16px;margin:14px 0 0}
.fig .legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px;color:var(--ink-2);margin:12px 0 0}
.fig .legend span{display:inline-flex;align-items:center;gap:7px}
.fig .plot{margin:12px 0 0;min-height:40px;position:relative}
.fig .plot svg{display:block;overflow:visible}
.fig figcaption{margin:16px 0 0;padding:14px 0 0;border-top:1px solid var(--line);font-size:15px;color:var(--ink-2);display:grid;gap:8px}
.fig figcaption p{margin:0}
.fig figcaption strong{color:var(--ink);font-weight:600}
.tbtn{flex:none;min-height:30px;padding:0 10px;font:500 12px var(--mono);border:1px solid var(--line);background:var(--surface);border-radius:4px;cursor:pointer;color:var(--ink-2)}
.tbtn:hover,.tbtn[aria-expanded="true"]{color:var(--ink);border-color:var(--muted)}
.tview{margin:16px 0 0}
.tview .tbl-wrap{margin:0}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--muted);stroke-width:1}
svg .tick{font:400 12px var(--mono);fill:var(--muted)}
svg .alab{font:400 12px var(--mono);fill:var(--muted)}
svg .cl{font:400 13.5px var(--sans);fill:var(--ink)}
svg .cl.b{font-weight:600}
svg .dl{font:500 12px var(--mono);fill:var(--ink)}
svg .ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
svg .reflab{font:400 12px var(--mono);fill:var(--ink-2)}
svg [tabindex]:focus{outline:none}
svg [tabindex]:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;padding:2px;gap:2px;background:var(--paper)}
.seg button{border:0;background:none;border-radius:4px;padding:0 10px;min-height:30px;font-size:14px;cursor:pointer;color:var(--ink-2)}
.seg button[aria-selected="true"]{background:var(--ink);color:var(--on-ink)}
.seg-l{display:inline-flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);flex-wrap:wrap}
#tip{position:fixed;z-index:60;pointer-events:none;display:none;max-width:260px;background:var(--surface);border:1px solid var(--muted);border-radius:6px;padding:8px 10px;font-size:13.5px;line-height:1.45;color:var(--ink);box-shadow:0 2px 10px rgba(0,0,0,.08)}
#tip .th{font-weight:600;margin-bottom:2px}
#tip .tr{display:flex;align-items:center;gap:6px}
#tip .tr b{margin-left:auto;padding-left:12px;font:500 12.5px var(--mono)}
#tip img{width:120px;height:120px;object-fit:cover;border-radius:3px;margin:0 0 6px}

/* confusion matrix */
.cm{border-collapse:separate;border-spacing:3px;width:auto;margin:0}
.cm th,.cm td{border:0;padding:0}
.cm td{width:64px;height:52px;text-align:center;vertical-align:middle;font:500 15px var(--mono);border-radius:3px}
.cm th{font:400 13px var(--sans);color:var(--ink-2);padding:0 8px;white-space:nowrap}
.cm thead th{text-align:center;padding:0 0 4px;vertical-align:bottom}
.cm tbody th{text-align:right;vertical-align:middle}
.cm-wrap{overflow-x:auto}
.cm-axis{font:400 12px var(--mono);color:var(--muted)}
@media (max-width:480px){ .cm td{width:48px;height:44px;font-size:14px} .cm th{font-size:12px;padding:0 4px} }

/* small diagrams */
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:16px 0 24px;font-size:14.5px}
.flow .b{border:1px solid var(--line);background:var(--surface);border-radius:6px;padding:6px 12px;line-height:1.35}
.flow .b small{display:block;font:400 12px var(--mono);color:var(--muted)}
.flow .b.mine{border-color:var(--ink)}
.flow .to{color:var(--muted);font-family:var(--mono)}
.folds{display:grid;gap:4px;margin:12px 0 8px;max-width:520px}
.folds .row{display:grid;grid-template-columns:56px repeat(5,1fr);gap:4px;align-items:center;font:400 12px var(--mono);color:var(--muted)}
.folds .row span.c{height:22px;border-radius:3px;background:var(--soft);border:1px solid var(--line)}
.folds .row span.c.v{background:var(--ink);border-color:var(--ink)}
.split{display:flex;height:28px;border-radius:4px;overflow:hidden;max-width:520px;margin:8px 0 4px;font:500 12px var(--mono)}
.split span{display:flex;align-items:center;padding:0 10px;white-space:nowrap}
.split .tr{background:var(--ink);color:var(--on-ink)}
.split .te{background:var(--soft);border:1px solid var(--line);color:var(--ink)}

/* model cards */
.cards{border-top:1px solid var(--line);margin:0 0 8px}
details.card{border-bottom:1px solid var(--line)}
details.card > summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:minmax(0,1fr) auto 20px;align-items:baseline;gap:4px 14px;padding:16px 0;border-radius:2px}
details.card > summary::-webkit-details-marker{display:none}
details.card > summary::after{content:"+";font:400 18px/1 var(--mono);color:var(--muted);grid-column:3;grid-row:1;text-align:right}
details.card[open] > summary::after{content:"\2212"}
details.card > summary:hover .t{text-decoration:underline;text-underline-offset:3px;text-decoration-color:var(--muted)}
summary .t{font-weight:600;font-size:17px;grid-column:1}
summary .tag{grid-column:2;grid-row:1}
summary .d{grid-column:1 / span 2;color:var(--ink-2);font-size:15px;line-height:1.5}
.card-body{padding:0 0 24px}
.card-body dl{display:grid;grid-template-columns:150px minmax(0,1fr);gap:10px 24px;margin:0;font-size:15.5px}
.card-body dt{font:500 12.5px/1.9 var(--mono);color:var(--muted)}
.card-body dd{margin:0;max-width:62ch}
.card-body dd ul{margin:0;padding-left:18px}
.formula{font:400 15px/1.6 var(--mono);background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:10px 14px;overflow-x:auto;white-space:nowrap}
@media (max-width:600px){ .card-body dl{grid-template-columns:1fr;gap:2px} .card-body dd{margin-bottom:10px} }
.lora-svg{width:100%;max-width:560px;height:auto;margin:8px 0}
.lora-svg text{font-family:var(--mono);font-size:12px;fill:var(--ink-2)}
.lora-svg .big{font-size:14px;fill:var(--ink);font-weight:500}

/* qa / glossary / generic details */
details.qa{border-bottom:1px solid var(--line)}
details.qa:first-of-type{border-top:1px solid var(--line)}
details.qa summary{cursor:pointer;padding:14px 28px 14px 0;font-weight:500;list-style:none;position:relative}
details.qa summary::-webkit-details-marker{display:none}
details.qa summary::after{content:"+";position:absolute;right:0;top:14px;font:400 18px/1.4 var(--mono);color:var(--muted)}
details.qa[open] summary::after{content:"\2212"}
details.qa .a{padding:0 0 16px}
details.qa .a p{margin:0 0 10px}
details.more{margin:16px 0 24px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
details.more > summary{cursor:pointer;padding:12px 16px;font-weight:500;font-size:15.5px}
details.more > div{padding:0 16px 16px}
.gloss{display:grid;grid-template-columns:200px minmax(0,1fr);gap:0;margin:0;border-top:1px solid var(--line)}
.gloss dt,.gloss dd{padding:12px 0;border-bottom:1px solid var(--line);margin:0}
.gloss dt{font-weight:600;padding-right:16px}
.gloss dfn{font-style:normal}
.gloss dd{color:var(--ink-2);font-size:16px}
.gloss :target{background:var(--hover)}
@media (max-width:600px){ .gloss{grid-template-columns:1fr} .gloss dt{border-bottom:0;padding-bottom:0} .gloss dd{padding-top:2px} }

/* grids of images / poems */
.mood-grid{display:grid;gap:18px;margin:16px 0 8px}
.mood-row{display:grid;grid-template-columns:110px minmax(0,1fr);gap:16px;align-items:center}
.mood-row .imgs{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}
.mood-row img{aspect-ratio:1;object-fit:cover;width:100%;height:auto;border-radius:3px}
@media (max-width:560px){ .mood-row{grid-template-columns:1fr;gap:6px} .mood-row .imgs{grid-template-columns:repeat(3,1fr)} .mood-row img:nth-child(n+4){display:none} }
.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:20px 0}
.gcard{border:1px solid var(--line);border-radius:8px;background:var(--surface);overflow:hidden}
.gcard img{width:100%;aspect-ratio:4/3;object-fit:cover}
.gcard .body{padding:16px}
.gcard .meta{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin:0 0 6px}
.gcard .cap{font-size:14.5px;margin:0 0 12px}
@media (max-width:640px){ .gallery{grid-template-columns:1fr} }
.compare{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;margin:16px 0 0}
.compare h5{font:500 12.5px var(--mono);color:var(--muted);margin:0 0 8px;display:flex;gap:8px;align-items:center}
.compare .col{border-top:2px solid var(--base);padding-top:12px}
.compare .col.tuned{border-top-color:var(--ink)}
.cmp-head{display:grid;grid-template-columns:120px minmax(0,1fr);gap:16px;align-items:center;margin:16px 0 0}
.cmp-head img{width:120px;height:120px;object-fit:cover;border-radius:4px}
@media (max-width:600px){ .compare{grid-template-columns:1fr} }
.examples{display:grid;gap:16px}
.ex{display:grid;grid-template-columns:96px minmax(0,1fr);gap:16px;border-top:1px solid var(--line);padding-top:16px}
.ex img{width:96px;height:96px;object-fit:cover;border-radius:4px}
.ex .lab{display:block;margin-bottom:6px}

/* tabs */
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin:16px 0 16px;overflow-x:auto}
.tabs [role=tab]{border:0;background:none;padding:10px 14px;min-height:40px;cursor:pointer;color:var(--muted);font-size:15px;font-weight:500;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
.tabs [role=tab]:hover{color:var(--ink)}
.tabs [role=tab][aria-selected="true"]{color:var(--ink);border-bottom-color:var(--ink)}

/* architecture */
.arch{display:grid;grid-template-columns:minmax(0,1fr) 28px minmax(0,2.2fr);align-items:stretch;margin:24px 0 8px}
.arch .arrow{align-self:center}
.arch .srv{border:1px solid var(--ink);border-radius:8px;padding:14px 16px;background:var(--surface)}
.arch .srv > b{display:block;margin-bottom:10px}
.arch .row3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.arch .node{padding:10px 12px}
.arch .node b{font-size:15px}
.arch .client{align-self:center}
@media (max-width:720px){ .arch{grid-template-columns:1fr} .arch .arrow{height:28px} .arch .arrow svg{transform:rotate(90deg)} .arch .row3{grid-template-columns:1fr} }

.refs{padding-left:0;list-style:none}
.refs li{padding:8px 0;border-bottom:1px solid var(--line);font-size:15.5px;max-width:none}
footer.foot{margin-top:72px;padding:24px 0 0;border-top:1px solid var(--line);font-size:14px;color:var(--muted);display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px}

/* ================= presentation layout =================
   Product first, then one screen per idea. Charts sit in pairs so I can point
   at both while explaining. Everything collapses to one column on phones. */
section + section{margin-top:48px}
.sec-head{flex-wrap:wrap}
.sec-head .kick{flex-basis:100%;margin:6px 0 0;font-size:17px;color:var(--ink-2);max-width:80ch}
.short{max-width:860px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:20px 0;align-items:start}
.grid2 > *{min-width:0}
.grid2 figure.fig{margin:0}
.grid2 .span2{grid-column:1 / -1}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:20px 0}
.grid3 > *{min-width:0}
@media (max-width:980px){ .grid2,.grid3{grid-template-columns:minmax(0,1fr)} }
.fig figcaption{font-size:14.5px}
.panel{border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:20px}
.panel h4{margin:0 0 10px}
.panel p:last-child,.panel ul:last-child,.panel .tbl-wrap:last-child{margin-bottom:0}

/* hero = the product */
.hero{padding-top:40px}
.hero h1{margin:4px 0 10px}
.hero .lead{margin:0 0 20px}
.hero-grid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:28px;align-items:start;border:0;background:none;padding:0;margin:0}
.hero-grid > *{min-width:0}
.hero-grid .drop{padding:44px 16px}
.hero-r{border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:20px;min-height:420px}
.hero-r .log{margin-top:0}
.hero-r .log:not(:empty) + .example{display:none}
.hero-r .result:first-child{border-top:0;padding-top:0}
.hero-r .example{display:grid;grid-template-columns:168px minmax(0,1fr);gap:24px;border:0;padding:0;margin:0;background:none}
.hero-r .example img{width:168px;height:168px}
@media (max-width:980px){ .hero-grid{grid-template-columns:minmax(0,1fr)} .hero-r{min-height:0} }
@media (max-width:600px){ .hero-r .example{grid-template-columns:1fr} .hero-r .example img{width:100%;height:220px} }

/* the six model cards */
.mcards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0 8px}
@media (max-width:1100px){ .mcards{grid-template-columns:repeat(2,minmax(0,1fr))} }
@media (max-width:700px){ .mcards{grid-template-columns:minmax(0,1fr)} }
.mcard{border:1px solid var(--ink);border-radius:8px;background:var(--surface);padding:18px 18px 16px;display:flex;flex-direction:column;gap:10px;min-width:0}
.mcard.pre{border:1px dashed var(--muted)}
.mcard .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.mcard .num{font:500 12.5px var(--mono);color:var(--muted)}
.mcard h3{margin:0;font-size:19px}
.mcard .id{font:400 12.5px/1.45 var(--mono);color:var(--muted);margin:-6px 0 0;overflow-wrap:anywhere}
.mcard .does{margin:0;font-size:15.5px;line-height:1.5}
.mcard dl{margin:0;display:grid;gap:8px;font-size:14.5px;line-height:1.5;border-top:1px solid var(--line);padding-top:10px}
.mcard dt{font:500 11.5px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.mcard dd{margin:-4px 0 0}
.mcard .big{font:600 26px/1.1 var(--sans);letter-spacing:-.02em}
.mcard .formula{font-size:13px;white-space:normal;padding:8px 10px}
.mcard code{font-size:.84em;overflow-wrap:anywhere}

/* five classifiers, compact */
.ccards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:16px 0 20px}
@media (max-width:1100px){ .ccards{grid-template-columns:repeat(3,minmax(0,1fr))} }
@media (max-width:640px){ .ccards{grid-template-columns:repeat(2,minmax(0,1fr))} }
.ccard{border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:14px;font-size:14px;line-height:1.45;min-width:0}
.ccard.win{border:2px solid var(--ink)}
.ccard h4{font-size:15.5px;margin:0 0 2px}
.ccard .acc{font:600 22px/1.2 var(--sans);margin:8px 0 0}
.ccard .acc small{font:400 12px var(--mono);color:var(--muted)}
.ccard p{margin:6px 0 0}
.ccard code{font-size:12px;overflow-wrap:anywhere}

/* numbered steps in a row */
.steps5{list-style:none;padding:0;margin:20px 0;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;counter-reset:s}
.steps5 li{margin:0;max-width:none;border-top:2px solid var(--ink);padding-top:10px;font-size:14.5px;line-height:1.5;counter-increment:s}
.steps5 li::before{content:counter(s,decimal-leading-zero);display:block;font:500 12px var(--mono);color:var(--muted);margin-bottom:4px}
.steps5 b{display:block;font-size:15.5px;margin-bottom:2px}
@media (max-width:980px){ .steps5{grid-template-columns:repeat(2,minmax(0,1fr))} }
@media (max-width:520px){ .steps5{grid-template-columns:minmax(0,1fr)} }
.verdict{border-top:2px solid var(--ink);padding-top:12px}
.verdict.neg{border-top-color:var(--base)}
.verdict h4{margin:0 0 6px}
.verdict p{font-size:15px;margin:0}
.verdict .big{font:600 26px/1.15 var(--sans);letter-spacing:-.02em;margin:0 0 6px}

/* samples in one row */
.gallery{grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
@media (max-width:1100px){ .gallery{grid-template-columns:repeat(2,minmax(0,1fr))} }
@media (max-width:600px){ .gallery{grid-template-columns:minmax(0,1fr)} }
.gcard .poem.sm{font-size:15px}

/* viva & more tabs */
#more [role=tabpanel]{padding-top:8px}
#more [role=tabpanel] > h3:first-child{margin-top:8px}

/* ================= accent: emerald = "mine" (trained / fine-tuned / chosen) =================
   One colour with one meaning, after the teacher / professor / classmate / designer review. */
.lab{font-size:13px}
section{padding-top:96px}
h2{font-size:40px;letter-spacing:-.025em}
@media (max-width:640px){ h2{font-size:28px} section{padding-top:56px} }
.sec-num{color:var(--accent);font-size:14px}
.toc a[aria-current="true"]{border-bottom-color:var(--accent)}
.status.live .dot{background:var(--accent);border-color:var(--accent)}
.short{border-left-color:var(--accent)}
.tabs [role=tab][aria-selected="true"]{border-bottom-color:var(--accent)}
.seg button[aria-selected="true"]{background:var(--accent);color:var(--on-accent)}
.btn{background:var(--accent);border-color:var(--accent);color:var(--on-accent);font-weight:600}
.tag.mine{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.node:not(.pre):not(.in),.flow .b.mine,.arch .srv{border-color:var(--accent)}
.pipe-legend i{border-color:var(--accent)}

/* hero */
.kpi .v{font-size:44px;color:var(--accent);line-height:1.05}
.kpi .v small{color:var(--muted)}
@media (max-width:1100px){.kpi .v{font-size:34px}}
.drop .pickbtn{display:inline-flex;align-items:center;min-height:40px;padding:0 18px;margin:12px 0 4px;border-radius:6px;background:var(--accent);color:var(--on-accent);font-weight:600;font-size:15px}
.drop:hover,.studio.over .drop{border-color:var(--accent);background:var(--accent-soft)}
.drop svg{color:var(--accent);width:28px;height:28px}
.picks .lab{flex-basis:100%}
.hero-r{border-left:4px solid var(--hero-mood,var(--line))}
.hero-r .poem{font-size:21px;line-height:1.65}
.r-head .chip{font-size:22px}

/* cards */
.mcard:not(.pre){border-color:var(--line);border-top:3px solid var(--accent)}
.mcard.pre{border-top:3px dashed var(--muted)}
.mcard .big{color:var(--accent)}
.ccard.win{border-color:var(--accent);background:var(--accent-soft)}
.ccard.win .acc,.ccard.win h4{color:var(--accent)}
.ccard:not(.win){opacity:.82}
.verdict{border-top-color:var(--accent)}
.verdict .big{font-size:36px;color:var(--accent)}
.verdict.neg{border-top-color:var(--energetic)}
.verdict.neg .big{color:var(--energetic)}
.gcard{border-top:4px solid var(--gm,var(--line))}
.compare .col.tuned{border-top-color:var(--accent)}
.panel,figure.fig,.mcard,.ccard,.hero-r{box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}

/* ================= portfolio skin (matches my portfolio site) =================
   Strict monochrome, Apple greys, true black in dark. Geist type, big tight
   headlines, mono uppercase labels, floating glass nav, rounded soft cards,
   pill buttons. Colour is kept only for data: the four moods and red = got worse. */
body{font-size:16.5px;letter-spacing:-.011em;font-feature-settings:"ss01","cv11";text-rendering:optimizeLegibility}
::selection{background:var(--ink);color:var(--paper)}
html{scrollbar-width:thin;scrollbar-color:var(--base) transparent}
:focus-visible{outline:3px solid color-mix(in srgb,var(--ink) 45%,transparent);outline-offset:2px;border-radius:8px}

/* floating glass pill nav */
:root{--top:76px}
.topbar{position:sticky;top:12px;background:none;border:0;padding:0 16px;height:0;overflow:visible}
.topbar-in{max-width:1180px;height:52px;padding:0 6px 0 22px;border-radius:999px;background:var(--glass);border:1px solid var(--glass-line);
  -webkit-backdrop-filter:saturate(180%) blur(20px);backdrop-filter:saturate(180%) blur(20px);box-shadow:var(--shadow);gap:8px}
.brand{font-size:15px;letter-spacing:-.02em}
.brand::after{content:".";color:var(--muted)}
.toc{margin-left:4px}
.toc ol{gap:0}
.toc a{height:34px;padding:0 11px;border:0;border-radius:999px;font-size:13px;color:var(--muted);transition:color .3s var(--ease),background-color .3s var(--ease)}
.toc a:hover{background:var(--hover);color:var(--ink)}
.toc a[aria-current="true"]{background:var(--hover);color:var(--ink);font-weight:500;border:0}
.status{font:500 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.status.live .dot,.status .dot{width:6px;height:6px;border:0;background:var(--muted)}
.status.live .dot{background:var(--ink)}
.icon-btn{width:38px;height:38px;border-radius:999px;border:0;background:transparent}
.icon-btn:hover{background:var(--hover)}
main{padding-top:52px}
@media (max-width:1100px){
  :root{--top:116px}
  .topbar-in{border-radius:24px;padding:6px 6px 4px 18px;height:auto}
  .topbar-in .toc{margin:2px -4px 0}
  .toc a{height:34px}
  main{padding-top:96px}
}

/* section masthead: 02 — LABEL on a hairline, then a big display title */
section{padding-top:120px}
section + section{margin-top:0;border-top:0}
.sec-head{display:block;margin:0 0 32px}
.eyebrow{display:flex;align-items:center;gap:12px;margin:0 0 22px;padding-top:16px;border-top:1px solid var(--line);
  font:500 11px var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--muted);max-width:none}
.eyebrow i{display:block;width:24px;height:1px;background:var(--line)}
.sec-num{color:var(--ink);font:500 11px var(--mono);min-width:0}
h1,h2{font-weight:600;letter-spacing:-.045em;line-height:.98}
h2{font-size:clamp(30px,3.6vw,48px);max-width:22ch;text-wrap:balance}
h2::after{content:".";color:var(--muted)}
h3{letter-spacing:-.025em}
.sec-head .kick{margin:16px 0 0;font-size:17px;line-height:1.6;color:var(--muted);max-width:60ch;text-wrap:pretty}
.lab{font:500 11px/1.6 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.lead{color:var(--muted)}
@media (max-width:640px){ section{padding-top:80px} h2{font-size:32px} }

/* hero */
.hero{padding-top:56px}
.hero-l > .lab:first-child{display:flex;align-items:center;gap:12px}
.hero-l > .lab:first-child::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--ink);box-shadow:0 0 0 0 color-mix(in srgb,var(--ink) 40%,transparent);animation:ping 2.4s var(--ease) infinite}
@keyframes ping{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ink) 40%,transparent)}80%,100%{box-shadow:0 0 0 8px transparent}}
.hero h1{font-size:clamp(44px,5.6vw,80px);margin:22px 0 18px}
.hero h1 .dim{color:var(--muted)}
.hero-grid.studio{border:0;background:none;box-shadow:none;border-radius:0;padding:0}
.hero-r .example{border:0;box-shadow:none;background:none}
#dropHelp,.picks .lab,.result .lab,.r-head .lab{text-transform:none;letter-spacing:0;font-size:12.5px}
.hero .lead{font-size:18px;line-height:1.6;max-width:44ch}
.kpis{border:0;border-top:1px solid var(--line);margin:56px 0 0}
.kpi{padding:22px 24px 0 0}
.kpi + .kpi{border-left:0;padding-left:0}
.kpi .v{font-size:clamp(30px,3.4vw,44px);font-weight:600;letter-spacing:-.045em;color:var(--ink)}
.kpi .v small{color:var(--muted);font-weight:400}
.kpi .k{color:var(--muted);font-size:14px}
.kpi .s{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase}
@media (max-width:720px){ .kpi:nth-child(n+3){border-top:0} .kpi{padding-top:18px} }

/* cards: rounded, soft, one hairline */
.panel,figure.fig,.mcard,.ccard,.hero-r,.gcard,details.more,.offline{border-radius:20px;border:1px solid var(--line);background:var(--surface);box-shadow:var(--shadow)}
figure.fig,.panel{padding:24px}
.hero-r{padding:24px;border-left:1px solid var(--line)}
.mcard,.mcard.pre,.mcard:not(.pre){border:1px solid var(--line);border-top:1px solid var(--line);padding:22px}
.mcard.pre{border-style:dashed}
.mcard .big,.ccard.win .acc,.ccard.win h4{color:var(--ink)}
.mcard h3{font-size:20px;letter-spacing:-.03em}
.mcard dt{font-size:10.5px;letter-spacing:.14em}
.ccard{padding:18px;opacity:1}
.ccard:not(.win){opacity:1;color:var(--muted)}
.ccard:not(.win) h4,.ccard:not(.win) .acc{color:var(--ink-2)}
.ccard.win{border:1px solid var(--ink);background:var(--surface)}
.ccard .acc{font-size:26px;letter-spacing:-.04em}
.gcard{border-top:1px solid var(--line);overflow:hidden}
.gcard img{transition:transform .9s var(--ease)}
.gcard:hover img{transform:scale(1.03)}
.node,.flow .b,.arch .srv{border-radius:14px}
.formula,.code{border-radius:12px}
code{border-radius:6px}
.tag,.tag.mine{border-radius:999px;padding:1px 9px;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase}
.tag.mine{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.verdict{border-top:1px solid var(--ink)}
.verdict .big{font-size:clamp(30px,3vw,40px);letter-spacing:-.045em;color:var(--ink)}
.verdict.neg{border-top-color:var(--energetic)}
.verdict.neg .big{color:var(--energetic)}
.steps5 li{border-top:1px solid var(--ink)}

/* studio */
.drop{border-radius:20px;border:1px dashed color-mix(in srgb,var(--ink) 22%,transparent);background:transparent;transition:background-color .5s var(--ease),border-color .5s var(--ease)}
.drop:hover,.studio.over .drop{border-color:var(--ink);background:var(--accent-soft)}
.drop svg{color:var(--ink)}
.drop .pickbtn,.btn{border-radius:999px;min-height:44px;padding:0 22px;background:var(--ink);border:0;color:var(--paper);font-weight:500;font-size:14px;transition:transform .5s var(--ease),opacity .3s}
.btn:active,.drop:active .pickbtn{transform:scale(.97)}
.btn.ghost{background:transparent;border:1px solid var(--line);color:var(--ink)}
.btn.ghost:hover{background:var(--hover)}
.pick{border-radius:999px;padding:4px 14px 4px 4px;transition:background-color .3s var(--ease)}
.pick:hover{background:var(--hover);border-color:var(--line)}
.pick img{border-radius:999px}
.result .thumb img,.hero-r .example img,.cmp-head img,.ex img,.example img{border-radius:14px}
.prow .pt{border-radius:999px;border:0;background:var(--soft)}
.hero-r .poem{font-size:20px}

/* tabs, toggles, small buttons as pills */
.tabs{border-bottom:0;gap:4px}
.tabs [role=tab]{border:0;border-radius:999px;margin:0;padding:8px 16px;min-height:36px;font-size:14px}
.tabs [role=tab]:hover{background:var(--hover)}
.tabs [role=tab][aria-selected="true"]{background:var(--ink);color:var(--paper);border:0}
.seg{border-radius:999px;padding:3px}
.seg button{border-radius:999px}
.seg button[aria-selected="true"]{background:var(--ink);color:var(--paper)}
.tbtn{border-radius:999px;padding:0 12px;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase}
.tbtn:hover,.tbtn[aria-expanded="true"]{background:var(--hover)}
.code .copy{border-radius:999px}
#tip{border-radius:12px;border-color:var(--line);box-shadow:var(--shadow);background:var(--surface)}
.status.live .dot{box-shadow:none}
footer.foot{font:500 11px var(--mono);letter-spacing:.14em;text-transform:uppercase}
"""

# ---------------------------------------------------------------- HTML
# {{KEY}} = number from data.json (see facts()), [[label|id]] = glossary term.
ARROW = '<div class="arrow" aria-hidden="true"><svg viewBox="0 0 24 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 6h21m-5-4 4 4-4 4" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image to Poem</title>
<meta name="description" content="Student ML project: a photo goes in, an 8-line poem comes out. BLIP caption, CLIP features with a scikit-learn mood classifier I trained, and a LoRA fine-tune of Qwen2.5-0.5B-Instruct.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23000'/%3E%3Cpath d='M9 11h14M9 16h10M9 21h12' stroke='%23f5f5f7' stroke-width='2.2' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preload" href="fonts/geist-400.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="fonts/geist-600.woff2" as="font" type="font/woff2" crossorigin>
<script>try{var t=localStorage.getItem('theme');if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t)}catch(e){}</script>
<style>/*CSS*/</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <div class="topbar-in">
    <a class="brand" href="#try">Image to Poem</a>
    <span class="brand-sub">ML mini-project</span>
    <nav class="toc" id="toc" aria-label="Sections"><ol>{{TOC}}</ol></nav>
    <div class="top-right">
      <span class="status" id="status" role="status" aria-live="polite" title="Checking the demo server"><i class="dot" aria-hidden="true"></i><span class="st">Checking</span></span>
      <button class="icon-btn" id="theme" type="button" aria-label="Switch to dark theme">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><circle cx="8" cy="8" r="6"/><path d="M8 2a6 6 0 0 0 0 12z" fill="currentColor"/></svg>
      </button>
    </div>
  </div>
</header>

<div class="shell">
<main id="main" tabindex="-1">

<!-- 01 · the product -->
<section id="try" class="hero" aria-labelledby="h-try">
  <div class="hero-grid studio" id="studio">
    <div class="hero-l">
      <span class="lab">College ML mini-project · runs offline on a laptop</span>
      <h1 id="h-try">Image to Poem<span class="dim">.</span></h1>
      <p class="lead">Drop in a photo. The app describes it, decides its mood and writes an eight-line poem about it, in about {{SEC_PER_IMAGE}} seconds.</p>
      <div class="offline" id="offline" hidden>
        <p><strong>The demo server is not running.</strong> The rest of this page still works. To write poems, run this in the project folder:</p>
        <div class="code"><button class="copy" type="button">Copy</button><pre><code>HF_HUB_OFFLINE=1 venv/bin/python app.py</code></pre></div>
        <p class="small muted" style="margin:0">Loading takes about half a minute. The dot in the top bar turns to Live on its own.</p>
      </div>
      <div class="drop" id="drop" role="button" tabindex="0" aria-describedby="dropHelp">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m4 18 5-5 4 4 3-3 4 4" stroke-linejoin="round"/></svg>
        <p class="big">Drop a photo here</p>
        <span class="pickbtn">Choose a photo</span>
        <p id="dropHelp" class="lab" style="margin-top:6px">JPG, PNG or WebP · or paste with <kbd>Ctrl</kbd>/<kbd>⌘</kbd> <kbd>V</kbd></p>
      </div>
      <input type="file" id="file" accept="image/jpeg,image/png,image/webp,image/*" class="sr" tabindex="-1" aria-hidden="true">
      <div class="picks" id="picks"><span class="lab">Or try a sample:</span></div>
      <p class="note" style="margin-top:14px">Nothing goes to the internet. The photo goes to <code>app.py</code> on this computer and is not saved.</p>
    </div>
    <div class="hero-r">
      <div class="log" id="log" aria-live="polite"></div>
      <div class="example" id="example" hidden></div>
    </div>
  </div>

  <div class="kpis" aria-label="Key numbers">
    <div class="kpi"><div class="v">{{N_IMAGES}}</div><div class="k">photos I collected and labelled</div><div class="s">4 moods, ~60 each</div></div>
    <div class="kpi"><div class="v">{{BEST_CV}}</div><div class="k">mood accuracy ({{BEST}})</div><div class="s">5-fold CV, ± {{BEST_STD}}</div></div>
    <div class="kpi"><div class="v">{{LINES_B}}<small> → </small>{{LINES_T}}</div><div class="k">poems with exactly 8 lines</div><div class="s">base → my fine-tune</div></div>
    <div class="kpi" id="kpi-sec"><div class="v">~{{SEC_PER_IMAGE}}<small> s</small></div><div class="k">per photo on a laptop CPU</div><div class="s">caption + mood + poem</div></div>
  </div>
</section>

<!-- 02 · the six models -->
<section id="models" aria-labelledby="h-models">
  <div class="sec-head"><span class="sec-num">02</span><h2 id="h-models">How it works: 6 models</h2>
    <p class="kick">Three steps, six parts. Solid border = I trained or wrote it. Dashed border = pretrained, used as-is.</p></div>
  <div class="pipe" role="img" aria-label="Pipeline: the image goes to BLIP for a caption and to CLIP plus my classifier for a mood; both go to the poem writer with my LoRA adapter.">
    <div class="node in"><span class="lab">Input</span><b>Photo</b><span class="d">JPG, PNG or WebP</span></div>
    """ + ARROW + r"""
    <div class="stack">
      <div class="node pre"><span class="lab">Step 1 · caption · model 1</span><b>BLIP</b><span class="d">“the sun is setting over the ocean”</span></div>
      <div class="node"><span class="lab">Step 2 · mood · models 2 + 3</span><b>CLIP → my mood classifier</b><span class="d">calm, 100 %</span></div>
    </div>
    """ + ARROW + r"""
    <div class="node"><span class="lab">Step 3 · poem · models 4 + 5 (+ 6 to judge)</span><b>Qwen + my LoRA adapter</b><span class="d">8 lines, checked by my poem scorer</span></div>
  </div>
  <div id="cards" class="mcards"></div>
  <p class="note">I did not train any large model from scratch. Training a captioner or a language model needs millions of examples and many GPUs. Reusing them is called <a class="term" href="#g-transfer"><abbr title="Reusing what a model learned on a big task as the starting point for a small one.">transfer learning</abbr></a>.</p>
</section>

<!-- 03 · data -->
<section id="dataset" aria-labelledby="h-data">
  <div class="sec-head"><span class="sec-num">03</span><h2 id="h-data">Mood data: {{N_IMAGES}} photos</h2>
    <p class="kick">Creative Commons photos from Openverse, five search phrases per mood, every folder cleaned by hand. The folder name is the label.</p></div>
  <div class="grid2">
    <div data-fig="counts"></div>
    <div class="panel"><h4>What the photos look like</h4><div class="mood-grid" id="moodGrid" style="margin:0"></div></div>
  </div>
  <div class="grid2">
    <div class="panel"><h4>Search phrases</h4><div class="tbl-wrap"><table><thead><tr><th>Mood</th><th>Phrases</th></tr></thead><tbody id="queries"></tbody></table></div></div>
    <div class="panel"><h4>How I split it</h4>
      <p class="small">Stratified, so every part keeps the same share of each mood. <code>random_state=42</code>, so results repeat.</p>
      <div class="split" aria-hidden="true"><span class="tr" style="flex:{{TRAIN}}">train · {{TRAIN}}</span><span class="te" style="flex:{{TEST}}">test · {{TEST}}</span></div>
      <p class="small" style="margin-top:14px">Inside the {{TRAIN}} training photos: 5-fold <a class="term" href="#g-cv"><abbr title="Train on 4 parts, score on the 5th, repeat 5 times, average.">cross-validation</abbr></a>. Solid block = the fold being scored.</p>
      <div class="folds" role="img" aria-label="Five rounds of cross-validation; in each round a different fold is used for scoring.">
        <div class="row"><span>round 1</span><span class="c v"></span><span class="c"></span><span class="c"></span><span class="c"></span><span class="c"></span></div>
        <div class="row"><span>round 2</span><span class="c"></span><span class="c v"></span><span class="c"></span><span class="c"></span><span class="c"></span></div>
        <div class="row"><span>round 3</span><span class="c"></span><span class="c"></span><span class="c v"></span><span class="c"></span><span class="c"></span></div>
        <div class="row"><span>round 4</span><span class="c"></span><span class="c"></span><span class="c"></span><span class="c v"></span><span class="c"></span></div>
        <div class="row"><span>round 5</span><span class="c"></span><span class="c"></span><span class="c"></span><span class="c"></span><span class="c v"></span></div>
      </div>
      <p class="small" style="margin:12px 0 0">Poem model: its own split by photo, {{P_TRAIN}} train pairs · {{P_VAL}} validation · {{P_TEST}} test photos.</p>
    </div>
  </div>
</section>

<!-- 04 · mood results -->
<section id="mood" aria-labelledby="h-mood">
  <div class="sec-head"><span class="sec-num">04</span><h2 id="h-mood">Mood classifier: 5 models compared</h2>
    <p class="kick">Same features, same pipeline, same folds. {{BEST}} wins on cross-validation with {{BEST_CV}}; Logistic Regression is almost tied at {{LR_CV}}.</p></div>
  <div class="flow" aria-label="Mood pipeline">
    <span class="b">CLIP vector<small>512 numbers</small></span><span class="to" aria-hidden="true">→</span>
    <span class="b mine">StandardScaler<small>mean 0, std 1</small></span><span class="to" aria-hidden="true">→</span>
    <span class="b mine">PCA<small>90 % variance → {{N90}}</small></span><span class="to" aria-hidden="true">→</span>
    <span class="b mine">one of 5 classifiers<small>below</small></span><span class="to" aria-hidden="true">→</span>
    <span class="b">mood<small>+ probabilities</small></span>
  </div>
  <div id="ccards" class="ccards"></div>
  <div class="grid2">
    <div data-fig="compare"></div>
    <div data-fig="confusion"></div>
    <div data-fig="metrics"></div>
    <div data-fig="pca"></div>
  </div>
  <p class="note">Scaler and PCA sit inside one scikit-learn <code>Pipeline</code>, so in every fold they are fitted on the training part only. No <a class="term" href="#g-leakage"><abbr title="Information from test data sneaking into training.">data leakage</abbr></a>. I choose by CV mean (5 scores) rather than test accuracy (one small set of {{TEST}} photos). The final model is refit on all {{N_IMAGES}} photos and saved for the app.</p>
</section>

<!-- 05 · clustering -->
<section id="unsupervised" aria-labelledby="h-unsup">
  <div class="sec-head"><span class="sec-num">05</span><h2 id="h-unsup">Without labels: do moods group by themselves?</h2>
    <p class="kick">Only weakly. K-Means with k = 4 matches my moods at an <a class="term" href="#g-ari"><abbr title="Agreement between two groupings: 0 = chance, 1 = identical.">ARI</abbr></a> of {{ARI}} (0 = chance, 1 = perfect) and the clusters overlap (silhouette {{SIL4}}). CLIP groups photos mostly by content, so labels are needed.</p></div>
  <div class="grid2">
    <div data-fig="elbow"></div>
    <div class="span2"><div data-fig="map"></div></div>
  </div>
</section>

<!-- 06 · poem training -->
<section id="poet-training" aria-labelledby="h-pt">
  <div class="sec-head"><span class="sec-num">06</span><h2 id="h-pt">Poem model: building data and fine-tuning</h2>
    <p class="kick">No image-to-poem dataset existed, so I built one, then trained a LoRA adapter on it in {{TRAIN_MIN}} minutes on my laptop's GPU.</p></div>
  <ol class="steps5">
    <li><b>Caption + mood</b>BLIP caption and the true mood label for each photo.</li>
    <li><b>Split by photo</b>80 / 10 / 10. Test photos never get poems.</li>
    <li><b>8 candidates</b>Base Qwen with a long, strict “teacher” prompt writes 8 poems per photo.</li>
    <li><b>Keep the best</b>My scorer rates them; keep 8-line poems scoring ≥ 0.45, up to 2 per photo.</li>
    <li><b>Short prompt</b>Train on the short everyday prompt, so the model learns the rules by itself.</li>
  </ol>
  <div class="code"><pre><code>Write a short 8-line poem about: {caption}. Mood: {mood}. Use vivid imagery, simple words, and a gentle rhyme.</code></pre></div>
  <div class="grid2">
    <div data-fig="scores"></div>
    <div data-fig="loss"></div>
  </div>
  <div class="grid2">
    <details class="more" style="margin:0"><summary>Training settings (from train_info.json)</summary><div><div class="tbl-wrap"><table><tbody id="trainInfo"></tbody></table></div>
      <p class="small" style="margin:0">Loss = cross-entropy on poem tokens only (prompt tokens get label −100). The adapter is saved at the last step ({{STEPS}}), after the best validation step ({{VAL_BEST_STEP}}).</p></div></details>
    <details class="more" style="margin:0"><summary>Show three training pairs</summary><div class="examples" id="examples"></div></details>
  </div>
</section>

<!-- 07 · poem results -->
<section id="poet-eval" aria-labelledby="h-pe">
  <div class="sec-head"><span class="sec-num">07</span><h2 id="h-pe">Poem model: base vs my fine-tune</h2>
    <p class="kick">{{P_TEST}} unseen test photos, 3 poems each ({{N_POEMS}} per model), same short prompt. Raw output is scored, with no trimming.</p></div>
  <div class="grid3">
    <div class="verdict"><p class="big">{{LINES_B}} → {{LINES_T}}</p><h4>Learned the format</h4><p>Almost every poem now has exactly 8 lines, and writing is about 42 % faster because it stops on time.</p></div>
    <div class="verdict neg"><p class="big">{{RHYME_B}} → {{RHYME_T}}</p><h4>Did not learn rhyme</h4><p>Only about 8 % of my training poems rhymed. A model cannot learn what is missing from its data.</p></div>
    <div class="verdict neg"><p class="big">{{KW_B}} → {{KW_T}}</p><h4>Fewer caption words</h4><p>Shorter poems repeat fewer caption words. CLIP match stayed the same ({{CLIP_B}} → {{CLIP_T}}).</p></div>
  </div>
  <div class="grid2">
    <div data-fig="before"></div>
    <div data-fig="lines"></div>
    <div data-fig="clip"></div>
    <div class="panel"><h4>Same photo, both models</h4>
      <div class="tabs" role="tablist" aria-label="Test images" id="cmpTabs" style="margin-top:0"></div>
      <div id="cmpPanels"></div></div>
  </div>
</section>

<!-- 08 · samples -->
<section id="gallery" aria-labelledby="h-gallery">
  <div class="sec-head"><span class="sec-num">08</span><h2 id="h-gallery">One photo per mood</h2>
    <p class="kick">Run end to end through the app with the fine-tuned model. Nothing edited. Timings from my laptop CPU.</p></div>
  <div class="gallery" id="samples"></div>
</section>

<!-- 09 · viva & more -->
<section id="more" aria-labelledby="h-more">
  <div class="sec-head"><span class="sec-num">09</span><h2 id="h-more">For the viva</h2>
    <p class="kick">Questions I expect, how the app is built, the syllabus map, honest limits, a glossary and how to run it.</p></div>
  <div class="tabs" role="tablist" aria-label="More" id="moreTabs">
    <button type="button" role="tab" id="mt-viva" aria-controls="mp-viva" aria-selected="true">Viva Q&amp;A</button>
    <button type="button" role="tab" id="mt-app" aria-controls="mp-app" aria-selected="false">The app</button>
    <button type="button" role="tab" id="mt-syl" aria-controls="mp-syl" aria-selected="false">Syllabus map</button>
    <button type="button" role="tab" id="mt-lim" aria-controls="mp-lim" aria-selected="false">Limits</button>
    <button type="button" role="tab" id="mt-gloss" aria-controls="mp-gloss" aria-selected="false">Glossary</button>
    <button type="button" role="tab" id="mt-run" aria-controls="mp-run" aria-selected="false">Run it</button>
  </div>

  <div role="tabpanel" id="mp-viva" aria-labelledby="mt-viva" tabindex="0"><div id="qa"></div></div>

  <div role="tabpanel" id="mp-app" aria-labelledby="mt-app" tabindex="0" hidden>
    <div class="arch" role="img" aria-label="Architecture: the browser calls app.py, a FastAPI server on port 7860, which holds BLIP, CLIP with my classifier, and the fine-tuned poem model.">
      <div class="node in client"><span class="lab">Browser</span><b>This page</b><span class="d">sends the photo with <code>fetch</code></span></div>
      """ + ARROW + r"""
      <div class="srv"><b>app.py · FastAPI on 127.0.0.1:7860</b>
        <div class="row3">
          <div class="node pre"><span class="lab">1</span><b>BLIP</b><span class="d">caption</span></div>
          <div class="node"><span class="lab">2</span><b>CLIP → scaler → PCA → {{BEST}}</b><span class="d">mood + probabilities</span></div>
          <div class="node"><span class="lab">3</span><b>Qwen + LoRA (merged)</b><span class="d">poem</span></div>
        </div>
      </div>
    </div>
    <div class="grid2">
      <div class="panel"><h4>Endpoints</h4><div class="tbl-wrap"><table>
        <thead><tr><th>Endpoint</th><th>Returns</th></tr></thead>
        <tbody>
          <tr><td><code>POST /api/poem</code></td><td>caption, mood, confidence, probs, poem, timing</td></tr>
          <tr><td><code>POST /api/regenerate</code></td><td>a new poem for the same caption and mood</td></tr>
          <tr><td><code>GET /api/health</code></td><td><code>{"ok": true}</code>, used by the Live dot</td></tr>
          <tr><td><code>/gradio</code></td><td>a plain Gradio page for the same pipeline</td></tr>
          <tr><td><code>/</code></td><td>this website</td></tr>
        </tbody></table></div>
        <ul class="small" style="margin:12px 0 0">
          <li>Models load once at start (~3 GB RAM).</li>
          <li>One request at a time (a lock), so timings stay honest.</li>
          <li>LoRA is merged into the base weights, so it costs no extra time.</li>
        </ul></div>
      <div data-fig="timing"></div>
    </div>
  </div>

  <div role="tabpanel" id="mp-syl" aria-labelledby="mt-syl" tabindex="0" hidden>
    <div class="tbl-wrap"><table>
      <thead><tr><th>Course topic</th><th>What I did</th><th>File</th></tr></thead>
      <tbody>
        <tr><td>Data collection</td><td>CC photos, 5 phrases per mood, cleaned by hand</td><td><code>src/build_dataset.py</code></td></tr>
        <tr><td>Feature extraction</td><td>Frozen CLIP: 512 numbers per photo (transfer learning)</td><td><code>src/features.py</code></td></tr>
        <tr><td>Scaling</td><td>StandardScaler inside the pipeline</td><td><code>src/train_mood.py</code></td></tr>
        <tr><td>Dimensionality reduction</td><td>PCA, 90 % variance (512 → {{N90}}); t-SNE map</td><td><code>src/train_mood.py</code></td></tr>
        <tr><td>Classification</td><td>Logistic Regression, KNN, Decision Tree, Random Forest, MLP</td><td><code>src/train_mood.py</code></td></tr>
        <tr><td>Model selection</td><td>Stratified 80/20 split, 5-fold CV</td><td><code>src/train_mood.py</code></td></tr>
        <tr><td>Metrics</td><td>Accuracy, precision, recall, F1, confusion matrix</td><td><code>src/evaluate.py</code></td></tr>
        <tr><td>Clustering</td><td>K-Means, elbow, silhouette, ARI</td><td><code>src/evaluate.py</code></td></tr>
        <tr><td>Fine-tuning (extension)</td><td>LoRA fine-tune of a pretrained language model</td><td><code>src/train_poet.py</code></td></tr>
        <tr><td>Deployment</td><td>FastAPI server, website, Gradio</td><td><code>app.py</code></td></tr>
      </tbody>
    </table></div>
  </div>

  <div role="tabpanel" id="mp-lim" aria-labelledby="mt-lim" tabindex="0" hidden>
    <div class="grid2">
      <div class="panel"><h4>Limits</h4><ul class="small">
        <li><strong>Small, one labeller.</strong> {{N_IMAGES}} photos labelled by me alone; ± 5 points between folds.</li>
        <li><strong>Only four moods,</strong> and they overlap (a sunny beach is happy and calm).</li>
        <li><strong>Overconfident.</strong> KNN often says 100 %: it is a vote share, not a calibrated probability.</li>
        <li><strong>Captions can be wrong;</strong> the poem follows the mistake.</li>
        <li><strong>Poems rarely rhyme</strong> (only ~8 % of targets did).</li>
        <li><strong>My scorer counts words,</strong> it cannot judge beauty.</li>
        <li><strong>Adapter saved at the last step,</strong> after validation loss started rising.</li>
      </ul></div>
      <div class="panel"><h4>Next steps</h4><ol class="small">
        <li>Rebuild poem data keeping only rhyming candidates, retrain.</li>
        <li>Keep the best validation checkpoint (early stopping).</li>
        <li>More photos, labelled by two classmates too; measure agreement.</li>
        <li>Calibrate the mood probabilities.</li>
        <li>Blind human rating of base vs fine-tuned poems.</li>
      </ol></div>
    </div>
  </div>

  <div role="tabpanel" id="mp-gloss" aria-labelledby="mt-gloss" tabindex="0" hidden>
    <dl class="gloss">{{GLOSSARY_HTML}}</dl>
  </div>

  <div role="tabpanel" id="mp-run" aria-labelledby="mt-run" tabindex="0" hidden>
    <div class="code"><button class="copy" type="button">Copy</button><pre><code>python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_HUB_OFFLINE=1                      <span class="cmt"># after the first model download (~2.5 GB)</span>
python app.py                                <span class="cmt"># website http://127.0.0.1:7860, Gradio at /gradio</span>

python -m src.build_dataset --per-mood 60    <span class="cmt"># mood data, then clean by hand</span>
python -m src.features                       <span class="cmt"># CLIP embeddings</span>
python -m src.train_mood                     <span class="cmt"># compare 5 models, save the best</span>

python -m src.poem_data                      <span class="cmt"># stop app.py first; ~1.5 h on Mac GPU</span>
python -m src.train_poet                     <span class="cmt"># LoRA fine-tune, ~6 min</span>
python -m src.eval_poet                      <span class="cmt"># base vs fine-tuned, ~5 min</span>

python -m src.make_samples && python -m src.site_data && python site/build_site.py</code></pre></div>
    <h3>References</h3>
    <ol class="refs">
      <li>Li et al. (2022). <em>BLIP.</em> <a href="https://arxiv.org/abs/2201.12086">arXiv:2201.12086</a></li>
      <li>Radford et al. (2021). <em>CLIP.</em> <a href="https://arxiv.org/abs/2103.00020">arXiv:2103.00020</a></li>
      <li>Hu et al. (2021). <em>LoRA.</em> <a href="https://arxiv.org/abs/2106.09685">arXiv:2106.09685</a></li>
      <li>Qwen Team (2024). <em>Qwen2.5 Technical Report.</em> <a href="https://arxiv.org/abs/2412.15115">arXiv:2412.15115</a></li>
      <li>Pedregosa et al. (2011). <em>Scikit-learn.</em> JMLR 12.</li>
      <li>van der Maaten and Hinton (2008). <em>t-SNE.</em> JMLR 9.</li>
      <li>Openverse, <a href="https://openverse.org">openverse.org</a>; CMU Pronouncing Dictionary via <code>pronouncing</code>.</li>
    </ol>
  </div>
  <footer class="foot"><span>Image to Poem · college ML mini-project</span><span class="mono">built {{BUILT}}</span></footer>
</section>

</main>
</div>
<div id="tip" role="tooltip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>/*JS*/</script>
</body>
</html>
"""

# ---------------------------------------------------------------- JS
JS = r"""
(function () {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
let D = {};
try { D = JSON.parse($('script#data').textContent || '{}'); } catch (e) { D = {}; }
const M = D.mood || {}, P = D.poet || {}, S = D.samples || [];
const MOODS = D.moods || ['happy', 'sad', 'calm', 'energetic'];
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const pct = (v, d = 0) => (Number(v) * 100).toFixed(d) + '%';
const f2 = v => Number(v).toFixed(2);
const sum = a => a.reduce((x, y) => x + y, 0);
const mean = a => a.length ? sum(a) / a.length : 0;
const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0);
const mc = m => `var(--${MOODS.includes(m) ? m : 'muted'})`;
const sw = c => `<i class="sw" style="background:${c}"></i>`;
const chip = m => `<span class="chip">${sw(mc(m))}${esc(m)}</span>`;
const tipAttr = html => `data-tip="${esc(html)}"`;
const SHAPE = { happy: 'circle', sad: 'square', calm: 'triangle', energetic: 'diamond' };
const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function nice(lo, hi, n = 5) {
  if (hi === lo) hi = lo + 1;
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const st = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= step0 - 1e-12);
  const a = Math.floor(lo / st + 1e-9) * st, b = Math.ceil(hi / st - 1e-9) * st, t = [];
  for (let v = a; v <= b + st / 1e6; v += st) t.push(+v.toFixed(10));
  return { lo: a, hi: b, ticks: t };
}
function shape(kind, x, y, r, style) {
  x = +x.toFixed(1); y = +y.toFixed(1);
  if (kind === 'square') { const s = r * .88; return `<rect x="${x - s}" y="${y - s}" width="${2 * s}" height="${2 * s}" style="${style}"/>`; }
  if (kind === 'triangle') { const s = r * 1.2; return `<path d="M${x},${y - s}L${x + s},${y + s * .8}L${x - s},${y + s * .8}Z" style="${style}"/>`; }
  if (kind === 'diamond') { const s = r * 1.3; return `<path d="M${x},${y - s}L${x + s},${y}L${x},${y + s}L${x - s},${y}Z" style="${style}"/>`; }
  return `<circle cx="${x}" cy="${y}" r="${r}" style="${style}"/>`;
}
const shapeIcon = (kind, style) => `<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">${shape(kind, 6, 6, 4, style)}</svg>`;
const lineIcon = (c, dash) => `<svg width="20" height="10" aria-hidden="true"><line x1="1" x2="19" y1="5" y2="5" style="stroke:${c};stroke-width:2;${dash ? 'stroke-dasharray:3 3' : ''}"/></svg>`;
const boxIcon = c => `<svg width="12" height="12" aria-hidden="true"><rect x="1" y="1" width="10" height="10" rx="2" style="fill:${c}"/></svg>`;
const legend = items => items.map(([label, icon]) => `<span>${icon}${esc(label)}</span>`).join('');
function table(head, rows, num = [], cap) {
  return `<div class="tbl-wrap"><table>${cap ? `<caption>${esc(cap)}</caption>` : ''}<thead><tr>${head.map((h, i) => `<th${num.includes(i) ? ' class="n"' : ''} scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${
    rows.map(r => `<tr${r.hl ? ' class="hl"' : ''}>${(r.cells || r).map((c, i) => `<td${num.includes(i) ? ' class="n"' : ''}>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
async function copyText(t) {
  try { await navigator.clipboard.writeText(t); return true; } catch (e) {
    const a = document.createElement('textarea'); a.value = t; a.style.position = 'fixed'; a.style.opacity = '0';
    document.body.appendChild(a); a.select(); let ok = false; try { ok = document.execCommand('copy'); } catch (_) {} a.remove(); return ok; }
}
function flash(btn, txt) { const o = btn.textContent; btn.textContent = txt; setTimeout(() => { btn.textContent = o; }, 1400); }

/* ---------- tooltip ---------- */
const tip = $('#tip');
function showTip(html, x, y) {
  tip.innerHTML = html; tip.style.display = 'block';
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let L = x + 14, T = y + 14;
  if (L + w > innerWidth - 8) L = x - w - 14; if (L < 8) L = 8;
  if (T + h > innerHeight - 8) T = y - h - 14; if (T < 8) T = 8;
  tip.style.left = L + 'px'; tip.style.top = T + 'px';
}
function hideTip() { tip.style.display = 'none'; }
document.addEventListener('pointermove', e => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (t) showTip(t.dataset.tip, e.clientX, e.clientY);
  else if (!(e.target.closest && e.target.closest('.xh'))) hideTip();
}, { passive: true });
document.addEventListener('focusin', e => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (t) { const r = t.getBoundingClientRect(); showTip(t.dataset.tip, r.right, r.top); }
});
document.addEventListener('focusout', hideTip);
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideTip(); });
addEventListener('scroll', hideTip, { passive: true });

/* ---------- tabs (WAI-ARIA, arrow keys) ---------- */
function tablist(list, onSelect) {
  const tabs = $$('[role=tab]', list);
  const select = (t, focus) => {
    tabs.forEach(x => {
      const on = x === t; x.setAttribute('aria-selected', on); x.tabIndex = on ? 0 : -1;
      const p = x.getAttribute('aria-controls'); const pe = p && document.getElementById(p); if (pe) pe.hidden = !on;
    });
    if (focus) t.focus();
    if (onSelect) onSelect(t.dataset.v, t);
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(t));
    t.addEventListener('keydown', e => {
      let j = null;
      if (e.key === 'ArrowRight') j = (i + 1) % tabs.length; else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') j = 0; else if (e.key === 'End') j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); select(tabs[j], true); }
    });
  });
  const init = tabs.find(t => t.getAttribute('aria-selected') === 'true') || tabs[0];
  if (init) select(init);
}
const seg = (label, items, cur) => `<span class="seg-l">${esc(label)}<span class="seg" role="tablist" aria-label="${esc(label)}">${items.map(([v, l]) =>
  `<button type="button" role="tab" data-v="${esc(v)}" aria-selected="${v === cur}">${esc(l)}</button>`).join('')}</span></span>`;

/* ---------- figure shell: title, controls, legend, plot, table toggle, caption ---------- */
let figN = 0;
function mount(el, draw) {
  let w = 0;
  const run = () => { const nw = Math.round(el.clientWidth); if (nw > 0 && nw !== w) { w = nw; draw(el, nw); } };
  if ('ResizeObserver' in window) new ResizeObserver(run).observe(el); else addEventListener('resize', run);
  el._redraw = () => { const nw = Math.round(el.clientWidth); if (nw > 0) { w = nw; draw(el, nw); } };
  run();
}
function fig(slot, o) {
  if (!slot) return null;
  const id = 'fig' + (++figN);
  const f = document.createElement('figure');
  f.className = 'fig';
  f.setAttribute('aria-labelledby', id + 't');
  f.innerHTML = `<div class="fig-head"><div><h4 id="${id}t">${esc(o.title)}</h4>${o.sub ? `<p class="sub">${esc(o.sub)}</p>` : ''}</div>${
    o.table ? `<button class="tbtn" type="button" aria-expanded="false" aria-controls="${id}v">Show table</button>` : ''}</div>${
    o.controls ? `<div class="controls">${o.controls}</div>` : ''}${o.legend ? `<div class="legend">${o.legend}</div>` : ''}<div class="plot"></div>${
    o.table ? `<div class="tview" id="${id}v" hidden></div>` : ''}<figcaption>${
    o.how ? `<p><strong>How to read it.</strong> ${o.how}</p>` : ''}${o.what ? `<p><strong>What it shows.</strong> ${o.what}</p>` : ''}</figcaption>`;
  slot.replaceWith(f);
  const plot = $('.plot', f);
  if (o.table) {
    const b = $('.tbtn', f), tv = $('.tview', f);
    f._refreshTable = () => { if (!tv.hidden) tv.innerHTML = typeof o.table === 'function' ? o.table() : o.table; };
    b.onclick = () => { const on = tv.hidden; tv.hidden = !on; b.setAttribute('aria-expanded', on); b.textContent = on ? 'Hide table' : 'Show table'; f._refreshTable(); };
  }
  if (o.draw) mount(plot, o.draw);
  f.plot = plot;
  return f;
}
const slot = name => $(`[data-fig="${name}"]`);

/* ---------- chart: line with crosshair + keyboard ---------- */
function lineChart(el, w, o) {
  const h = o.h || 260, m = { l: o.ml || 46, r: 16, t: 22, b: 38 };
  const X = lin(o.xd[0], o.xd[1], m.l, w - m.r), Y = lin(o.yn.lo, o.yn.hi, h - m.b, m.t);
  let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.aria || 'Line chart')}">`;
  o.yn.ticks.forEach(t => { s += `<line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="tick" x="${m.l - 8}" y="${Y(t)}" dy=".32em" text-anchor="end">${esc(o.fy(t))}</text>`; });
  const xt = o.xticks || nice(o.xd[0], o.xd[1], Math.max(2, Math.floor((w - m.l - m.r) / 70))).ticks.filter(t => t >= o.xd[0] && t <= o.xd[1]);
  xt.forEach(t => { s += `<text class="tick" x="${X(t)}" y="${h - m.b + 18}" text-anchor="middle">${esc(o.fx(t))}</text>`; });
  s += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}"/>`;
  if (o.xLabel) s += `<text class="alab" x="${w - m.r}" y="${h - 3}" text-anchor="end">${esc(o.xLabel)}</text>`;
  if (o.yLabel) s += `<text class="alab" x="${m.l - 8}" y="10">${esc(o.yLabel)}</text>`;
  (o.refs || []).forEach(r => {
    if (r.y != null) s += `<line class="ref" x1="${m.l}" x2="${w - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}"/><text class="reflab" x="${m.l + 6}" y="${Y(r.y) - 6}">${esc(r.label)}</text>`;
    if (r.x != null) { const end = !r.top && X(r.x) > w - 120;
      s += `<line class="ref" x1="${X(r.x)}" x2="${X(r.x)}" y1="${m.t}" y2="${h - m.b}"/><text class="reflab" x="${X(r.x) + (end ? -6 : 6)}" y="${r.top ? m.t + 10 : h - m.b - 8}" text-anchor="${end ? 'end' : 'start'}">${esc(r.label)}</text>`; }
  });
  o.series.forEach(se => {
    let d = '', pen = false;
    o.x.forEach((xv, i) => { const v = se.ys[i]; if (v == null || isNaN(v)) { if (!se.connect) pen = false; return; }
      d += (pen ? 'L' : 'M') + X(xv).toFixed(1) + ',' + Y(v).toFixed(1); pen = true; });
    if (d) s += `<path d="${d}" style="fill:none;stroke:${se.color};stroke-width:${se.width || 2};stroke-linejoin:round;stroke-linecap:round;${se.dash ? 'stroke-dasharray:4 3' : ''}"/>`;
    if (se.dots) o.x.forEach((xv, i) => { const v = se.ys[i]; if (v != null) s += `<circle cx="${X(xv)}" cy="${Y(v)}" r="${se.dots}" style="fill:${se.color};stroke:var(--surface);stroke-width:1.5"/>`; });
  });
  (o.marks || []).forEach(k => { s += `<circle cx="${X(k.x)}" cy="${Y(k.y)}" r="5.5" style="fill:none;stroke:var(--ink);stroke-width:1.5"/>`;
    if (k.label) { const end = X(k.x) > w - 140; s += `<text class="dl" x="${X(k.x) + (end ? -10 : 10)}" y="${Y(k.y) + (k.below ? 18 : -10)}" text-anchor="${end ? 'end' : 'start'}">${esc(k.label)}</text>`; } });
  s += `<g pointer-events="none"><line class="xhl" y1="${m.t}" y2="${h - m.b}" style="stroke:var(--muted);stroke-width:1" visibility="hidden"/><g class="hld"></g></g>`;
  s += `<rect class="xh" x="${m.l}" y="${m.t}" width="${Math.max(0, w - m.l - m.r)}" height="${h - m.t - m.b}" fill="transparent" tabindex="0" aria-label="${esc((o.aria || 'Chart') + '. Use left and right arrow keys to read values.')}"/></svg>`;
  el.innerHTML = s;
  const svg = el.firstChild, hl = $('.xhl', svg), hd = $('.hld', svg), ov = $('.xh', svg);
  let cur = 0;
  const at = i => {
    const xv = o.x[i]; hl.setAttribute('x1', X(xv)); hl.setAttribute('x2', X(xv)); hl.setAttribute('visibility', 'visible');
    let dots = '', rows = '';
    o.series.forEach(se => { let v = se.ys[i], xi = xv;
      if (v == null && se.connect) { for (let d = 1; d <= 6 && v == null; d++) for (const j of [i - d, i + d]) if (v == null && se.ys[j] != null) { v = se.ys[j]; xi = o.x[j]; } }
      if (v == null) return;
      dots += `<circle cx="${X(xi)}" cy="${Y(v)}" r="4.5" style="fill:${se.color};stroke:var(--surface);stroke-width:2"/>`;
      rows += `<div class="tr">${sw(se.color)}${esc(se.name)}<b>${esc((o.ft || o.fy)(v))}</b></div>`; });
    hd.innerHTML = dots;
    return `<div class="th">${esc(o.tipX ? o.tipX(xv) : o.fx(xv))}</div>${rows}`;
  };
  const near = px => { let b = 0, bd = 1e9; o.x.forEach((xv, i) => { const d = Math.abs(X(xv) - px); if (d < bd) { bd = d; b = i; } }); return b; };
  const off = () => { hl.setAttribute('visibility', 'hidden'); hd.innerHTML = ''; hideTip(); };
  ov.addEventListener('pointermove', e => { const r = svg.getBoundingClientRect(); cur = near(e.clientX - r.left); showTip(at(cur), e.clientX, e.clientY); });
  ov.addEventListener('pointerleave', off); ov.addEventListener('blur', off);
  ov.addEventListener('keydown', e => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return; e.preventDefault();
    cur = Math.max(0, Math.min(o.x.length - 1, cur + (e.key === 'ArrowRight' ? 1 : -1)));
    const r = svg.getBoundingClientRect(); showTip(at(cur), r.left + X(o.x[cur]), r.top + m.t);
  });
}

/* ---------- chart: grouped vertical bars ---------- */
function vbars(el, w, o) {
  const h = o.h || 240, m = { l: 44, r: 12, t: 20, b: 38 };
  const n = o.cats.length, cw = (w - m.l - m.r) / n, ns = o.series.length, gap = 2;
  const bw = Math.max(3, Math.min(o.maxBw || 40, (cw * .72 - gap * (ns - 1)) / ns));
  const Y = lin(o.yn.lo, o.yn.hi, h - m.b, m.t);
  let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.aria || 'Bar chart')}">`;
  o.yn.ticks.forEach(t => { s += `<line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="tick" x="${m.l - 8}" y="${Y(t)}" dy=".32em" text-anchor="end">${esc(o.fy(t))}</text>`; });
  const every = Math.max(1, Math.ceil(30 / cw));
  o.cats.forEach((c, ci) => {
    if (ci % every === 0) s += `<text class="tick" x="${m.l + (ci + .5) * cw}" y="${h - m.b + 18}" text-anchor="middle">${esc(c)}</text>`;
    const gx = m.l + ci * cw + (cw - (ns * bw + (ns - 1) * gap)) / 2;
    o.series.forEach((se, si) => {
      const v = se.vals[ci]; if (v == null) return;
      const x = gx + si * (bw + gap), y = Y(v), hh = Math.max(0, Y(0) - y);
      const t = o.tip ? o.tip(ci, si) : `<div class="th">${esc(c)}</div><div class="tr">${sw(se.color)}${esc(se.name)}<b>${esc((o.ft || o.fy)(v))}</b></div>`;
      if (hh > 0) s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${hh.toFixed(1)}" rx="1.5" style="fill:${(se.colors && se.colors[ci]) || se.color}" pointer-events="none"/>`;
      s += `<rect x="${x.toFixed(1)}" y="${m.t}" width="${bw.toFixed(1)}" height="${h - m.b - m.t}" fill="transparent" ${tipAttr(t)} tabindex="0" aria-label="${esc(c + ', ' + se.name + ': ' + (o.ft || o.fy)(v))}"/>`;
      if (o.labels && bw >= 18) s += `<text class="dl" x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" pointer-events="none">${esc((o.fl || o.fy)(v))}</text>`;
    });
  });
  (o.refs || []).forEach(r => { const x = m.l + r.at * cw; s += `<line class="ref" x1="${x}" x2="${x}" y1="${m.t - 6}" y2="${h - m.b}"/><text class="reflab" x="${x + 6}" y="${m.t + 4}">${esc(r.label)}</text>`; });
  s += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${Y(0)}" y2="${Y(0)}"/>`;
  if (o.xLabel) s += `<text class="alab" x="${w - m.r}" y="${h - 3}" text-anchor="end">${esc(o.xLabel)}</text>`;
  if (o.yLabel) s += `<text class="alab" x="${m.l - 8}" y="10">${esc(o.yLabel)}</text>`;
  el.innerHTML = s + '</svg>';
}

/* ---------- chart: horizontal bars, direct labels, optional groups ---------- */
function hbars(el, w, o) {
  // o.rows: [{label, bars:[{v,color,name,tip}]}]
  const nb = Math.max(...o.rows.map(r => r.bars.length)), bh = o.bh || 14, gap = 3;
  const rowH = nb * bh + (nb - 1) * gap + 18, lw = Math.min(o.lw || 130, w * .38), rw = 52, h = o.rows.length * rowH + 4;
  const max = o.max || Math.max(...o.rows.flatMap(r => r.bars.map(b => b.v))) || 1;
  const X = lin(0, max, lw, w - rw);
  let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.aria || 'Bar chart')}">`;
  s += `<line class="axis" x1="${lw}" x2="${lw}" y1="0" y2="${h - 4}"/>`;
  o.rows.forEach((r, i) => {
    const y0 = i * rowH + 9, mid = y0 + (nb * bh + (nb - 1) * gap) / 2;
    s += `<text class="cl${r.bold ? ' b' : ''}" x="${lw - 10}" y="${mid}" dy=".34em" text-anchor="end">${esc(r.label)}</text>`;
    r.bars.forEach((b, j) => {
      const y = y0 + j * (bh + gap), x1 = X(b.v);
      s += `<rect x="${lw}" y="${y}" width="${Math.max(0, x1 - lw).toFixed(1)}" height="${bh}" rx="1.5" style="fill:${b.color}"/>`;
      s += `<text class="dl" x="${x1 + 6}" y="${y + bh / 2}" dy=".34em">${esc(o.fmt(b.v))}</text>`;
    });
    const t = r.tip || `<div class="th">${esc(r.label)}</div>` + r.bars.map(b => `<div class="tr">${sw(b.color)}${esc(b.name || '')}<b>${esc(o.fmt(b.v))}</b></div>`).join('');
    s += `<rect x="0" y="${i * rowH}" width="${w}" height="${rowH}" fill="transparent" ${tipAttr(t)} tabindex="0" aria-label="${esc(r.label + ': ' + r.bars.map(b => (b.name ? b.name + ' ' : '') + o.fmt(b.v)).join(', '))}"/>`;
  });
  el.innerHTML = s + '</svg>';
}

/* ================= FIGURES: data & mood classifier ================= */
const models = (M.models || []).slice();
const byName = Object.fromEntries(models.map(m => [m.name, m]));
const best = byName[M.best] || models[0];

(function counts() {
  const c = M.counts; if (!c) return;
  const tot = sum(MOODS.map(m => c[m] || 0));
  fig(slot('counts'), {
    title: 'Images per mood after cleaning', sub: `${tot} images in total`,
    draw: (el, w) => hbars(el, w, { aria: 'Images per mood', lw: 100, fmt: v => String(v),
      rows: MOODS.map(m => ({ label: m, bars: [{ v: c[m] || 0, color: mc(m), name: 'images' }] })) }),
    table: () => table(['Mood', 'Images', 'Share'], MOODS.map(m => [m, c[m], pct((c[m] || 0) / tot, 1)]), [1, 2]),
    how: 'One bar per mood; the number at the end of each bar is the image count.',
    what: `The classes are close to balanced (${Math.min(...MOODS.map(m => c[m]))} to ${Math.max(...MOODS.map(m => c[m]))} images), so plain accuracy is a fair metric and no class needs re-weighting.`
  });
})();

(function queries() {
  const Q = { happy: ['smiling people', 'birthday party', 'sunflower field', 'children laughing', 'colorful balloons'],
    sad: ['rainy window', 'lonely person', 'gray rainy street', 'wilted flower', 'abandoned house'],
    calm: ['calm lake', 'misty forest', 'zen garden', 'quiet beach sunset', 'still water reflection'],
    energetic: ['concert crowd', 'running athlete', 'fireworks', 'skateboarding trick', 'dance party'] };
  const tb = $('#queries'); if (!tb) return;
  tb.innerHTML = MOODS.map(m => `<tr><td>${chip(m)}</td><td>${(Q[m] || []).map(esc).join(', ')}</td></tr>`).join('');
})();

(function moodGrid() {
  const g = $('#moodGrid'), pts = M.points || []; if (!g || !pts.length) { if (g) g.remove(); return; }
  g.innerHTML = MOODS.map(m => {
    const imgs = pts.filter(p => p.mood === m).slice(0, 6);
    return `<div class="mood-row"><div>${chip(m)}<div class="lab">${(M.counts || {})[m] || ''} images</div></div><div class="imgs">${
      imgs.map((p, i) => `<img src="${esc(p.thumb)}" alt="Training image ${i + 1} labelled ${esc(m)}" loading="lazy" width="160" height="160">`).join('')}</div></div>`;
  }).join('');
})();

(function compare() {
  if (!models.length) return;
  const rows = models.slice().sort((a, b) => b.cv_mean - a.cv_mean);
  const all = rows.flatMap(r => [...(r.cv_folds || []), r.test.accuracy, r.cv_mean - r.cv_std]);
  const lo = Math.floor(Math.min(...all) * 10) / 10, hi = 1;
  fig(slot('compare'), {
    title: 'Five models, same features: accuracy', sub: 'Sorted by cross-validation mean. Higher is better.',
    legend: legend([['CV mean ± 1 std', lineIcon('var(--ink)')], ['one CV fold', shapeIcon('circle', 'fill:none;stroke:var(--ink-2);stroke-width:1.2')], ['held-out test accuracy', shapeIcon('diamond', 'fill:var(--surface);stroke:var(--ink);stroke-width:1.5')]]),
    draw: (el, w) => {
      const lw = Math.min(150, w * .34), rowH = 46, m = { t: 8, b: 34 }, h = rows.length * rowH + m.t + m.b;
      const X = lin(lo, hi, lw, w - 16), tk = nice(lo, hi, Math.max(2, Math.floor((w - lw) / 60))).ticks.filter(t => t >= lo && t <= hi);
      let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Accuracy per model: cross-validation folds, mean with standard deviation, and test accuracy">`;
      tk.forEach(t => { s += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${h - m.b}"/><text class="tick" x="${X(t)}" y="${h - m.b + 18}" text-anchor="middle">${Math.round(t * 100)}%</text>`; });
      s += `<text class="alab" x="${w - 16}" y="${h - 2}" text-anchor="end">accuracy</text>`;
      rows.forEach((r, i) => {
        const y = m.t + i * rowH + rowH / 2, isBest = r.name === M.best;
        const short = { 'Logistic Regression': 'LogReg', 'Random Forest': 'Forest', 'Decision Tree': 'Tree' };
        s += `<text class="cl${isBest ? ' b' : ''}" style="${isBest ? 'fill:var(--accent)' : ''}" x="${lw - 14}" y="${y}" dy=".34em" text-anchor="end">${esc(w < 520 ? (short[r.name] || r.name) : r.name)}</text>`;
        (r.cv_folds || []).forEach((v, k) => { s += shape('circle', X(v), y - 9 + (k % 2) * 3, 3, 'fill:none;stroke:var(--ink-2);stroke-width:1.2'); });
        const a = X(r.cv_mean - r.cv_std), b = X(Math.min(1, r.cv_mean + r.cv_std));
        s += `<line x1="${a}" x2="${b}" y1="${y + 4}" y2="${y + 4}" style="stroke:${isBest ? 'var(--accent)' : 'var(--ink-2)'};stroke-width:${isBest ? 3 : 2}"/><line x1="${a}" x2="${a}" y1="${y}" y2="${y + 8}" style="stroke:${isBest ? 'var(--accent)' : 'var(--ink-2)'};stroke-width:1.5"/><line x1="${b}" x2="${b}" y1="${y}" y2="${y + 8}" style="stroke:${isBest ? 'var(--accent)' : 'var(--ink-2)'};stroke-width:1.5"/>`;
        s += `<circle cx="${X(r.cv_mean)}" cy="${y + 4}" r="${isBest ? 6 : 4.5}" style="fill:${isBest ? 'var(--accent)' : 'var(--ink-2)'}"/>`;
        s += shape('diamond', X(r.test.accuracy), y + 4, 4.5, 'fill:var(--surface);stroke:var(--ink);stroke-width:1.5');
        const t = `<div class="th">${esc(r.name)}${isBest ? ' (chosen)' : ''}</div><div class="tr">CV mean<b>${pct(r.cv_mean, 1)} ± ${pct(r.cv_std, 1)}</b></div><div class="tr">folds<b>${(r.cv_folds || []).map(v => Math.round(v * 100)).join(' · ')}</b></div><div class="tr">test accuracy<b>${pct(r.test.accuracy, 1)}</b></div>`;
        s += `<rect x="0" y="${y - rowH / 2}" width="${w}" height="${rowH}" fill="transparent" ${tipAttr(t)} tabindex="0" aria-label="${esc(`${r.name}: CV ${pct(r.cv_mean, 1)} plus or minus ${pct(r.cv_std, 1)}, test ${pct(r.test.accuracy, 1)}`)}"/>`;
      });
      el.innerHTML = s + '</svg>';
    },
    table: () => table(['Model', 'CV mean', 'CV std', 'Folds', 'Test acc.'], rows.map(r => ({ hl: r.name === M.best, cells: [r.name, pct(r.cv_mean, 1), pct(r.cv_std, 1), (r.cv_folds || []).map(v => pct(v, 1)).join(', '), pct(r.test.accuracy, 1)] })), [1, 2, 4]),
    how: 'Each row is one model. The solid dot and line show the mean of the five cross-validation scores plus and minus one standard deviation; the small hollow circles are the five individual folds; the diamond is accuracy on the 51 test images.',
    what: (() => { const [a, b] = rows; const dt = byName['Decision Tree']; const mlp = byName.MLP;
      return `${esc(a.name)} (${pct(a.cv_mean, 1)}) and ${esc(b.name)} (${pct(b.cv_mean, 1)}) are within each other's spread, so they are effectively tied; I keep ${esc(a.name)} because it has the higher CV mean.${dt ? ` The Decision Tree is clearly worse (${pct(dt.cv_mean, 1)}): single threshold splits on PCA axes do not suit dense features.` : ''}${mlp ? ` The MLP has the highest test score (${pct(mlp.test.accuracy, 1)}), but on 51 images one picture moves the score by 2 points, so I did not choose on the test set.` : ''}`; })()
  });
})();

(function metrics() {
  if (!models.length) return;
  const rows = models.slice().sort((a, b) => b.cv_mean - a.cv_mean);
  fig(slot('metrics'), {
    title: 'Test-set metrics for every model', sub: 'Macro average: each mood counts equally.',
    draw: (el) => { el.innerHTML = table(['Model', 'CV acc.', 'Test acc.', 'Precision', 'Recall', 'F1', 'Fit time'], rows.map(r => ({ hl: r.name === M.best,
      cells: [r.name, r.cv_mean.toFixed(3), r.test.accuracy.toFixed(3), r.test.precision.toFixed(3), r.test.recall.toFixed(3), r.test.f1.toFixed(3), (r.seconds != null ? r.seconds.toFixed(2) + ' s' : '')] })), [1, 2, 3, 4, 5, 6]); },
    how: 'Rows are models, sorted by CV accuracy; the chosen model is in bold. Precision, recall and F1 are averaged over the four moods. Fit time is for 5-fold CV plus the final fit on my laptop.',
    what: `Precision and recall are close to accuracy for every model, so no model wins by ignoring one mood. All five train in well under a second because they work on ${M.pca ? M.pca.n90 : 98} PCA numbers per image, not on pixels.`
  });
})();

(function confusion() {
  if (!models.length) return;
  let cur = best.name;
  let f = null;
  f = fig(slot('confusion'), {
    title: 'Confusion matrix on the test images', sub: 'Rows: true mood. Columns: predicted mood.',
    controls: seg('Model', models.map(m => [m.name, m.name === 'Logistic Regression' ? 'LogReg' : m.name === 'Random Forest' ? 'Forest' : m.name === 'Decision Tree' ? 'Tree' : m.name]), cur),
    draw: (el) => {
      const r = byName[cur], cm = r.confusion, mx = Math.max(...cm.flat());
      el.innerHTML = `<div class="cm-wrap"><table class="cm"><caption class="sr">Confusion matrix for ${esc(cur)}</caption><thead><tr><th></th>${MOODS.map(m => `<th scope="col">${sw(mc(m))} ${esc(m)}</th>`).join('')}</tr></thead><tbody>${
        cm.map((row, i) => `<tr><th scope="row">${sw(mc(MOODS[i]))} ${esc(MOODS[i])}</th>${row.map((v, j) => { const a = v / mx, mix = v ? Math.round(8 + a * 84) : 0, dark = mix > 50;
          return `<td style="background:color-mix(in srgb, ${i === j ? 'var(--accent)' : 'var(--energetic)'} ${i === j ? mix : Math.round(mix * .7)}%, var(--soft));color:${dark && i === j ? 'var(--on-accent)' : 'var(--ink)'};${i === j ? 'box-shadow:inset 0 0 0 1.5px var(--accent)' : ''}" ${tipAttr(`<div class="th">true ${esc(MOODS[i])}, predicted ${esc(MOODS[j])}</div>${v} of ${sum(row)} images`)} tabindex="0">${v}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>
        <p class="cm-axis" style="margin:8px 0 0">${esc(cur)}: ${sum(cm.map((row, i) => row[i]))} of ${sum(cm.flat())} correct</p>`;
      if (f && f._refreshTable) f._refreshTable();
    },
    table: () => { const r = byName[cur];
      return table(['Mood', 'Precision', 'Recall', 'F1', 'Test images'], MOODS.map(m => { const p = r.per_class[m]; return [m, p.precision.toFixed(3), p.recall.toFixed(3), p.f1.toFixed(3), p.support]; }), [1, 2, 3, 4], `Per-class results, ${cur}`); },
    how: 'Bright diagonal = correct predictions; red cells = mistakes. The stronger the shade, the more images in the cell. Read across a row to see where images of one true mood ended up. Switch models above; Show table gives per-mood precision, recall and F1.',
    what: (() => { const pc = best.per_class; const worst = MOODS.slice().sort((a, b) => pc[a].recall - pc[b].recall)[0]; const i = MOODS.indexOf(worst);
      const row = best.confusion[i]; const j = row.map((v, k) => k === i ? -1 : v).reduce((bi, v, k, a) => v > a[bi] ? k : bi, 0);
      return `For ${esc(best.name)}, <em>${esc(worst)}</em> is the hardest mood: only ${row[i]} of ${sum(row)} are found (recall ${pct(pc[worst].recall)}), and the most common mistake is calling them <em>${esc(MOODS[j])}</em>. Calm and energetic are the easiest. This matches the data: happy photos overlap with both peaceful scenes and lively parties.`; })()
  });
  if (f) tablist($('.seg', f), v => { cur = v; f.plot._redraw(); });
})();

(function pcaFig() {
  const p = M.pca; if (!p || !p.cumulative) return;
  const x = p.cumulative.map((_, i) => i + 1);
  const at = k => p.cumulative[k - 1];
  fig(slot('pca'), {
    title: 'PCA: how much variance the first components keep', sub: 'Fitted on the standardised training features.',
    draw: (el, w) => lineChart(el, w, { aria: 'Cumulative explained variance by number of PCA components', x, xd: [1, x.length],
      yn: { lo: 0, hi: 1, ticks: [0, .25, .5, .75, 1] }, fy: v => Math.round(v * 100) + '%', ft: v => pct(v, 1), fx: v => String(v), tipX: v => `${v} components`,
      xLabel: 'number of components', series: [{ name: 'variance kept', color: 'var(--ink)', ys: p.cumulative }],
      refs: [{ y: .9, label: '90 %' }], marks: [{ x: p.n90, y: at(p.n90), label: `${p.n90} components`, below: true }] }),
    table: () => table(['Components', 'Variance kept'], [1, 2, 5, 10, 20, 50, p.n90, 100, 150].filter(k => k <= x.length).map(k => [k, pct(at(k), 1)]), [0, 1]),
    how: 'The line shows the share of the total variation kept if I keep the first n components. The dashed line is the 90 % target I set in the code (n_components=0.90).',
    what: `Reaching 90 % needs ${p.n90} of 512 components: an 81 % reduction. The first 10 components hold only ${pct(at(10))}, so mood information is spread over many directions, which is why a 2-D plot alone cannot separate the moods.`
  });
})();

/* ================= FIGURES: unsupervised ================= */
(function elbow() {
  const k = M.kmeans; const s0 = slot('elbow'); if (!k || !s0) return;
  const s1 = document.createElement('div'); s0.after(s1);
  fig(s0, {
    title: 'Elbow plot: K-Means inertia for k = 1 to 10', sub: 'Inertia = sum of squared distances to the nearest centre. Lower = tighter.',
    draw: (el, w) => { const yn = nice(Math.min(...k.inertia) * .95, Math.max(...k.inertia), 4);
      lineChart(el, w, { aria: 'K-Means inertia by k', x: k.k, xd: [1, 10], xticks: k.k, yn, fy: v => String(Math.round(v)), ft: v => v.toFixed(1), fx: v => String(v), tipX: v => `k = ${v}`,
        xLabel: 'k (clusters)', series: [{ name: 'inertia', color: 'var(--ink)', ys: k.inertia, dots: 3.5 }] }); },
    table: () => table(['k', 'Inertia'], k.k.map((v, i) => [v, k.inertia[i].toFixed(2)]), [0, 1]),
    how: 'An "elbow", a point where the curve suddenly flattens, would suggest a natural number of groups.',
    what: `There is no clear elbow: inertia falls smoothly from ${Math.round(k.inertia[0])} to ${Math.round(k.inertia[k.inertia.length - 1])}. The images do not form a few tight, well-separated groups.`
  });
  const ks = Object.keys(k.silhouette).map(Number), sv = ks.map(v => k.silhouette[v]);
  fig(s1, {
    title: 'Silhouette score for k = 2 to 10', sub: 'From -1 to 1. Near 0 means clusters overlap.',
    draw: (el, w) => vbars(el, w, { aria: 'Silhouette score by k', cats: ks.map(String), yn: { lo: 0, hi: .1, ticks: [0, .025, .05, .075, .1] }, fy: v => v.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''), ft: v => v.toFixed(3), fl: v => v.toFixed(2),
      series: [{ name: 'silhouette', color: 'var(--base)', colors: ks.map(v => v === 4 ? 'var(--accent)' : 'var(--base)'), vals: sv }], labels: true, xLabel: 'k (clusters)', maxBw: 28,
      tip: ci => `<div class="th">k = ${ks[ci]}${ks[ci] === 4 ? ' (number of moods)' : ''}</div>silhouette <b>${sv[ci].toFixed(3)}</b>` }),
    table: () => table(['k', 'Silhouette'], ks.map((v, i) => [v, sv[i].toFixed(3)]), [0, 1]),
    how: 'One bar per number of clusters; the solid bar is k = 4, the number of moods. A score above about 0.5 means clear clusters; around 0 means points sit between clusters.',
    what: `Every value is below 0.1. At k = 4, the number of moods, it is ${(k.silhouette[4] || 0).toFixed(2)}. So the clusters K-Means finds are weak, whichever k I choose.`
  });
})();

(function map() {
  const pts = M.points; if (!pts || !pts.length) return;
  let proj = 'pca', by = 'mood';
  const CL = ['A', 'B', 'C', 'D', 'E', 'F'];
  const clStyle = [['circle', 'fill:var(--ink)'], ['square', 'fill:none;stroke:var(--ink);stroke-width:1.4'], ['triangle', 'fill:var(--base)'], ['diamond', 'fill:none;stroke:var(--ink-2);stroke-width:1.4']];
  const styleOf = p => by === 'mood' ? [SHAPE[p.mood], `fill:${mc(p.mood)};fill-opacity:.85;stroke:var(--surface);stroke-width:.6`] : clStyle[p.cluster % 4];
  const legendHTML = () => by === 'mood' ? legend(MOODS.map(m => [m, shapeIcon(SHAPE[m], `fill:${mc(m)}`)])) : legend([0, 1, 2, 3].map(c => [`cluster ${CL[c]}`, shapeIcon(clStyle[c][0], clStyle[c][1])]));
  const pv = M.pca && M.pca.pc2_var;
  const f = fig(slot('map'), {
    title: 'The 251 images on a 2-D map', sub: 'Each mark is one image. Hover to see the photo.',
    controls: seg('Map', [['pca', 'PCA'], ['tsne', 't-SNE']], proj) + seg('Colour by', [['mood', 'My mood label'], ['cluster', 'K-Means cluster']], by),
    legend: legendHTML(),
    draw: (el, w) => {
      const h = Math.round(Math.min(440, Math.max(300, w * .66))), m = 18;
      const xs = pts.map(p => p[proj][0]), ys = pts.map(p => p[proj][1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const X = lin(x0, x1, m, w - m), Y = lin(y0, y1, h - m - 16, m);
      let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Scatter plot of all images projected with ${proj === 'pca' ? 'PCA' : 't-SNE'}, marked by ${by === 'mood' ? 'mood' : 'cluster'}">`;
      s += `<rect x=".5" y=".5" width="${w - 1}" height="${h - 1}" rx="4" style="fill:none;stroke:var(--line)"/>`;
      pts.forEach(p => { const [k, st] = styleOf(p); s += shape(k, X(p[proj][0]), Y(p[proj][1]), 4, st); });
      const ax = proj === 'pca' && pv ? `PC1 (${pct(pv[0], 1)} of variance) →   PC2 (${pct(pv[1], 1)}) ↑` : 't-SNE axes have no units';
      s += `<text class="alab" x="${m}" y="${h - 8}">${esc(ax)}</text>`;
      s += `<circle class="hlp" r="8" style="fill:none;stroke:var(--ink);stroke-width:1.5" visibility="hidden"/>`;
      s += `<rect class="xh" x="0" y="0" width="${w}" height="${h}" fill="transparent"/></svg>`;
      el.innerHTML = s;
      const svg = el.firstChild, ov = $('.xh', svg), ring = $('.hlp', svg);
      ov.addEventListener('pointermove', e => {
        const r = svg.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
        let bi = -1, bd = 196;
        pts.forEach((p, i) => { const dx = X(p[proj][0]) - px, dy = Y(p[proj][1]) - py, d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = i; } });
        if (bi < 0) { ring.setAttribute('visibility', 'hidden'); hideTip(); return; }
        const p = pts[bi]; ring.setAttribute('cx', X(p[proj][0])); ring.setAttribute('cy', Y(p[proj][1])); ring.setAttribute('visibility', 'visible');
        showTip(`<img src="${esc(p.thumb)}" alt=""><div class="tr">${sw(mc(p.mood))}label<b>${esc(p.mood)}</b></div><div class="tr">K-Means cluster<b>${CL[p.cluster] || p.cluster}</b></div>`, e.clientX, e.clientY);
      });
      ov.addEventListener('pointerleave', () => { ring.setAttribute('visibility', 'hidden'); hideTip(); });
    },
    table: () => { const cl = [...new Set(pts.map(p => p.cluster))].sort();
      return table(['Cluster', ...MOODS, 'Total'], cl.map(c => { const g = pts.filter(p => p.cluster === c); return [CL[c] || c, ...MOODS.map(m => g.filter(p => p.mood === m).length), g.length]; }), [1, 2, 3, 4, 5], 'Images per K-Means cluster (k = 4) and mood'); },
    how: 'Close marks are images with similar CLIP features. PCA keeps the two directions of largest variance; t-SNE instead keeps local neighbours together, so distances between far-apart groups mean little. Colour and shape show either my label or the K-Means cluster. The table counts how the four clusters split across moods.',
    what: 'With mood labels, each mood occupies its own region, especially on the t-SNE map, but the regions touch and overlap. Switch to clusters and you see that K-Means cuts the map differently from my labels, which is what the ARI of ' + (M.kmeans ? M.kmeans.ari_k4.toFixed(2) : '') + ' measures.'
  });
  if (!f) return;
  const segs = $$('.seg', f);
  tablist(segs[0], v => { proj = v; f.plot._redraw(); });
  tablist(segs[1], v => { by = v; $('.legend', f).innerHTML = legendHTML(); f.plot._redraw(); });
})();

/* ================= FIGURES: poem model ================= */
(function scores() {
  const sc = P.scores; if (!sc) return;
  const lo = -0.3, hi = 0.8, bw = 0.05, nb = Math.round((hi - lo) / bw);
  const binsOf = a => { const b = new Array(nb).fill(0); a.forEach(v => { b[Math.max(0, Math.min(nb - 1, Math.floor((v - lo) / bw + 1e-9)))]++; }); return b; };
  const bc = binsOf(sc.candidates), bk = binsOf(sc.kept);
  const nc = sc.candidates.length, nk = sc.kept.length;
  const label = i => (lo + i * bw).toFixed(2);
  fig(slot('scores'), {
    title: 'Scores of all candidate poems vs the ones I kept', sub: `${nc.toLocaleString('en')} candidates, ${nk} kept. Share of each group per score bin.`,
    legend: legend([['all candidates', boxIcon('var(--base)')], ['kept for training', boxIcon('var(--accent)')]]),
    draw: (el, w) => vbars(el, w, { aria: 'Histogram of poem scores, candidates versus kept', cats: bc.map((_, i) => label(i)),
      yn: nice(0, Math.max(...bc.map(v => v / nc), ...bk.map(v => v / nk)), 4), fy: v => Math.round(v * 100) + '%', ft: v => pct(v, 1),
      series: [{ name: 'all candidates', color: 'var(--base)', vals: bc.map(v => v / nc) }, { name: 'kept', color: 'var(--accent)', vals: bk.map(v => v / nk) }],
      refs: [{ at: Math.round((0.45 - lo) / bw), label: 'keep ≥ 0.45' }], xLabel: 'poem score (bin start)', maxBw: 14,
      tip: ci => `<div class="th">score ${label(ci)} to ${(lo + (ci + 1) * bw).toFixed(2)}</div><div class="tr">${sw('var(--base)')}candidates<b>${bc[ci]} (${pct(bc[ci] / nc, 1)})</b></div><div class="tr">${sw('var(--ink)')}kept<b>${bk[ci]} (${pct(bk[ci] / nk, 1)})</b></div>` }),
    table: () => table(['Score bin', 'Candidates', 'Kept'], bc.map((v, i) => [`${label(i)} to ${(lo + (i + 1) * bw).toFixed(2)}`, v, bk[i]]).filter(r => r[1] || r[2]), [1, 2]),
    how: 'Pale bars are all poems the teacher wrote, solid bars are the poems that became training targets. Each group adds up to 100 %. The dashed line is the minimum score to be kept.',
    what: `Most candidates score low (mean ${mean(sc.candidates).toFixed(2)}); the kept poems are the top tail (mean ${mean(sc.kept).toFixed(2)}). Filtering raised the quality of the targets, but even the best candidates rarely rhyme, because the scorer can only pick from what the 0.5B teacher wrote.`
  });
})();

(function examples() {
  const el = $('#examples'), ex = (P.examples || []).slice(0, 3); if (!el) return;
  if (!ex.length) { el.closest('details').remove(); return; }
  el.innerHTML = ex.map(e => `<div class="ex"><img src="${esc(e.thumb)}" alt="Training image: ${esc(e.caption)}" loading="lazy"><div>
    <span class="lab">input: “${esc(e.caption)}” · mood ${esc(e.mood)} · score ${Number(e.score).toFixed(2)}</span>
    <p class="poem sm">${esc(e.poem)}</p></div></div>`).join('');
})();

(function trainInfo() {
  const el = $('#trainInfo'), t = P.train_info; if (!el || !t) return;
  const rows = [
    ['Base model', `${t.base} (frozen)`], ['Adapter', 'LoRA, rank r = 16, alpha 32 (scale alpha / r = 2), dropout 0.05'],
    ['Adapted layers', 'q, k, v, o attention projections and gate, up, down MLP projections, in all 24 layers'],
    ['Trainable parameters', `${t.trainable_params.toLocaleString('en')} of ${t.total_params.toLocaleString('en')} (${pct(t.trainable_params / t.total_params, 2)})`],
    ['Training data', `${t.train_pairs} pairs, validation ${t.val_pairs} pairs`],
    ['Epochs', String(t.epochs)], ['Batch', `${t.micro_batch} per step × ${t.grad_accum} accumulation steps = ${t.batch}`],
    ['Optimiser', `AdamW, learning rate ${t.lr}, weight decay 0, gradient clipping at 1.0`],
    ['Schedule', 'cosine decay with 6 % linear warm-up'],
    ['Memory savers', 'gradient checkpointing; batch 4 × 2 instead of 8 (batch 8 ran out of GPU memory)'],
    ['Precision / device', `float32 on ${t.device === 'mps' ? 'Apple GPU (MPS)' : esc(t.device)}`],
    ['Training time', `${t.train_seconds} s (${(t.train_seconds / 60).toFixed(1)} min)`]];
  el.innerHTML = rows.map(([k, v]) => `<tr><th scope="row" style="width:34%">${esc(k)}</th><td>${esc(v)}</td></tr>`).join('');
})();

(function loss() {
  const L = P.loss; if (!L) return;
  const vals = L.val.map((v, i) => [L.step[i], v]).filter(p => p[1] != null);
  const bestV = vals.reduce((a, b) => b[1] < a[1] ? b : a);
  const lastV = vals[vals.length - 1], firstV = vals[0];
  const trainVals = L.train.filter(v => v != null);
  const steps = L.step[L.step.length - 1], per = steps / ((P.train_info || {}).epochs || 3);
  fig(slot('loss'), {
    title: 'Training and validation loss', sub: 'Cross-entropy on poem tokens. Lower is better.',
    legend: legend([['training loss (each step, 8 poems)', lineIcon('var(--base)')], ['validation loss (every 10 steps, 19 poems)', lineIcon('var(--accent)')]]),
    draw: (el, w) => lineChart(el, w, { aria: 'Training and validation loss by step', x: L.step, xd: [0, steps],
      yn: nice(Math.min(...trainVals) * .95, Math.max(...trainVals, firstV[1]), 4), fy: v => v.toFixed(1), ft: v => v.toFixed(3), fx: v => String(v), tipX: v => `step ${v} · epoch ${(v / per).toFixed(2)}`,
      xLabel: 'optimiser step', series: [{ name: 'training', color: 'var(--base)', ys: L.train, width: 1.5 }, { name: 'validation', color: 'var(--accent)', ys: L.val, connect: true, dots: 3.5 }],
      refs: [{ x: Math.round(per), label: 'epoch 2', top: true }, { x: Math.round(per * 2), label: 'epoch 3', top: true }],
      marks: [{ x: bestV[0], y: bestV[1], label: `best val ${bestV[1].toFixed(2)}` }] }),
    table: () => table(['Step', 'Epoch', 'Training loss', 'Validation loss'], L.step.map((s, i) => [s, (s / per).toFixed(2), L.train[i] == null ? '' : L.train[i].toFixed(3), L.val[i] == null ? '' : L.val[i].toFixed(3)]).filter(r => r[3] !== '' || r[0] % 5 === 0), [0, 1, 2, 3]),
    how: 'The pale line is the loss on each training batch, so it is noisy. The solid line is the loss on the 19 validation poems the model never trains on. Dashed lines mark the start of epochs 2 and 3.',
    what: `Validation loss drops from ${firstV[1].toFixed(2)} to its best, ${bestV[1].toFixed(2)}, at step ${bestV[0]} (end of the first epoch), then slowly rises to ${lastV[1].toFixed(2)} while training loss keeps falling to about ${trainVals[trainVals.length - 1].toFixed(1)}. That gap is overfitting: in epochs 2 and 3 the model starts memorising the 270 training poems instead of learning general poem style.`
  });
})();

(function before() {
  const E = P.eval; if (!E) return;
  const keys = [['lines', 'exactly 8 lines'], ['rhyme', 'rhyme'], ['keyword', 'caption words'], ['mood', 'mood words'], ['total', 'weighted total']];
  fig(slot('before'), {
    title: 'Base model vs my fine-tuned model', sub: `Mean over ${E.n_poems} poems per model on ${(P.dataset || {}).test || 26} unseen test images. Higher is better.`,
    legend: legend([['base Qwen2.5-0.5B-Instruct', boxIcon('var(--base)')], ['fine-tuned (my LoRA)', boxIcon('var(--accent)')]]),
    draw: (el, w) => hbars(el, w, { aria: 'Poem scores, base versus fine-tuned', lw: 120, max: 1, fmt: v => v.toFixed(2),
      rows: keys.map(([k, l]) => ({ label: l, bold: k === 'lines', bars: [{ v: E.base[k], color: 'var(--base)', name: 'base' }, { v: E['fine-tuned'][k], color: 'var(--accent)', name: 'fine-tuned' }] })) }),
    table: () => table(['Metric', 'Base', 'Fine-tuned', 'Change'], [...keys, ['clip', 'CLIP similarity']].map(([k, l]) => [l, E.base[k].toFixed(3), E['fine-tuned'][k].toFixed(3), (E['fine-tuned'][k] - E.base[k] >= 0 ? '+' : '') + (E['fine-tuned'][k] - E.base[k]).toFixed(3)]), [1, 2, 3]),
    how: 'Each metric has two bars: pale for the base model, solid for my fine-tuned model. All scores run from 0 to 1 and come from my poem scorer on the raw output.',
    what: `The big change is format: ${pct(E.base.lines)} → ${pct(E['fine-tuned'].lines)} of poems have exactly 8 lines. Rhyme stays near zero (${E.base.rhyme.toFixed(2)} → ${E['fine-tuned'].rhyme.toFixed(2)}), and caption and mood words drop. The weighted total is slightly lower because it does not include the line score.`
  });
})();

(function lines() {
  const LC = P.line_counts; if (!LC) return;
  const ks = [...new Set([...Object.keys(LC.base), ...Object.keys(LC['fine-tuned'])].map(Number))].sort((a, b) => a - b);
  const nB = sum(Object.values(LC.base)), nT = sum(Object.values(LC['fine-tuned']));
  fig(slot('lines'), {
    title: 'How many lines each poem has', sub: 'Raw output, before the app trims anything.',
    legend: legend([['base', boxIcon('var(--base)')], ['fine-tuned', boxIcon('var(--accent)')]]),
    draw: (el, w) => vbars(el, w, { aria: 'Number of poems by line count, base versus fine-tuned', cats: ks.map(String), yn: nice(0, Math.max(...ks.map(k => Math.max(LC.base[k] || 0, LC['fine-tuned'][k] || 0))), 4),
      fy: v => String(Math.round(v)), series: [{ name: 'base', color: 'var(--base)', vals: ks.map(k => LC.base[k] || 0) }, { name: 'fine-tuned', color: 'var(--accent)', vals: ks.map(k => LC['fine-tuned'][k] || 0) }],
      xLabel: 'lines in poem', maxBw: 16, tip: ci => `<div class="th">${ks[ci]} lines</div><div class="tr">${sw('var(--base)')}base<b>${LC.base[ks[ci]] || 0}</b></div><div class="tr">${sw('var(--accent)')}fine-tuned<b>${LC['fine-tuned'][ks[ci]] || 0}</b></div>` }),
    table: () => table(['Lines', 'Base', 'Fine-tuned'], ks.map(k => [k, LC.base[k] || 0, LC['fine-tuned'][k] || 0]), [0, 1, 2]),
    how: 'Each pair of bars counts poems with that many lines. 8 is the target.',
    what: `The base model is all over the place: only ${LC.base[8] || 0} of ${nB} poems have 8 lines, and ${LC.base[12] || 0} have 12. After fine-tuning, ${LC['fine-tuned'][8] || 0} of ${nT} have exactly 8. The format is the clearest thing the model learned from my data.`
  });
})();

(function clip() {
  const C = P.clip; if (!C) return;
  const series = [['base', 'var(--base)'], ['fine-tuned', 'var(--accent)']];
  const all = [...C.base, ...C['fine-tuned']];
  fig(slot('clip'), {
    title: 'CLIP similarity between each poem and its image', sub: 'Cosine similarity of CLIP image and text embeddings. Higher = poem matches the image better.',
    draw: (el, w) => {
      const lw = 96, h = 150, m = { t: 10, b: 34 }, rowH = (h - m.t - m.b) / 2;
      const yn = nice(Math.min(...all), Math.max(...all), Math.max(2, Math.floor((w - lw) / 70)));
      const X = lin(yn.lo, yn.hi, lw, w - 12);
      let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Strip plot of CLIP similarity for base and fine-tuned poems">`;
      yn.ticks.forEach(t => { s += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${h - m.b}"/><text class="tick" x="${X(t)}" y="${h - m.b + 18}" text-anchor="middle">${t.toFixed(2)}</text>`; });
      series.forEach(([k, c], r) => {
        const cy = m.t + r * rowH + rowH / 2, vals = C[k], mu = mean(vals);
        s += `<text class="cl" x="${lw - 12}" y="${cy}" dy=".34em" text-anchor="end">${esc(k)}</text>`;
        vals.forEach((v, i) => { const j = ((i * 37) % 13) / 13 - .5; s += `<circle cx="${X(v).toFixed(1)}" cy="${(cy + j * rowH * .6).toFixed(1)}" r="2.6" style="fill:${c};fill-opacity:.6"/>`; });
        s += `<line x1="${X(mu)}" x2="${X(mu)}" y1="${cy - rowH * .42}" y2="${cy + rowH * .42}" style="stroke:var(--ink);stroke-width:2"/>`;
        const t = `<div class="th">${esc(k)}</div><div class="tr">mean<b>${mu.toFixed(3)}</b></div><div class="tr">min · max<b>${Math.min(...vals).toFixed(3)} · ${Math.max(...vals).toFixed(3)}</b></div><div class="tr">poems<b>${vals.length}</b></div>`;
        s += `<rect x="0" y="${cy - rowH / 2}" width="${w}" height="${rowH}" fill="transparent" ${tipAttr(t)} tabindex="0" aria-label="${esc(`${k}: mean ${mu.toFixed(3)}`)}"/>`;
      });
      el.innerHTML = s + '</svg>';
    },
    table: () => table(['Model', 'Mean', 'Min', 'Max', 'Poems'], series.map(([k]) => [k, mean(C[k]).toFixed(3), Math.min(...C[k]).toFixed(3), Math.max(...C[k]).toFixed(3), C[k].length]), [1, 2, 3, 4]),
    how: 'Each dot is one poem; the vertical line is the mean. Dots are spread up and down only so they do not overlap.',
    what: `The two clouds sit on top of each other (means ${mean(C.base).toFixed(3)} and ${mean(C['fine-tuned']).toFixed(3)}). Fine-tuning changed the form of the poems without making them drift away from the picture.`
  });
})();

(function compareTabs() {
  const list = $('#cmpTabs'), panels = $('#cmpPanels'), C = P.compare || [];
  if (!list || !C.length) { if (list) list.remove(); return; }
  list.innerHTML = C.map((c, i) => `<button type="button" role="tab" id="ct${i}" aria-controls="cp${i}" aria-selected="${i === 0}">Test image ${i + 1}</button>`).join('');
  panels.innerHTML = C.map((c, i) => `<div role="tabpanel" id="cp${i}" aria-labelledby="ct${i}" tabindex="0" ${i ? 'hidden' : ''}>
    <div class="cmp-head"><img src="${esc(c.thumb)}" alt="Test image: ${esc(c.caption)}" loading="lazy"><div><span class="lab">caption</span><p style="margin:0 0 6px">“${esc(c.caption)}”</p>${c.mood ? chip(c.mood) : ''}</div></div>
    <div class="compare">
      <div class="col"><h5>${sw('var(--base)')}base model · ${c.base_lines} lines</h5><p class="poem sm">${esc(c.base)}</p></div>
      <div class="col tuned"><h5>${sw('var(--accent)')}fine-tuned · ${c.tuned_lines} lines</h5><p class="poem sm">${esc(c.tuned)}</p></div>
    </div></div>`).join('');
  tablist(list);
})();

/* ================= samples, overview example, timing ================= */
const stepKey = k => /^cap/i.test(k) ? 'caption' : /^mood/i.test(k) ? 'mood' : /^poem/i.test(k) ? 'poem' : k;
(function gallery() {
  const g = $('#samples'); if (!g) return;
  if (!S.length) { g.innerHTML = '<p class="note">No samples yet. Run python -m src.make_samples.</p>'; return; }
  g.innerHTML = S.map(s => { const t = s.timing || {}, tot = sum(Object.values(t).map(Number));
    return `<article class="gcard" style="--gm:${mc(s.mood)}"><img src="${esc(s.image)}" alt="Sample photo: ${esc(s.caption)}" loading="lazy"><div class="body">
      <div class="meta">${chip(s.mood)}<span class="lab">${pct(s.confidence || 0)} confidence</span></div>
      <p class="cap">“${esc(s.caption)}”</p><p class="poem sm">${esc(s.poem)}</p>
      <p class="lab" style="margin:14px 0 0">${Object.entries(t).map(([k, v]) => `${esc(stepKey(k))} ${f2(v)} s`).join(' · ')} · total ${f2(tot)} s</p></div></article>`; }).join('');
})();

(function example() {
  const el = $('#example'), s = S.find(x => x.mood === 'calm') || S[0]; if (!el || !s) return;
  // Shown in the result panel until the first real result arrives.
  el.hidden = false;
  const hr = el.closest('.hero-r'); if (hr) hr.style.setProperty('--hero-mood', mc(s.mood));
  el.innerHTML = `<div><img src="${esc(s.image)}" alt="Example photo: ${esc(s.caption)}"><div class="lab" style="margin-top:6px">example output</div></div><div>
    <div class="r-head">${chip(s.mood)}<span class="lab">${pct(s.confidence || 0)} confidence</span></div>
    <p class="cap"><span class="lab">caption</span><br>“${esc(s.caption)}”</p>
    <p class="poem">${esc(s.poem)}</p>
    <p class="lab" style="margin:14px 0 0">Your poem appears here. Drop a photo on the left.</p></div>`;
})();

(function timing() {
  if (!S.length) return;
  const keys = ['caption', 'mood', 'poem'], col = { caption: 'var(--base)', mood: 'var(--ink-2)', poem: 'var(--accent)' };
  const T = S.map(s => { const o = {}; Object.entries(s.timing || {}).forEach(([k, v]) => { o[stepKey(k)] = Number(v); }); return o; });
  const totals = T.map(t => sum(keys.map(k => t[k] || 0)));
  fig(slot('timing'), {
    title: 'Where the time goes, per image', sub: 'The four sample images, on my laptop CPU. Seconds.',
    legend: legend(keys.map(k => [k === 'caption' ? 'caption (BLIP)' : k === 'mood' ? 'mood (CLIP + classifier)' : 'poem (fine-tuned Qwen)', boxIcon(col[k])])),
    draw: (el, w) => {
      const lw = 96, rowH = 34, bh = 16, h = S.length * rowH + 30, max = Math.ceil(Math.max(...totals));
      const X = lin(0, max, lw, w - 56);
      let s = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Stacked bars of time per step for each sample image">`;
      nice(0, max, 4).ticks.filter(t => t <= max).forEach(t => { s += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="0" y2="${h - 26}"/><text class="tick" x="${X(t)}" y="${h - 8}" text-anchor="middle">${t} s</text>`; });
      S.forEach((smp, i) => { const y = i * rowH + (rowH - bh) / 2; let x = lw;
        s += `<text class="cl" x="${lw - 12}" y="${y + bh / 2}" dy=".34em" text-anchor="end">${esc(smp.mood)}</text>`;
        keys.forEach(k => { const v = T[i][k] || 0, ww = X(v) - lw; s += `<rect x="${x.toFixed(1)}" y="${y}" width="${Math.max(0, ww - 1).toFixed(1)}" height="${bh}" style="fill:${col[k]}"/>`; x += ww; });
        s += `<text class="dl" x="${x + 6}" y="${y + bh / 2}" dy=".34em">${totals[i].toFixed(1)} s</text>`;
        const t = `<div class="th">${esc(smp.mood)} sample</div>` + keys.map(k => `<div class="tr">${sw(col[k])}${k}<b>${f2(T[i][k] || 0)} s</b></div>`).join('') + `<div class="tr">total<b>${f2(totals[i])} s</b></div>`;
        s += `<rect x="0" y="${i * rowH}" width="${w}" height="${rowH}" fill="transparent" ${tipAttr(t)} tabindex="0" aria-label="${esc(`${smp.mood}: total ${totals[i].toFixed(1)} seconds`)}"/>`; });
      el.innerHTML = s + '</svg>';
    },
    table: () => table(['Sample', 'Caption', 'Mood', 'Poem', 'Total'], S.map((s, i) => [s.mood, ...keys.map(k => f2(T[i][k] || 0) + ' s'), f2(totals[i]) + ' s']), [1, 2, 3, 4]),
    how: 'One bar per sample image, split into the three steps. The number at the end is the total.',
    what: `Writing the poem takes the longest (about ${mean(T.map(t => t.poem || 0)).toFixed(1)} s on average), captioning comes next, and the mood step, CLIP plus my classifier, takes about ${mean(T.map(t => t.mood || 0)).toFixed(2)} s. The first image is slower because the models warm up.`
  });
})();

/* ================= model cards ================= */
(function cards() {
  const el = $('#cards'); if (!el) return;
  const cv = n => byName[n] ? `CV ${pct(byName[n].cv_mean, 1)} ± ${pct(byName[n].cv_std, 1)}, test ${pct(byName[n].test.accuracy, 1)}` : '';
  const ti = P.train_info || {};
  // The six parts of the app, in pipeline order. pre = pretrained (dashed), mine = I trained / wrote it.
  const tp = ti.trainable_params || 8798208, tt = ti.total_params || 502830976;
  const C6 = [
    { n: '1', step: 'Step 1 · caption', name: 'Captioner: BLIP base', id: 'Salesforce/blip-image-captioning-base', kind: 'pre', tag: 'Pretrained · frozen',
      does: 'Looks at the photo and writes one short English sentence about it.',
      rows: [['Input → output', 'Photo (resized to 384 × 384) → caption, e.g. “the sun is setting over the ocean at sunset”.'],
        ['How it works', 'A Vision Transformer reads the image; a text decoder writes the caption word by word. Trained by Salesforce on millions of image-caption pairs.'],
        ['Settings', 'Beam search, <code>num_beams=3</code>, <code>max_new_tokens=30</code>. <code>src/caption.py</code>'],
        ['Size · speed', '~247 M parameters (990 MB) · 0.9–2.6 s per photo on CPU'],
        ['Why this one', 'Good captions from a model small enough for a laptop CPU. The caption gives the poem concrete things to write about.']] },
    { n: '2', step: 'Step 2 · mood', name: 'Image features: CLIP ViT-B/32', id: 'openai/clip-vit-base-patch32', kind: 'pre', tag: 'Pretrained · frozen',
      does: 'Turns each photo into 512 numbers (an embedding) that describe what it shows.',
      rows: [['Input → output', 'Photo (224 × 224) → 512 numbers, L2-normalised (length 1).'],
        ['How it works', 'OpenAI trained it on 400 M image-text pairs to match pictures with descriptions. I use only its image half.'],
        ['Settings', '<code>get_image_features</code>, batch 16, cached in <code>models/features.npz</code>. <code>src/features.py</code>'],
        ['Size · speed', '~151 M parameters (605 MB) · with my classifier 0.06–0.22 s per photo'],
        ['Why this one', `${M.n_images || 251} photos are far too few to learn vision from pixels. CLIP already knows “sunset” or “crowd”, so a small model on top can learn mood. This is transfer learning.`]] },
    { n: '3', step: 'Step 2 · mood', name: 'Mood classifier', id: `StandardScaler → PCA → ${M.best || 'KNN'} (scikit-learn)`, kind: 'mine', tag: 'Trained by me',
      does: 'Reads the 512 CLIP numbers and picks happy, sad, calm or energetic, with a probability for each.',
      big: best ? `${pct(best.cv_mean, 1)} <small class="lab">CV accuracy · test ${pct(best.test.accuracy, 1)}</small>` : '',
      rows: [['Input → output', '512 CLIP numbers → mood + 4 probabilities.'],
        ['How it works', `Scale every feature to mean 0, std 1 → PCA keeps ${M.pca ? M.pca.n90 : 98} directions (90 % of the variance) → KNN: the 5 most similar training photos vote; closer ones count more.`],
        ['Settings', '<code>PCA(n_components=0.90)</code>, <code>KNeighborsClassifier(n_neighbors=5, weights="distance")</code>. <code>src/train_mood.py</code>'],
        ['My work', `Collected and labelled ${M.n_images || 251} photos, compared 5 models with 5-fold cross-validation, picked the best.`],
        ['Why KNN', 'CLIP puts similar scenes close together, which is exactly what “find the nearest photos” needs. Best CV mean of the five.']] },
    { n: '4', step: 'Step 3 · poem', name: 'Poem writer: Qwen2.5-0.5B-Instruct', id: 'Qwen/Qwen2.5-0.5B-Instruct', kind: 'pre', tag: 'Pretrained · frozen',
      does: 'A small chat language model that writes the poem. Its own weights never change.',
      rows: [['Input → output', 'Short prompt with caption + mood → poem text, trimmed to 8 lines in two verses.'],
        ['How it works', 'Decoder-only transformer by the Qwen team (Alibaba): 24 layers, writes one token at a time.'],
        ['Settings', 'Temperature 0.9, top-p 0.9, repetition penalty 1.1, max 120 new tokens. <code>src/poem.py</code>'],
        ['Size · speed', '~494 M parameters (988 MB) · ~2.5 s per poem on CPU; the app writes 3 at once and keeps the best (~5.5 s)'],
        ['Why this one', 'The smallest instruction model that writes fluent English, small enough to fine-tune on my laptop.']] },
    { n: '5', step: 'Step 3 · poem', name: 'Poem adapter: LoRA', id: 'models/poet_lora/ · rank 16', kind: 'mine', tag: 'Fine-tuned by me',
      does: 'Small extra matrices added to Qwen that teach it my poem format. This is my fine-tune.',
      big: `${(tp / 1e6).toFixed(1)} M <small class="lab">trained numbers · ${pct(tp / tt, 2)} of the model</small>`,
      extra: `<p class="formula">W' = W + (α / r) · B · A &nbsp; r = 16, α = 32</p>`,
      rows: [['How it works', 'W stays frozen. LoRA learns two thin matrices A (16 × 896) and B (896 × 16); their product is added to W.'],
        ['Training', `${P.dataset ? P.dataset.train : 270} image→poem pairs I built · 3 epochs · lr 2e-4 · batch 4 × 2 · ${Math.round(ti.train_seconds || 349)} s on my Mac GPU. <code>src/train_poet.py</code>`],
        ['Targets', 'q, k, v, o, gate, up, down projections in all 24 layers.'],
        ['Why LoRA', 'Full fine-tuning would not fit in my laptop’s memory and would overfit 270 examples. The adapter is a 35 MB file, merged into Qwen when the app starts.']] },
    { n: '6', step: 'Data + evaluation', name: 'Poem scorer', id: 'src/poem_metrics.py · plain Python rules', kind: 'mine', tag: 'Written by me',
      does: 'Scores a poem from 0 to 1. Used to pick the training poems and to compare base vs fine-tuned.',
      extra: `<p class="formula">total = 0.4 · rhyme + 0.3 · keyword + 0.3 · mood − penalties</p>`,
      rows: [['Parts', '<b>rhyme</b>: share of line pairs that rhyme (CMU dictionary). <b>keyword</b>: share of caption words used. <b>mood</b>: words from a 16-word list per mood. <b>lines</b>: exactly 8? (reported apart).'],
        ['Penalties', '−0.3 repeated line · −0.2 line over 12 words · −0.3 talks about itself (“Here’s a poem”).'],
        ['Why', 'I needed a repeatable judge for 1,800 candidate poems. Every part can be checked by hand.'],
        ['Limit', 'It counts things; it cannot tell if a poem is beautiful.']] },
  ];
  el.innerHTML = C6.map(c => `<article class="mcard ${c.kind}">
    <div class="top"><span class="num">${esc(c.n)} · ${esc(c.step)}</span><span class="tag ${c.kind}">${esc(c.tag)}</span></div>
    <h3>${esc(c.name)}</h3><p class="id">${esc(c.id)}</p>
    <p class="does">${c.does}</p>${c.big ? `<div class="big">${c.big}</div>` : ''}${c.extra || ''}
    <dl>${c.rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl></article>`).join('');
})();

/* ================= the five classifiers, compact ================= */
(function classifierCards() {
  const el = $('#ccards'); if (!el || !models.length) { if (el) el.remove(); return; }
  const INFO = {
    'KNN': ['The 5 most similar photos vote; closer ones count more.', 'k = 5, distance weights'],
    'Logistic Regression': ['A weighted sum per mood, turned into probabilities.', 'C = 1, max_iter 2000'],
    'Random Forest': ['200 trees on random samples; majority vote.', '200 trees'],
    'MLP': ['Small neural net: 128 hidden units learn non-linear mixes.', '128 ReLU, alpha 1e-3'],
    'Decision Tree': ['A chain of yes/no questions on one feature at a time.', 'max_depth 8'],
  };
  el.innerHTML = models.slice().sort((a, b) => b.cv_mean - a.cv_mean).map(m => { const [how, set] = INFO[m.name] || ['', ''];
    return `<div class="ccard${m.name === M.best ? ' win' : ''}"><h4>${esc(m.name)}</h4><span class="lab">${m.name === M.best ? 'chosen · used in the app' : esc(set)}</span>
      <p>${esc(how)}</p><p class="acc">${pct(m.cv_mean, 1)} <small>CV ± ${pct(m.cv_std, 1)}</small></p><p class="lab" style="margin:2px 0 0">test ${pct(m.test.accuracy, 1)}</p></div>`; }).join('');
})();

/* ================= viva Q&A ================= */
(function qa() {
  const el = $('#qa'); if (!el) return;
  const E = P.eval || {}, b = best || {};
  const Q = [
    ['Did you train the poem model from scratch?', `No. The poem model is Qwen2.5-0.5B-Instruct, pretrained by the Qwen team. I fine-tuned it with LoRA: its 494 M weights stay frozen and I trained 8.8 M new adapter weights (1.75 %) on my own 270 image-to-poem pairs. Training a language model from scratch needs billions of words and many GPUs; fine-tuning is the realistic way to teach a small model a new style on a laptop.`],
    ['What exactly did you train yourself?', `The mood classifier, from scratch: the scaler, the PCA and five classifiers, on ${M.n_images || 251} images I collected and labelled. The LoRA adapter of the poem model. I also built the poem dataset and wrote the poem scorer. BLIP and CLIP are pretrained and frozen.`],
    ['Why use CLIP features instead of training a CNN on the images?', 'With about 60 images per mood a CNN would overfit badly. CLIP was trained on 400 million image-text pairs and already encodes scene concepts, so a simple classifier on its 512 numbers learns mood from few examples. This is transfer learning.'],
    ['Why PCA before the classifier?', `The 512 features are partly redundant. Keeping 90 % of the variance needs only ${M.pca ? M.pca.n90 : 98} components, which removes noise, speeds up training and helps distance-based KNN. It also shows a syllabus technique in a real use.`],
    ['How did you avoid data leakage?', 'The scaler, PCA and classifier are one scikit-learn Pipeline, so in cross-validation they are refitted on the training folds only. The test set is split off first and used once at the end. For the poem model the split is by image, and test images never get a training poem.'],
    ['Why KNN, when the MLP had a higher test accuracy?', `I choose on the cross-validation mean (5 scores on ${(M.split || {}).train || 200} images) and not on the test set (51 images, where one image is 2 points). KNN had the best CV mean, ${b.cv_mean ? pct(b.cv_mean, 1) : ''}. Choosing by test score would turn the test set into a second validation set and make the final number optimistic.`],
    ['Why do you report accuracy as plus or minus something?', `The ± is the standard deviation over the 5 folds (${b.cv_std ? pct(b.cv_std, 1) : ''} for KNN). It shows how much the score depends on which images land in the scoring fold. Differences smaller than that, like KNN vs Logistic Regression, are not meaningful.`],
    ['What does the confusion matrix tell you?', 'Most errors involve happy: happy photos get called calm or energetic. That matches the data, because a sunny beach can be happy or calm and a party can be happy or energetic. Calm and energetic are the easiest moods.'],
    ['What is LoRA and why only 1.75 % of the weights?', 'LoRA adds a low-rank update B·A next to each frozen weight matrix W, with rank r = 16. Only A and B are trained. Fine-tuning mostly needs a small change in behaviour, and a low-rank update can express that with a tiny fraction of the parameters, so it fits in memory and overfits less.'],
    ['Your fine-tuned model has a lower total score. Is it worse?', `In some ways, yes, and I report that. The total (${E.base ? E.base.total.toFixed(2) : ''} → ${E['fine-tuned'] ? E['fine-tuned'].total.toFixed(2) : ''}) rewards rhyme and word overlap, which dropped slightly. But the model learned the requested format (8 lines ${E.base ? pct(E.base.lines) : ''} → ${E['fine-tuned'] ? pct(E['fine-tuned'].lines) : ''}), is about 42 % faster, and CLIP similarity is unchanged. It did not learn rhyme because only about 8 % of the training poems rhymed.`],
    ['Why did validation loss go up after step 30?', 'Overfitting. After one epoch the model starts memorising the 270 training poems: training loss keeps falling, validation loss rises from 1.89 to 1.97. I saved the final adapter rather than the best one; next time I would use early stopping and keep the step-30 checkpoint.'],
    ['How do you know a poem fits the image?', 'Two automatic checks: the keyword score (how many caption words appear in the poem) and CLIP similarity between the poem text and the image. Neither measures quality, so the honest answer is: only roughly. A blind human rating would be the next step.'],
    ['Why does K-Means not find the four moods?', `Without labels, K-Means groups images by their strongest differences, mostly content such as people or landscapes, not feeling. The ARI of ${M.kmeans ? M.kmeans.ari_k4.toFixed(2) : ''} shows only partial agreement. The mood signal is in the features, but it takes labels to find it.`],
  ];
  el.innerHTML = Q.map(([q, a]) => `<details class="qa"><summary>${esc(q)}</summary><div class="a"><p>${esc(a)}</p></div></details>`).join('');
})();

/* ================= STUDIO ================= */
const studio = $('#studio'), drop = $('#drop'), input = $('#file'), log = $('#log'), offBox = $('#offline');
let online = null;
const STEPS = [['caption', 'Captioning the image (BLIP)'], ['mood', 'Reading the mood (CLIP + ' + (M.best || 'classifier') + ')'], ['poem', 'Writing the poem (fine-tuned Qwen)']];
const EST = { caption: 1.4, mood: 0.25 };    // typical seconds, only used to move the highlight while waiting

(function picks() {
  const p = $('#picks'); if (!p) return;
  if (!S.length) { p.remove(); return; }
  p.insertAdjacentHTML('beforeend', S.map((s, i) => `<button type="button" class="pick" data-sample="${i}" aria-label="Use the ${esc(s.mood)} sample image"><img src="${esc(s.image)}" alt="">${esc(s.mood)}</button>`).join(''));
  $$('[data-sample]', p).forEach(b => b.onclick = () => sendSample(S[+b.dataset.sample]));
})();

async function sendSample(s) {
  try { const r = await fetch(s.image); if (!r.ok) throw new Error(r.statusText); const b = await r.blob();
    send(new File([b], s.image.split('/').pop(), { type: b.type || 'image/jpeg' }));
  } catch (e) { showError('Could not load the sample image', 'Open this page through the demo server (http://127.0.0.1:7860) rather than as a file.'); }
}
function showError(title, msg) {
  const d = document.createElement('div'); d.className = 'err'; d.setAttribute('role', 'alert');
  d.innerHTML = `<strong>${esc(title)}</strong><span>${esc(msg)}</span>`;
  log.prepend(d); setTimeout(() => d.remove(), 12000);
}
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', body });
  if (!r.ok) { let msg = r.statusText; try { const j = await r.json(); msg = j.detail || msg; } catch (_) {} const err = new Error(msg); err.status = r.status; throw err; }
  return r.json();
}
function stepsHTML() { return `<ol class="steps">${STEPS.map(([k, l]) => `<li data-k="${k}"><span class="m" aria-hidden="true"></span><span>${esc(l)}</span><span class="t"></span></li>`).join('')}</ol><p class="elapsed" aria-hidden="true">0.0 s</p>`; }
function timingHTML(t) {
  const e = Object.entries(t || {}).map(([k, v]) => [stepKey(k), Number(v) || 0]), tot = sum(e.map(x => x[1]));
  return `<div class="timing">${e.map(([k, v]) => `<span>${esc(k)} <b>${f2(v)} s</b></span>`).join('')}<span>total <b>${f2(tot)} s</b></span></div>`;
}
function renderResult(card, d, imgURL) {
  const probs = d.probs || { [d.mood]: d.confidence };
  const hr = card.closest('.hero-r'); if (hr) hr.style.setProperty('--hero-mood', mc(d.mood));
  let timing = { ...(d.timing || {}) }, poem = d.poem;
  card.innerHTML = `<div class="thumb"><img src="${imgURL}" alt="Your image"><div class="lab">your image</div></div>
    <div class="body">
      <div class="r-head">${chip(d.mood)}<span class="lab">${pct(d.confidence || 0)} confidence</span></div>
      <div class="probs" role="list" aria-label="Mood probabilities">${MOODS.map(m => { const p = Number(probs[m]) || 0;
        return `<div class="prow" role="listitem"><span>${esc(m)}</span><span class="pt" aria-hidden="true"><span class="pb" style="width:${(p * 100).toFixed(1)}%;background:${mc(m)}"></span></span><span class="pv">${pct(p)}</span></div>`; }).join('')}</div>
      <p class="cap"><span class="lab">caption</span><br>“${esc(d.caption)}”</p>
      <p class="poem">${esc(poem)}</p>
      <div class="tstrip">${timingHTML(timing)}</div>
      <div class="actions"><button class="btn again" type="button">Another poem</button><button class="btn ghost cp" type="button">Copy poem</button></div>
    </div>`;
  const again = $('.again', card);
  $('.cp', card).onclick = async e => flash(e.currentTarget, (await copyText(poem)) ? 'Copied' : 'Copy failed');
  again.onclick = async () => {
    again.disabled = true; again.textContent = 'Writing…';
    const f = new FormData(); f.append('caption', d.caption); f.append('mood', d.mood);
    try {
      const r = await post('/api/regenerate', f); poem = r.poem;
      $('.poem', card).textContent = r.poem;
      Object.keys(timing).forEach(k => { if (/^poem/.test(k)) delete timing[k]; });
      timing = { ...timing, ...r.timing }; $('.tstrip', card).innerHTML = timingHTML(timing);
    } catch (e) {
      const off = e instanceof TypeError; if (off) setStatus(false);
      $('.tstrip', card).insertAdjacentHTML('beforeend', `<p class="note" role="alert">${off ? 'The demo server stopped responding.' : 'Error: ' + esc(e.message)}</p>`);
    }
    again.disabled = false; again.textContent = 'Another poem';
  };
  const tot = sum(Object.values(timing).map(Number)), k = $('#kpi-sec');
  if (k) { $('.v', k).innerHTML = `${tot.toFixed(1)}<small> s</small>`; $('.s', k).textContent = 'measured on your last image'; }
}
async function send(file) {
  if (!file) return;
  if (!/^image\//.test(file.type)) { showError('That file is not an image.', 'Use a JPG, PNG or WebP photo.'); return; }
  if (file.size > 20 * 1024 * 1024) { showError('That image is too large.', 'Please use an image under 20 MB.'); return; }
  if (location.protocol === 'file:') { setStatus(false); showError('The demo server is not running.', 'Start it with the command above, then open http://127.0.0.1:7860.'); return; }
  const url = URL.createObjectURL(file);
  const card = document.createElement('div'); card.className = 'result';
  card.innerHTML = `<div class="thumb"><img src="${url}" alt="Your image"></div><div class="body" role="status" aria-live="polite">${stepsHTML()}</div>`;
  log.prepend(card);
  const t0 = performance.now(), items = $$('.steps li', card), el = $('.elapsed', card);
  const paint = () => { const s = (performance.now() - t0) / 1000; el.textContent = s.toFixed(1) + ' s';
    const k = s < EST.caption ? 0 : s < EST.caption + EST.mood ? 1 : 2;
    items.forEach((li, i) => li.className = i < k ? 'done' : i === k ? 'active' : ''); };
  paint(); const timer = setInterval(paint, 100);
  const f = new FormData(); f.append('image', file);
  try {
    const data = await post('/api/poem', f);
    clearInterval(timer); setStatus(true);
    renderResult(card, data, url);
  } catch (e) {
    clearInterval(timer);
    const off = e instanceof TypeError || e.status === 404 || e.status === 405;
    if (off) setStatus(false);
    items.forEach(li => { if (li.className !== 'done') li.className = 'fail'; });
    $('.body', card).insertAdjacentHTML('beforeend', `<p class="note" role="alert">${off ? '<strong>The demo server is not running.</strong> Start it with the command shown above and try again.' : '<strong>Something went wrong:</strong> ' + esc(e.message)}</p>`);
  }
  if (card.getBoundingClientRect().top < 60) card.scrollIntoView({ block: 'start', behavior: reduce() ? 'auto' : 'smooth' });
  input.value = '';
}
if (drop) {
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => send(input.files[0]));
  let depth = 0;
  studio.addEventListener('dragenter', e => { e.preventDefault(); depth++; studio.classList.add('over'); });
  studio.addEventListener('dragover', e => e.preventDefault());
  studio.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) studio.classList.remove('over'); });
  studio.addEventListener('drop', e => { e.preventDefault(); depth = 0; studio.classList.remove('over'); const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) send(f); });
  document.addEventListener('paste', e => {
    if (e.target.closest && e.target.closest('input,textarea,[contenteditable]')) return;
    const it = [...((e.clipboardData && e.clipboardData.items) || [])].find(i => i.type.startsWith('image/'));
    if (it) { e.preventDefault(); send(it.getAsFile()); $('#try').scrollIntoView({ block: 'start' }); }
  });
}

/* ---------- server status ---------- */
const statusEl = $('#status'); let fails = 0;
function setStatus(ok, model) {
  online = ok; statusEl.classList.toggle('live', ok);
  $('.st', statusEl).textContent = ok ? 'Live' : 'Offline';
  statusEl.title = ok ? `Demo server running${model ? ', mood model ' + model : ''}` : 'Demo server not running: start it with python app.py';
  if (offBox) offBox.hidden = ok;
}
async function health() {
  if (location.protocol === 'file:') { setStatus(false); return; }
  try {
    const c = new AbortController(), t = setTimeout(() => c.abort(), 8000);
    const r = await fetch('/api/health', { cache: 'no-store', signal: c.signal }); clearTimeout(t);
    const j = await r.json(); fails = 0; setStatus(!!j.ok, j.model);
  } catch (e) { if (++fails >= 2 || online === null) setStatus(false); }
}
health(); setInterval(health, 15000);

/* ---------- theme ---------- */
const themeBtn = $('#theme');
const curTheme = () => document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
const syncTheme = () => themeBtn.setAttribute('aria-label', curTheme() === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
themeBtn.addEventListener('click', () => {
  const next = curTheme() === 'dark' ? 'light' : 'dark'; document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('theme', next); } catch (e) {}
  syncTheme();
});
syncTheme();

/* ---------- section links in the top bar + scrollspy ---------- */
const toc = $('#toc');
const links = Object.fromEntries($$('a', toc).map(a => [a.getAttribute('href').slice(1), a]));
let lastMark = null;
const mark = id => {
  if (id === lastMark) return; lastMark = id;
  Object.entries(links).forEach(([k, a]) => { if (k === id) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
  // keep the active link visible when the bar scrolls sideways (phones)
  const a = links[id]; if (a && toc.scrollWidth > toc.clientWidth) toc.scrollTo({ left: a.offsetLeft - toc.offsetLeft - 16, behavior: 'auto' });
};
const secs = $$('main > section');
let spyQueued = false;
const spy = () => {
  spyQueued = false;
  // current section = the last one whose top has passed 35 % of the window height
  const line = innerHeight * .35; let cur = secs[0];
  secs.forEach(s => { if (s.getBoundingClientRect().top <= line) cur = s; });
  if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) cur = secs[secs.length - 1];
  if (cur) mark(cur.id);
};
addEventListener('scroll', () => { if (!spyQueued) { spyQueued = true; requestAnimationFrame(spy); } }, { passive: true });
addEventListener('resize', spy);
spy();

/* ---------- tabs + copy buttons ---------- */
tablist($('#moreTabs'));
// glossary terms live in a tab: open that tab first, then jump to the term
document.addEventListener('click', e => {
  const a = e.target.closest && e.target.closest('a[href^="#g-"]'); if (!a) return;
  const t = document.getElementById(a.getAttribute('href').slice(1)); if (!t) return;
  e.preventDefault(); $('#mt-gloss').click();
  t.scrollIntoView({ block: 'center', behavior: reduce() ? 'auto' : 'smooth' });
});
document.addEventListener('click', async e => {
  const b = e.target.closest && e.target.closest('.copy'); if (!b) return;
  const code = $('code', b.closest('.code')); if (!code) return;
  const txt = code.textContent.split('\n').map(l => l.replace(/\s+#.*$/, '')).join('\n').trim();
  flash(b, (await copyText(txt)) ? 'Copied' : 'Failed');
});
})();
"""


# ---------------------------------------------------------------- build
def facts(data):
    """Every number used in the static HTML text, read from data.json."""
    m, p = data.get("mood") or {}, data.get("poet") or {}
    models = {x["name"]: x for x in m.get("models", [])}
    best = models.get(m.get("best")) or {}
    lr = models.get("Logistic Regression") or {}
    ev, ti, loss = p.get("eval") or {}, p.get("train_info") or {}, p.get("loss") or {}
    base, tuned = ev.get("base") or {}, ev.get("fine-tuned") or {}
    pc = lambda v, d=0: "—" if v is None else f"{v * 100:.{d}f}\u00a0%"
    f2 = lambda v: "—" if v is None else f"{v:.2f}"
    vals = [(s, v) for s, v in zip(loss.get("step", []), loss.get("val", [])) if v is not None]
    best_v = min(vals, key=lambda t: t[1]) if vals else (None, None)
    train = [v for v in loss.get("train", []) if v is not None]
    sc = p.get("scores") or {}
    samples = data.get("samples") or []
    secs = [sum(s.get("timing", {}).values()) for s in samples]
    tp, tot = ti.get("trainable_params"), ti.get("total_params")
    return {
        "N_IMAGES": m.get("n_images", "—"), "N_FEAT": m.get("n_features", 512), "BEST": m.get("best", "KNN"),
        "BEST_CV": pc(best.get("cv_mean"), 1), "BEST_STD": pc(best.get("cv_std"), 1),
        "BEST_TEST": pc((best.get("test") or {}).get("accuracy"), 1), "LR_CV": pc(lr.get("cv_mean"), 1),
        "TRAIN": (m.get("split") or {}).get("train", "—"), "TEST": (m.get("split") or {}).get("test", "—"),
        "N90": (m.get("pca") or {}).get("n90", "—"),
        "ARI": f2((m.get("kmeans") or {}).get("ari_k4")), "SIL4": f2(((m.get("kmeans") or {}).get("silhouette") or {}).get("4")),
        "P_TRAIN": (p.get("dataset") or {}).get("train", "—"), "P_VAL": (p.get("dataset") or {}).get("val", "—"),
        "P_TEST": (p.get("dataset") or {}).get("test", "—"), "N_KEPT": len(sc.get("kept", [])) or "—",
        "TRAINABLE": f"{tp / 1e6:.1f} M" if tp else "—", "LORA_PCT": pc(tp / tot, 2) if tp and tot else "—",
        "TRAIN_MIN": f"{ti['train_seconds'] / 60:.0f}" if ti.get("train_seconds") else "—",
        "LINES_B": pc(base.get("lines")), "LINES_T": pc(tuned.get("lines")),
        "RHYME_B": f2(base.get("rhyme")), "RHYME_T": f2(tuned.get("rhyme")),
        "KW_B": f2(base.get("keyword")), "KW_T": f2(tuned.get("keyword")),
        "MOOD_B": f2(base.get("mood")), "MOOD_T": f2(tuned.get("mood")),
        "CLIP_B": f"{base['clip']:.3f}" if "clip" in base else "—", "CLIP_T": f"{tuned['clip']:.3f}" if "clip" in tuned else "—",
        "TOTAL_B": f2(base.get("total")), "TOTAL_T": f2(tuned.get("total")), "N_POEMS": ev.get("n_poems", "—"),
        "VAL_BEST_STEP": best_v[0] if best_v[0] is not None else "—",
        "STEPS": loss["step"][-1] if loss.get("step") else "—",
        "SEC_PER_IMAGE": f"{sum(secs) / len(secs):.0f}" if secs else "4",
    }


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def term(match):
    """[[label|id]] -> link to the glossary entry, with the definition as a tooltip."""
    label, gid = match.group(1), match.group(2)
    title = GLOSSARY.get(gid, ("", ""))[1]
    return f'<a class="term" href="#g-{gid}"><abbr title="{esc(title)}">{label}</abbr></a>'


def build(data, built):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    toc = "".join(f'<li><a href="#{sid}"><span class="n">{i:02d}</span><span>{esc(t)}</span></a></li>'
                  for i, (sid, t) in enumerate(SECTIONS, 1))
    gloss = "".join(f'<dt id="g-{gid}"><dfn>{esc(t)}</dfn></dt><dd>{esc(d)}</dd>'
                    for gid, (t, d) in sorted(GLOSSARY.items(), key=lambda kv: kv[1][0].lower()))
    html = HTML
    for k, v in {**facts(data), "TOC": toc, "GLOSSARY_HTML": gloss, "BUILT": built}.items():
        html = html.replace("{{" + k + "}}", str(v))
    html = re.sub(r"\[\[([^|\]]+)\|([a-z]+)\]\]", term, html)
    # portfolio-style masthead: mono index + label on a hairline above each title
    labels = {f"{i:02d}": t for i, (_, t) in enumerate(SECTIONS, 1)}
    html = re.sub(r'<div class="sec-head"><span class="sec-num">(\d\d)</span>',
                  lambda m: f'<div class="sec-head"><p class="eyebrow"><span class="sec-num">{m.group(1)}</span>'
                            f'<i aria-hidden="true"></i><span>{esc(labels.get(m.group(1), ""))}</span></p>', html)
    css = CSS.replace("/*FONTS*/", font_css()).replace("/*DARK*/", DARK)
    return html.replace("/*CSS*/", css).replace("/*JS*/", JS).replace("__DATA__", payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=SITE / "data.json", help="input JSON (default: site/data.json)")
    ap.add_argument("--out", type=Path, default=SITE / "index.html", help="output HTML (default: site/index.html)")
    args = ap.parse_args()
    if args.data.exists():
        data = json.loads(args.data.read_text(encoding="utf-8"))
    else:
        print(f"  {args.data} not found: building with placeholders (run python -m src.site_data)")
        data = {}
    html = build(data, datetime.date.today().isoformat())
    left = sorted(set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", html)))
    if left:
        print("  warning: unfilled placeholders:", ", ".join(left))
    args.out.write_text(html, encoding="utf-8")
    missing = [f for *_, f in FONTS if not (SITE / "fonts" / f).exists()]
    print(f"wrote {args.out} ({len(html) // 1024} KB)" + (f" · missing fonts: {missing}" if missing else " · fonts: Geist + IBM Plex Serif (self-hosted)"))


if __name__ == "__main__":
    main()
