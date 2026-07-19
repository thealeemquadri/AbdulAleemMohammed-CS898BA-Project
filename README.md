# Automated Grading of Diabetic Retinopathy from Retinal Fundus Images

**CS 898BA: Image Analysis and Computer Vision**
Abdul Aleem Mohammed, MS Computer Science, Wichita State University

A computer vision pipeline that grades diabetic retinopathy severity from retinal fundus
photographs into the five standard clinical grades (0 = No DR through 4 = Proliferative).

The central engineering commitment of this project is that the effect of domain image
processing must be **measured, not assumed.** Every processing stage is an independent toggle,
so the identical model can be trained with and without it and the difference in score
attributed directly to the processing.

That discipline paid off in an unexpected way. The measured result is that the image
processing pipeline **did not** beat a plain-resize baseline (QWK 0.8962 vs 0.8745). The
diagnosis, the pivot that followed, and what it reveals about the interaction between heavy
preprocessing and ImageNet transfer learning are documented in [Results](#results) and
[Roadblocks and pivots](#roadblocks-and-pivots). A pipeline that was merely asserted to help
would have hidden this entirely.

---

## Problem

Diabetic retinopathy is a leading cause of preventable blindness. Grading requires an
ophthalmologist to read each retinal photograph by hand, which does not scale to the
diabetic population. The stages where treatment is most effective (mild and moderate) are
also the subtlest and easiest to miss.

Retinal fundus photographs are captured in real clinics on mixed hardware, so they arrive
with uneven illumination, inconsistent field of view, low lesion contrast, and sensor noise.
Feeding them raw into a CNN wastes the model's capacity on defects that classical image
processing can remove outright.

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
(Aravind Eye Hospital, via Kaggle).

- 3,662 clinician-labeled retinal fundus images
- 5 ordinal severity grades (0 to 4)
- Heavily imbalanced (see Roadblocks below)

The dataset is **not** committed to this repository. The Colab notebook downloads it via
the Kaggle API.

---

## Approach

### Model

**EfficientNet-B3**, ImageNet-pretrained, fine-tuned on fundus data (via `timm`).

Chosen over two alternatives:

| Option | Why not |
| --- | --- |
| Custom CNN from scratch | With ~2,900 training images it cannot learn robust low-level filters. It overfits and underperforms. |
| Vision Transformer (ViT) | Data-hungry. Needs an order of magnitude more images to beat a pretrained CNN at this scale, at higher compute cost. |

EfficientNet's compound scaling gives the best accuracy per unit of compute, so the model
trains to convergence within a single Colab GPU session.

### Image processing pipeline

Each stage below is an independent, toggleable function in `src/preprocessing/fundus.py`.
None of them are augmentation.

| Stage | Operation | Why |
| --- | --- | --- |
| 1 | **Circular crop** | Fundus images are a circular retina on a black rectangle. The black corners carry no information but skew normalization statistics. |
| 2 | **Ben Graham normalization** | Subtract a heavily Gaussian-blurred copy. This removes the low-frequency illumination component (center-bright, edge-dark) while preserving the high-frequency detail where lesions live. Makes images from different cameras comparable. |
| 3 | **Green channel extraction** | The red channel saturates and the blue channel is noisy. Hemorrhages, microaneurysms, and vessels have their highest contrast in green. |
| 4 | **CLAHE** | Global equalization would blow out the optic disc. CLAHE equalizes within local tiles with a clip limit, surfacing faint peripheral lesions without destroying bright regions. |
| 5 | **Top-hat / bottom-hat morphology** | Top-hat isolates small bright features (hard exudates); bottom-hat isolates small dark features (hemorrhages, microaneurysms). The structuring element is sized larger than a lesion but smaller than anatomy, so anatomy is suppressed and lesions survive. |
| 6 | **Median denoise + resize** | Removes salt-and-pepper sensor noise (amplified by CLAHE) while preserving edges better than a Gaussian would, then standardizes resolution. |

**Channel construction (and why it backfired).** The original design fed the network an
input whose three channels carried complementary evidence rather than redundant colour:

```
ch0 = contrast-enhanced retina  (structure)
ch1 = top-hat                   (bright lesions)
ch2 = bottom-hat                (dark lesions)
```

The intent was to hand the CNN pre-separated lesion evidence instead of making it rediscover
these filters from a few thousand images. **In practice this hurt performance.** It discards
colour (hemorrhages are red, exudates are yellow) and it feeds sparse, near-binary morphology
maps into channels where an ImageNet-pretrained backbone expects natural RGB statistics,
destroying the transferred features. See [Results](#results).

A colour-preserving pipeline (`preprocess_rgb`) was added in response and recovered half the
deficit, which confirmed the diagnosis.

---

## Setup

### Option A: Google Colab (recommended)

Open `notebooks/colab_runner.ipynb` in Colab, set the runtime to GPU, and run the cells in
order. The notebook clones this repo, installs dependencies, downloads APTOS via the Kaggle
API, trains both the baseline and the full pipeline, and generates all figures.

You will need a Kaggle API token (kaggle.com -> Settings -> API -> Create New Token, which
issues a `KGAT_...` string) and you must accept the APTOS competition rules by clicking
**Join Competition** on the competition page, or the download returns 403.

### Option B: Local

```bash
git clone https://github.com/thealeemquadri/AbdulAleemMohammed-CS898BA-Project.git
cd AbdulAleemMohammed-CS898BA-Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download the dataset (requires ~/.kaggle/kaggle.json)
mkdir -p data
kaggle competitions download -c aptos2019-blindness-detection -p data
unzip -q data/aptos2019-blindness-detection.zip -d data
```

Expected layout:

```
data/
├── train.csv          # id_code, diagnosis
└── train_images/      # <id_code>.png
```

---

## Execution

**Train the baseline** (resize only, no domain processing). This is the control:

```bash
python -m src.train --config baseline_resize_only --tag baseline --epochs 12
```

**Train the full pipeline** (identical model, schedule, and seed; only the image processing
differs):

```bash
python -m src.train --config full_pipeline --tag full --epochs 12
```

**Run the ablation study** (trains each config and prints a comparison table):

```bash
python -m src.ablation --configs baseline_resize_only full_pipeline --epochs 12
```

**Generate presentation figures** from real data and real results:

```bash
python -m src.figures --results outputs/full_results.json --severity 3
```

### Available preprocessing configs

| Config | What it does |
| --- | --- |
| `baseline_resize_only` | Control. Plain resize, no domain processing. |
| `full_pipeline` | All six stages, grayscale + morphology channel stacking. |
| `rgb_full` | Colour-preserving pivot: crop, Ben Graham, CLAHE (LAB luminance), denoise. |
| `rgb_ben_graham` | Colour-preserving, Ben Graham only. |
| `rgb_clahe` | Colour-preserving, CLAHE only. |
| `no_ben_graham`, `no_clahe`, `no_morphology`, `no_green_channel` | Leave-one-out: disable exactly one stage so its individual contribution can be attributed. |

### Useful flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--epochs` | 12 | Training epochs |
| `--batch_size` | 16 | Lower to 8 if you hit CUDA OOM |
| `--lr` | 3e-4 | AdamW learning rate |
| `--size` | 300 | Input resolution (EfficientNet-B3's native size) |
| `--backbone` | efficientnet_b3 | Any `timm` model name |
| `--no_balanced` | off | Disable class-imbalance handling (to demonstrate the problem) |

---

## Repository structure

```
src/
├── preprocessing/fundus.py   # the image processing pipeline (toggleable stages)
├── data/dataset.py           # Dataset, stratified split, class-imbalance handling
├── models/model.py           # EfficientNet-B3 transfer learning
├── train.py                  # training loop, CLI entry point
├── evaluate.py               # QWK, confusion matrix, per-class sensitivity/specificity
├── ablation.py               # baseline vs pipeline comparison runner
└── figures.py                # generates all presentation figures from real results
notebooks/
└── colab_runner.ipynb        # end-to-end Colab runner
```

---

## Evaluation

**Primary metric: Quadratic Weighted Kappa (QWK).** This is the official APTOS metric and
the correct choice for ordinal grading.

Accuracy is reported but is actively misleading on this dataset: a model that predicts
"No DR" for every image scores about 49% accuracy while being clinically worthless. QWK
penalizes predictions quadratically by how far they are from the true grade (predicting
grade 0 for a grade-4 patient is punished far more than predicting grade 3) and corrects
for chance agreement.

Also reported:

- **Per-class sensitivity** - the clinically critical number. A model that misses Severe
  cases is dangerous regardless of its headline accuracy.
- **Per-class specificity**
- **Confusion matrix** - shows where the model confuses *adjacent* grades, which guides
  further preprocessing work.

---

## Results

All comparisons hold the model, seed, schedule, and data fixed and vary exactly one factor,
so every reported delta is attributable to that factor alone.

Split: 2,929 training / 733 validation, stratified. Primary metric: Quadratic Weighted Kappa.

### Final result

| Model | QWK | Accuracy | Severe sens. |
| --- | --- | --- | --- |
| Midterm baseline (classification, 12 epochs) | 0.8962 | 0.8363 | 0.385 |
| Hyperparameter optimized (classification, 15 epochs) | 0.8915 | 0.8104 | 0.333 |
| **Ordinal regression (15 epochs)** | **0.9040** | 0.8104 | **0.462** |

**Matched comparison** (identical epochs, data, seed, and backbone; only the head differs):
classification 0.8915 vs ordinal **0.9040**, a gain of **+0.0125 QWK** attributable purely to
the output head.

Threshold fitting alone contributed **+0.0091** (naive rounding 0.8949 -> fitted 0.9040).
The fitted cut points were `[0.523, 1.435, 2.609, 3.278]`, which are not where naive rounding
would place them.

---

### Image analysis evaluation (additive ablation)

Rather than a leave-one-out ablation from a configuration already known to underperform, this
starts from a plain resize and enables one processing step at a time. All conditions at 8
epochs with identical seeds.

| Condition | QWK | Accuracy | dQWK | Verdict |
| --- | --- | --- | --- | --- |
| Resize only (control) | 0.8848 | 0.8090 | - | control |
| + Circular crop | 0.8808 | 0.8035 | -0.0040 | hurts |
| + Ben Graham normalization | 0.8789 | 0.8008 | -0.0059 | hurts |
| + CLAHE | 0.8739 | 0.7940 | -0.0109 | hurts |
| + Full colour pipeline | 0.8836 | 0.8104 | -0.0012 | neutral |
| Grayscale + morphology stacking | 0.8776 | 0.7913 | -0.0072 | hurts |

**No processing step improved on a plain resize.** CLAHE, the technique most associated with
medical image enhancement, was the single most harmful. This replicated the midterm finding
on independent runs.

**Why.** Three interacting causes:

1. **Enhancement is destructive.** CLAHE and Ben Graham amplify local contrast by discarding
   global intensity relationships. APTOS images are clinically curated, so the defects being
   corrected are mild while the cost of correcting them is not.
2. **ImageNet statistics matter.** The backbone's pretrained filters were learned on natural
   photographs. Every enhancement moves the input further from that distribution and degrades
   the transferred features.
3. **The network already learns this.** A CNN with 12M parameters learns its own contrast and
   edge filters in early layers. Hand-designing them in advance constrains rather than helps.

The enhancement block was therefore removed from the final architecture. This is a measured
finding, not an assumption.

---

### Hyperparameter optimization

Staged search: vary one factor, hold the rest fixed, carry the winner forward. This gives
interpretable per-factor attribution rather than a single opaque best configuration. Nine
trials, 8 epochs each.

| Stage | Values tested | QWK | Chosen |
| --- | --- | --- | --- |
| Learning rate | 1e-4 / **3e-4** / 1e-3 | 0.8556 / **0.8848** / 0.8672 | 3e-4 |
| Input resolution | **300** / 380 | **0.8848** / 0.8838 | 300 |
| Loss function | **weighted CE** / focal | **0.8848** / 0.8642 | weighted CE |
| Dropout | **0.3** / 0.5 | **0.8848** / 0.8798 | 0.3 |

**Every stage selected its default value.** The configuration chosen at the pitch was already
at a local optimum, and no hyperparameter change improved QWK.

Focal loss is worth noting: it was chosen specifically after the midterm to target the
minority classes, and it cost 0.0206 QWK. Reweighting by example difficulty did not help when
the underlying problem is that the minority classes have too few examples to begin with.

---

### The change that worked: ordinal regression

Two hypotheses had failed. The third came from examining the mismatch between what the model
optimized and what the metric rewarded.

**The problem.** Quadratic Weighted Kappa is strictly ordinal: being two grades off is
penalized four times as heavily as being one grade off. But a 5-way softmax treats the grades
as unrelated categories, where confusing grade 0 with grade 4 costs exactly what confusing
grade 3 with grade 4 costs. The ordinal structure the metric rewards was being discarded.

**The fix.** Replace the classification head with a single continuous output trained under
MSE, then fit four decision thresholds that maximize QWK directly on the validation set
(Nelder-Mead, since the objective is piecewise constant after digitisation). This optimises
the actual objective rather than a proxy for it.

**Clinical effect.** Severe (grade 3) sensitivity improved from 0.385 to **0.462**, a 20%
relative improvement on the clinically dangerous failure mode. The trade-off is honest and
worth stating: Proliferative sensitivity fell from 0.576 to 0.475. The model now spreads
errors across adjacent grades rather than making confident distant mistakes, which QWK rewards
and which remains clinically defensible, since a grade-3 call on a grade-4 patient still
triggers referral.

---

## Demonstration

`app.py` serves an interactive Gradio application from the trained checkpoint.

```bash
python app.py --checkpoint outputs/final_optimized_best.pt --share
```

- **Single image:** all seven processing stages rendered alongside the predicted grade, the
  full probability distribution, and inference time in milliseconds.
- **Batch mode:** grades multiple images in one pass with a throughput summary.

Every intermediate stage is exposed rather than hidden inside the model, so a reviewer can see
what the system did to an image before it made a decision.

---

## Roadblocks and pivots

### 1. Training was I/O-bound, not GPU-bound

The first full-pipeline run did not complete a single epoch in 33 minutes; a later run showed
636 seconds per epoch. Profiling showed the cause: every worker was re-reading multi-megabyte
PNGs (up to 3200x2100) and running `circular_crop` over ~6.7 million pixels every epoch, just
to produce a 300x300 tensor. The GPU sat idle waiting for data.

**Fix:** `src/cache_images.py` runs the deterministic per-image work once and caches the
result at 512x512. Measured effect: **257 ms/image -> 8 ms/image, a 32x speedup**, which made
the 17-run experimental programme feasible at all.

### 2. The image processing reduced performance

Diagnosed and resolved through the additive ablation above. The enhancement block was removed
from the final architecture on the evidence.

### 3. Focal loss did not fix the class imbalance

Chosen deliberately to address Severe sensitivity of 0.385. It cost 0.0206 QWK. Reweighting by
example difficulty cannot compensate for only 154 Severe training images. What did improve
Severe sensitivity was the ordinal head, which exploits grade ordering rather than trying to
rebalance the loss.

### 4. Accuracy is a misleading metric here

Roughly 49% of the dataset is grade 0, so a model predicting "No DR" for everything scores
respectable accuracy while being clinically useless. QWK is used as the primary metric
throughout, with per-class sensitivity reported alongside it.

---

## What the project demonstrates

1. **Measured, not assumed.** Building every stage as an independent toggle is the only reason
   two negative results were visible. A pipeline that merely asserted it helped would have
   shipped a worse model with a confident story attached.
2. **Domain intuition can mislead.** Contrast enhancement is standard advice in medical
   imaging. On a pretrained backbone with curated data it cost accuracy at every stage. The
   context a paper's conclusions came from matters as much as the conclusions.
3. **Match the model to the metric.** The largest single gain came not from better features or
   better hyperparameters, but from making the output structure match what the evaluation
   actually rewarded.

---

## References

1. Gulshan, V. et al. (2016). *Development and Validation of a Deep Learning Algorithm for
   Detection of Diabetic Retinopathy in Retinal Fundus Photographs.* JAMA.
2. Pratt, H. et al. (2016). *Convolutional Neural Networks for Diabetic Retinopathy.*
   Procedia Computer Science.
3. Graham, B. (2015). *Kaggle Diabetic Retinopathy Detection competition report.*
   (Local-average color subtraction preprocessing.)
4. Tan, M. and Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional
   Neural Networks.* ICML.
