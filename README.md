# Automated Grading of Diabetic Retinopathy from Retinal Fundus Images

**CS 898BA: Image Analysis and Computer Vision**
Abdul Aleem Mohammed, MS Computer Science, Wichita State University

A computer vision pipeline that grades diabetic retinopathy severity from retinal fundus
photographs into the five standard clinical grades (0 = No DR through 4 = Proliferative).

The central engineering claim of this project is that **domain-specific image processing,
not data augmentation, is what makes the model work.** The codebase is built to prove that
claim with measurements rather than assert it: every processing stage is an independent
toggle, so the same model can be trained with and without it and the difference in score
attributed directly to the processing.

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

**Channel construction.** Rather than feeding three redundant color channels, the pipeline
builds an input whose channels carry complementary evidence:

```
ch0 = contrast-enhanced retina  (structure)
ch1 = top-hat                   (bright lesions)
ch2 = bottom-hat                (dark lesions)
```

This hands the CNN pre-separated lesion evidence instead of forcing it to rediscover these
filters from only a few thousand images.

---

## Setup

### Option A: Google Colab (recommended)

Open `notebooks/colab_runner.ipynb` in Colab, set the runtime to GPU, and run the cells in
order. The notebook clones this repo, installs dependencies, downloads APTOS via the Kaggle
API, trains both the baseline and the full pipeline, and generates all figures.

You will need a Kaggle API token (`kaggle.json`: kaggle.com → Settings → API → Create New
Token) and you must accept the APTOS competition rules on the competition page, or the
download returns 403.

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

`baseline_resize_only`, `full_pipeline`, `no_ben_graham`, `no_clahe`, `no_morphology`,
`no_green_channel`

The `no_*` configs are leave-one-out: they disable exactly one stage so its individual
contribution can be attributed.

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

All runs use an identical model (EfficientNet-B3, ImageNet pretrained), identical seed,
schedule, and 300x300 input size. The **only** variable is the image processing, so any
difference in QWK is attributable to it.

Split: 2,929 training / 733 validation, stratified.

### Baseline vs image processing pipeline

| Configuration | QWK | Accuracy | Severe sens. | Proliferative sens. | dQWK |
| --- | --- | --- | --- | --- | --- |
| **Baseline** (resize only) | **0.8962** | 0.8363 | 0.385 | 0.576 | - |
| Full pipeline (gray + morphology) | 0.8745 | 0.8090 | 0.410 | 0.508 | -0.0217 |
| RGB pivot (colour-preserving) | 0.8852 | 0.7913 | - | - | -0.0110 |

### The headline finding

**The image processing pipeline did not beat the baseline.** This was the opposite of the
hypothesis, and it is the most informative result of the project so far.

### Diagnosis

The original pipeline fed the network a 3-channel tensor constructed as:

```
ch0 = grayscale enhanced retina
ch1 = top-hat    (sparse, near-black)
ch2 = bottom-hat (sparse, near-black)
```

Two problems with this:

1. **Colour was discarded.** Hemorrhages are red and hard exudates are yellow. Extracting
   only the green channel throws away genuine diagnostic signal.
2. **ImageNet compatibility was broken.** EfficientNet's pretrained filters expect natural
   RGB image statistics. Feeding sparse morphology maps into channels 1 and 2 destroys the
   transferred features, which are the entire justification for choosing transfer learning
   over a from-scratch CNN.

Notably, Ben Graham's original Kaggle-winning method **preserved colour.** The illumination
normalization was implemented correctly; the way it was packaged for the network was not.

### The pivot

A colour-preserving pipeline was built (`preprocess_rgb`): CLAHE applied only to the
luminance channel in LAB space so hue is untouched, Ben Graham applied in colour, and the
morphology channel-stacking removed.

**The deficit halved, from -0.0217 to -0.0110.** This confirms the channel construction was
a real cause. However, the pipeline still does not beat raw RGB, which indicates the
enhancement itself is also removing information the pretrained backbone was using.

### What this means

On a strong ImageNet-pretrained backbone with a relatively clean dataset, aggressive
preprocessing can hurt more than it helps. This is a real and defensible finding, not a bug.

The remaining performance is **not** in preprocessing. Per-class sensitivity shows the model
catches No DR almost perfectly (0.96) but misses roughly 6 in 10 Severe cases (0.385). With
only 154 Severe images in training, class imbalance, not contrast enhancement, is the binding
constraint.

### Reproduce

```bash
python -m src.train --config baseline_resize_only --tag baseline --epochs 12
python -m src.train --config full_pipeline        --tag full     --epochs 12
python -m src.train --config rgb_full             --tag rgb_full --epochs 12
```

---

## Roadblocks and pivots

### 1. Training was CPU-bound, not GPU-bound

The first full-pipeline run did not complete a single epoch in 33 minutes. Ben Graham, CLAHE,
and morphology were being applied at full sensor resolution (up to 3200x2100) before
downscaling to the network's 300x300 input, so the GPU sat idle waiting on the data loader.

**Fix:** move the resize ahead of the expensive operations and raise the number of data loader
workers. Epoch time dropped to roughly 3 minutes, with no loss of the lesion features the model
uses.

### 2. The image processing reduced performance

Covered in detail above. Diagnosed as colour loss plus broken ImageNet input statistics;
pivoted to a colour-preserving pipeline, which recovered half the deficit and confirmed the
diagnosis.

### 3. Accuracy is a misleading metric here

Roughly 49% of the dataset is grade 0. A model predicting "No DR" for everything scores
respectable accuracy while being clinically useless. This drove the adoption of Quadratic
Weighted Kappa as the primary metric, with per-class sensitivity reported alongside it.

### 4. Severe class imbalance

Class counts in training: `[1444, 296, 799, 154, 236]`. Addressed with two independent
mechanisms: a `WeightedRandomSampler` (rebalances what the model sees) and class-weighted
cross-entropy (rebalances what each mistake costs). Splitting is stratified so the 154 Severe
images are not concentrated in one split.

These help but do not solve it. Severe sensitivity remains 0.385, and this is the primary
target for the final milestone.

---

## Next steps

1. **Leave-one-out ablation.** Isolate each processing step (`no_clahe`, `no_ben_graham`,
   `no_morphology`, `no_green_channel`) to test whether any single stage helps in isolation,
   rather than judging the pipeline only as a block.
2. **Attack the imbalance directly.** Focal loss, targeted oversampling, and ordinal regression
   (the grades are ordered; plain cross-entropy ignores that).
3. **Hyperparameter optimization.** Learning rate, input resolution, and backbone depth, each
   measured against the 0.8962 baseline.
4. **Live demo.** A hosted interface that runs the pipeline on an uploaded fundus image and
   returns a predicted grade.

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
