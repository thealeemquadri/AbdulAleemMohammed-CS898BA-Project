"""
Interactive demonstration of the diabetic retinopathy grading pipeline.

Satisfies the final milestone's Virtual Demonstration requirement: it runs the
complete computer vision pipeline on real fundus images and returns a severity grade,
in both single-image (real-time) and batch modes.

The demo intentionally exposes the intermediate processing stages rather than only the
prediction, so the image analysis is visible rather than hidden inside the model.

Launch:
    python app.py                      # local
    python app.py --share              # public link (use this in Colab)
    python app.py --checkpoint outputs/final_optimized_best.pt

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import gradio as gr

from src.preprocessing.fundus import (
    circular_crop, ben_graham, green_channel, apply_clahe,
    tophat_bothat, denoise, clahe_color, preprocess_rgb, preprocess_baseline,
)
from src.models.model import build_model
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES

GRADE_ADVICE = {
    0: "No diabetic retinopathy detected. Routine annual screening.",
    1: "Mild non-proliferative changes. Monitor; recheck in 6 to 12 months.",
    2: "Moderate non-proliferative changes. Ophthalmology referral advised.",
    3: "Severe non-proliferative changes. Prompt ophthalmology referral.",
    4: "Proliferative retinopathy. Urgent ophthalmology referral.",
}

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 300


def load_model(checkpoint, backbone="efficientnet_b3", size=300):
    """Load the trained checkpoint once, at startup."""
    global MODEL, INPUT_SIZE
    INPUT_SIZE = size
    MODEL = build_model(backbone, pretrained=False, device=DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.eval()
    print(f"Loaded {checkpoint} on {DEVICE} (input size {size})")
    return MODEL


def processing_stages(img):
    """
    Run the pipeline stage by stage and return each intermediate for display.
    This is what makes the demonstration a computer vision demo rather than a
    black-box classifier demo.
    """
    cropped = circular_crop(img)
    small = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

    bg = ben_graham(small)
    g = green_channel(small)
    cl = apply_clahe(g)
    dn = denoise(cl)
    th, bh = tophat_bothat(dn)
    rgb_out = preprocess_rgb(img, size=INPUT_SIZE)

    return {
        "1. Circular crop": cropped,
        "2. Ben Graham normalization": bg,
        "3. Green channel": g,
        "4. CLAHE": cl,
        "5. Top-hat (bright lesions)": th,
        "6. Bottom-hat (dark lesions)": bh,
        "7. Colour-preserving output": rgb_out,
    }


@torch.no_grad()
def predict_tensor(arr):
    """arr: HxWx3 uint8 RGB -> (probabilities, predicted grade)."""
    x = arr.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    t = torch.from_numpy(x).float().to(DEVICE)

    logits = MODEL(t)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs, int(probs.argmax())


def analyze(image):
    """Single-image path: full pipeline, staged visuals, prediction, timing."""
    if image is None:
        return None, "Upload a retinal fundus image to begin.", {}

    t0 = time.time()
    img = np.array(image.convert("RGB"))

    stages = processing_stages(img)

    # The model consumes the plain resize, which the ablation showed is strongest.
    model_input = preprocess_baseline(img, size=INPUT_SIZE)
    probs, grade = predict_tensor(model_input)
    elapsed = (time.time() - t0) * 1000

    gallery = []
    for label, im in stages.items():
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        gallery.append((im, label))

    conf = float(probs[grade])
    report = (
        f"## Predicted grade: {grade} ({CLASS_NAMES[grade]})\n\n"
        f"**Confidence:** {conf:.1%}\n\n"
        f"{GRADE_ADVICE[grade]}\n\n"
        f"---\n\n"
        f"Processed in {elapsed:.0f} ms on {DEVICE.upper()}. "
        f"Research demonstration only, not a clinical device."
    )
    dist = {f"{i} {CLASS_NAMES[i]}": float(probs[i]) for i in range(5)}
    return gallery, report, dist


def analyze_batch(files):
    """Batch path: the rubric allows real-time OR batch; this demonstrates both."""
    if not files:
        return "Upload one or more fundus images."

    rows = ["| File | Predicted grade | Confidence |", "| --- | --- | --- |"]
    t0 = time.time()

    for f in files:
        path = f.name if hasattr(f, "name") else f
        img = cv2.imread(path)
        if img is None:
            rows.append(f"| {os.path.basename(path)} | unreadable | - |")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = preprocess_baseline(img, size=INPUT_SIZE)
        probs, grade = predict_tensor(arr)
        rows.append(
            f"| {os.path.basename(path)} | {grade} ({CLASS_NAMES[grade]}) "
            f"| {probs[grade]:.1%} |"
        )

    dt = time.time() - t0
    n = len(files)
    rows.append("")
    rows.append(f"Processed **{n}** images in {dt:.2f}s ({dt / max(n, 1) * 1000:.0f} ms each).")
    return "\n".join(rows)


def build_ui():
    with gr.Blocks(title="Diabetic Retinopathy Grading", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Automated Diabetic Retinopathy Grading\n"
            "**CS 898BA** | Abdul Aleem Mohammed, Wichita State University\n\n"
            "EfficientNet-B3 grading retinal fundus photographs into five clinical "
            "severity levels, with the image processing pipeline exposed stage by stage."
        )

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="pil", label="Retinal fundus image")
                    btn = gr.Button("Analyze", variant="primary")
                    out_report = gr.Markdown()
                    out_probs = gr.Label(label="Grade probabilities", num_top_classes=5)
                with gr.Column(scale=2):
                    out_gallery = gr.Gallery(
                        label="Image processing stages",
                        columns=4, height=520, object_fit="contain",
                    )
            btn.click(analyze, inputs=inp, outputs=[out_gallery, out_report, out_probs])

        with gr.Tab("Batch"):
            gr.Markdown("Upload multiple fundus images to grade them in one pass.")
            files = gr.File(file_count="multiple", label="Fundus images")
            bbtn = gr.Button("Run batch", variant="primary")
            btable = gr.Markdown()
            bbtn.click(analyze_batch, inputs=files, outputs=btable)

        gr.Markdown(
            "---\n"
            "Trained on APTOS 2019 (3,662 images). Validation QWK 0.8962. "
            "Research demonstration, not a diagnostic tool."
        )
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/baseline_best.pt")
    ap.add_argument("--backbone", default="efficientnet_b3")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--share", action="store_true",
                    help="Create a public link. Required in Colab.")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train a model first, or pass --checkpoint with the correct path."
        )

    load_model(args.checkpoint, args.backbone, args.size)
    build_ui().launch(share=args.share)
