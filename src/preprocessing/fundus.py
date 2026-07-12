"""
Fundus image processing pipeline for diabetic retinopathy grading.

Every step is an independent, toggleable function. This is deliberate: it lets us
run controlled ablations that measure exactly how much each image processing step
contributes to model performance, rather than asserting that it helps.

None of these are data augmentation. They are domain-specific image analysis steps
chosen because retinal fundus photographs have known, characteristic defects:
uneven illumination, inconsistent field of view, low lesion contrast, and sensor noise.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Step 1: Circular crop / border removal
# ----------------------------------------------------------------------
def circular_crop(img, tol=7):
    """
    Fundus photographs are a circular retina on a black rectangle. The black
    corners carry no information but do skew normalization statistics and waste
    model capacity. We threshold to find the retinal disc and crop to its bounding box.

    Args:
        img: RGB uint8 image.
        tol: intensity below which a pixel is considered background.
    Returns:
        Cropped RGB image containing only the retinal disc.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    mask = gray > tol
    if mask.sum() == 0:  # fully black image, nothing to crop
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


# ----------------------------------------------------------------------
# Step 2: Ben Graham illumination normalization
# ----------------------------------------------------------------------
def ben_graham(img, sigma_scale=30.0, alpha=4.0, beta=-4.0, gamma=128.0):
    """
    Ben Graham's method (2015 Kaggle DR winner). Retinal photos are lit unevenly:
    bright at the center, dark at the periphery, and inconsistent between cameras.

    Subtracting a heavily Gaussian-blurred copy of the image removes the low-frequency
    illumination component while preserving high-frequency detail, which is exactly
    where the lesions live. This makes images from different cameras comparable.

    Args:
        img: RGB uint8 image.
        sigma_scale: blur sigma is width / sigma_scale.
    Returns:
        Illumination-normalized RGB uint8 image.
    """
    sigma = img.shape[1] / sigma_scale
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, alpha, blurred, beta, gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Step 3: Green channel extraction
# ----------------------------------------------------------------------
def green_channel(img):
    """
    In RGB fundus images the red channel is saturated and the blue channel is noisy.
    Hemorrhages, microaneurysms, and vessels have their highest contrast against the
    retinal background in the GREEN channel. This is standard in retinal imaging.

    Returns a single-channel uint8 image.
    """
    return img[:, :, 1]


# ----------------------------------------------------------------------
# Step 4: CLAHE
# ----------------------------------------------------------------------
def apply_clahe(gray, clip_limit=3.0, tile=8):
    """
    Contrast Limited Adaptive Histogram Equalization.

    Global histogram equalization would blow out the bright optic disc. CLAHE
    equalizes within local tiles and clips the histogram to limit noise amplification,
    so faint lesions in dark peripheral regions become visible without destroying
    the bright regions.

    Args:
        gray: single-channel uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


# ----------------------------------------------------------------------
# Step 5: Morphological top-hat and bottom-hat
# ----------------------------------------------------------------------
def tophat_bothat(gray, ksize=15):
    """
    Morphological lesion isolation using an elliptical structuring element.

    - TOP-HAT (image minus its opening) isolates small BRIGHT features on a darker
      background. In fundus images these are hard exudates.
    - BOTTOM-HAT (closing minus image) isolates small DARK features on a lighter
      background: hemorrhages and microaneurysms.

    The structuring element is sized larger than a lesion but smaller than anatomical
    structures, so anatomy is suppressed and lesions survive. This is genuine
    morphological image analysis, not augmentation.

    Returns:
        (tophat, bothat) single-channel uint8 images.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bothat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return tophat, bothat


# ----------------------------------------------------------------------
# Step 6: Denoise
# ----------------------------------------------------------------------
def denoise(gray, ksize=3):
    """
    Median filtering removes salt-and-pepper sensor noise while preserving edges
    better than a Gaussian blur would. Applied after contrast enhancement, since
    CLAHE amplifies noise along with signal.
    """
    return cv2.medianBlur(gray, ksize)


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
def preprocess(
    img,
    size=300,
    use_crop=True,
    use_ben_graham=True,
    use_green=True,
    use_clahe=True,
    use_morphology=True,
    use_denoise=True,
):
    """
    Run the configurable fundus processing pipeline.

    Every stage can be switched off independently. Passing all flags False yields the
    BASELINE condition: a plain resize, which is what a naive transfer-learning setup
    would use. Comparing that against the full pipeline is our ablation study.

    Channel construction:
        When morphology is enabled we build a 3-channel image whose channels carry
        complementary evidence rather than redundant color:
            ch0 = contrast-enhanced retina (structure)
            ch1 = top-hat   (bright lesions / exudates)
            ch2 = bottom-hat (dark lesions / hemorrhages)
        This hands the CNN pre-separated lesion evidence instead of making it
        rediscover these filters from only 3,662 images.

    Args:
        img: RGB uint8 image (H, W, 3).
        size: output square resolution.
    Returns:
        RGB uint8 image of shape (size, size, 3).
    """
    if use_crop:
        img = circular_crop(img)

    # Resize BEFORE the expensive operations. Running Ben Graham, CLAHE, and
    # morphology at full sensor resolution (up to 3200x2100) wastes enormous CPU
    # time when the network only ever sees a size x size input. Downscaling first
    # is far faster and preserves the lesion features the model relies on.
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if use_ben_graham:
        img = ben_graham(img)

    if use_green:
        gray = green_channel(img)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if use_clahe:
        gray = apply_clahe(gray)

    if use_denoise:
        gray = denoise(gray)

    if use_morphology:
        tophat, bothat = tophat_bothat(gray)
        out = np.stack([gray, tophat, bothat], axis=-1)
    else:
        # No morphology: replicate the single channel so the tensor shape stays (H, W, 3)
        # and the ImageNet-pretrained backbone still accepts it.
        out = np.stack([gray, gray, gray], axis=-1)

    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out.astype(np.uint8)


def preprocess_baseline(img, size=300):
    """
    BASELINE condition: resize only, no domain processing.
    This is the control we measure the full pipeline against.
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)


# Named configurations used by the ablation study (src/ablation.py).
# Each turns off exactly one stage so we can attribute the change in QWK to it.
ABLATION_CONFIGS = {
    "baseline_resize_only": dict(
        use_crop=False, use_ben_graham=False, use_green=False,
        use_clahe=False, use_morphology=False, use_denoise=False,
    ),
    "full_pipeline": dict(
        use_crop=True, use_ben_graham=True, use_green=True,
        use_clahe=True, use_morphology=True, use_denoise=True,
    ),
    "no_ben_graham": dict(
        use_crop=True, use_ben_graham=False, use_green=True,
        use_clahe=True, use_morphology=True, use_denoise=True,
    ),
    "no_clahe": dict(
        use_crop=True, use_ben_graham=True, use_green=True,
        use_clahe=False, use_morphology=True, use_denoise=True,
    ),
    "no_morphology": dict(
        use_crop=True, use_ben_graham=True, use_green=True,
        use_clahe=True, use_morphology=False, use_denoise=True,
    ),
    "no_green_channel": dict(
        use_crop=True, use_ben_graham=True, use_green=False,
        use_clahe=True, use_morphology=True, use_denoise=True,
    ),
}


# ======================================================================
# PIVOT (added after the first ablation)
# ----------------------------------------------------------------------
# The first full_pipeline run UNDERPERFORMED the resize-only baseline
# (QWK 0.8745 vs 0.8962). Diagnosis:
#
#   1. Discarding color. Green-channel-only throws away clinically real signal:
#      hemorrhages are red, hard exudates are yellow.
#   2. Breaking ImageNet compatibility. Stacking [gray, top-hat, bottom-hat]
#      feeds sparse, near-binary morphology maps into channels the pretrained
#      EfficientNet expects to contain natural RGB statistics. That destroys the
#      transfer-learned filters, which are the entire justification for the
#      architecture choice.
#
# Note that Ben Graham's original Kaggle-winning method PRESERVED color. This
# pivot applies the same enhancements while keeping the image in RGB.
# ======================================================================

def clahe_color(img, clip_limit=3.0, tile=8):
    """
    Apply CLAHE to the LUMINANCE channel only, in LAB colour space.

    Applying CLAHE per-RGB-channel independently shifts hue and creates colour
    casts. Converting to LAB, equalising only L, and converting back enhances
    local contrast while leaving the colour information (a, b) intact.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def preprocess_rgb(img, size=300, use_crop=True, use_ben_graham=True,
                   use_clahe=True, use_denoise=True):
    """
    Colour-preserving processing pipeline.

    Identical intent to preprocess(), but the output stays a natural 3-channel RGB
    image, so the ImageNet-pretrained backbone still receives the input statistics
    it was trained on.
    """
    if use_crop:
        img = circular_crop(img)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if use_ben_graham:
        img = ben_graham(img)

    if use_clahe:
        img = clahe_color(img)

    if use_denoise:
        img = cv2.medianBlur(img, 3)

    return img.astype(np.uint8)


# Configs that route through preprocess_rgb (flagged with rgb=True).
RGB_CONFIGS = {
    "rgb_ben_graham": dict(rgb=True, use_crop=True, use_ben_graham=True,
                           use_clahe=False, use_denoise=True),
    "rgb_clahe": dict(rgb=True, use_crop=True, use_ben_graham=False,
                      use_clahe=True, use_denoise=True),
    "rgb_full": dict(rgb=True, use_crop=True, use_ben_graham=True,
                     use_clahe=True, use_denoise=True),
}

ABLATION_CONFIGS.update(RGB_CONFIGS)
