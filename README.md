# Context-Aware Image Captioning with Scene Graphs

> Does conditioning a caption decoder on an explicit scene graph — structured
> `subject → predicate → object` triples produce better captions than image features alone?

Submitted as course final project for **CSCI E-25: Computer Vision**, Harvard Extension School, Spring 2026.

[Full project report](./SG_captioning.md)

---

## Results

| Metric | Image-only baseline | **Scene-graph fusion** | Improvement |
|--------|--------------------|-----------------------|-------------|
| BLEU-4 | 0.0597 | **0.0665** | +11.4% |
| BLEU-1 | 0.6222 | **0.6386** | +2.6% |
| METEOR | 0.4319 | **0.4474** | +3.6% |
| ROUGE-L | 0.2857 | **0.2924** | +2.3% |

Evaluated on a 9,180-image held-out validation set (beam-4 decoding).
Visual predicate classifier: **top-1 55.5% / top-5 89.8% / top-10 95.8%** on the VG150 medium split.

---

## Overview

The pipeline has three stages:

1. **YOLO object detector** — fine-tuned on VG150 bounding boxes to detect the 150 curated
   object classes.
2. **Visual predicate classifier** — a ResNet-50 ROI feature extractor over the union of
   subject + object bounding boxes, concatenated with 64-d class embeddings and an 8-d
   spatial vector, fed to a 3-layer MLP. Trained with inverse-square-root frequency
   weighting to handle the long predicate tail.
3. **Vision-conditioned caption decoder** — a `ViT-base/16` image encoder and a T5-small
   encoder for the scene-graph text, concatenated in encoder-output space, decoded by T5
   with cross-attention to the joint sequence. Ablating `sg_input_ids=None` gives the
   image-only baseline.

A classical CV preprocessing library (CLAHE, bilateral denoising, unsharp masking,
white balance, adaptive gamma) is built and benchmarked with PSNR, SSIM, and
colorfulness metrics.

### Dataset

[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) under the
[VG150 curated subset](https://arxiv.org/abs/2305.18668) (150 object classes, 50 predicates).
Caption ground truth is **hybrid**: MS-COCO captions for the 43,887 VG images with a
`coco_id` link; best-scored VG region descriptions as fallback.

---

## Setup

**Requirements:** Python 3.13+, `uv`, a CUDA-capable GPU (tested on RTX 3060 Ti 8 GB
and Google Colab A100/V100/T4).

```bash
git clone https://github.com/nthapaliya/scene-graph-captioning.git
cd scene-graph-captioning

# Install dependencies with uv
uv python pin 3.13
uv sync
```

### Download the dataset

```bash
bash download.bash
```

This fetches:
- Visual Genome images (~15 GB, two zips) to `data/VG_100K/`
- VG-SGG annotations (`VG-SGG.h5`, `VG-SGG-dicts.json`, `image_data.json`)
- COCO 2014 captions

> **Note:** total disk requirement is approximately 20 GB once extracted.

---

## Running the notebook

```bash
uv run jupyter lab
```

Open `SG_captioning.ipynb` and run all cells top to bottom.
The notebook auto-detects the available GPU tier and scales batch sizes accordingly.

`SG_captioning.html` is created from the notebook with code cells hidden so as to read more like a report.

> `jupyter nbconvert --to markdown SG_captioning.ipynb`


---

## Key findings

Scene-graph context yields a **small but consistent improvement across all metrics** over
the image-only baseline. BLEU-4 sees the largest relative lift (+11.4%), suggesting
the SG-conditioned model commits to specific multi-word phrases ("riding a skateboard")
rather than generic templates.

The dominant failure mode is upstream detector error: when YOLO misclassifies an object,
the captioner follows the wrong label. End-to-end joint training would likely close this gap.

---

## References

- Krishna et al. (2017). [Visual Genome](https://arxiv.org/abs/1602.07332)
- Tang et al. (2020). [Unbiased Scene Graph Generation](https://arxiv.org/abs/2002.11949)
- Dosovitskiy et al. (2020). [ViT: An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)
- Raffel et al. (2019). [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- Redmon & Farhadi. [YOLO: You Only Look Once](https://arxiv.org/abs/1506.02640)
