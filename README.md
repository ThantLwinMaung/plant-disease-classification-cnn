# Plant Disease Classification with Deep Learning

Detecting **38 plant-disease categories** from leaf photographs using TensorFlow/Keras on the PlantVillage dataset. Four models compared on a held-out test set that is never used for training or selection.

**Author:** Thant Lwin Maung · GitHub · Kaggle

## Project Overview

Crop disease detected too late is a major driver of yield loss for smallholder farmers across South and Southeast Asia. A model that recognises disease from a phone photo of a leaf turns any handset into a first-line diagnostic tool — enabling earlier intervention, targeted pesticide use, and reduced yield loss.

This is a multi-class image-classification problem: given one RGB leaf image, predict one of 38 categories (crop × disease, including healthy classes).

## Dataset

- Source: PlantVillage (Kaggle: mohitsingh1804)
- 38 classes across Tomato, Apple, Corn, Potato, Grape, Peach, Orange, Soybean, Squash, and more
- 43,444 training images · 10,861 validation images · 256×256 RGB, resized to 224×224
- Imbalance: 36.4× (largest vs smallest class), handled with inverse-frequency class weights

> **Known Limitation**
>
> PlantVillage images use clean, uniform backgrounds under controlled lighting. Models degrade on real field photos, so these numbers are an upper bound on field performance — not a deployment guarantee.

## Methodology

- Real train / val / test split — the official val/ folder is stratified-split 50/50; the test set is never seen until final scoring
- Single evaluation harness — every model scored the same way; no metrics hardcoded
- Data augmentation — rotation, zoom, shift, flip (training stream only)
- Class weights — to counter the measured imbalance
- Callbacks — EarlyStopping, ModelCheckpoint(save_best_only), ReduceLROnPlateau

## Models Compared

- Baseline CNN — from-scratch reference; parameter-heavy (23.9M) and prone to overfitting
- Improved CNN — adds BatchNorm, Global Average Pooling, Dropout, augmentation; only ~137K params
- MobileNetV2 — ImageNet transfer learning; small and fast, best fit for on-device use
- EfficientNetB0 — heavier ImageNet backbone; highest accuracy of the four

## Results

All metrics on the held-out test set (5,431 images), macro-averaged.

| Model | Test Accuracy | F1 (macro) | Parameters |
|---|---|---|---|
| EfficientNetB0 | 96.23% | 0.9533 | 4,098,249 |
| MobileNetV2 | 94.83% | 0.9384 | 2,306,662 |
| Improved CNN | 91.00% | 0.8927 | 136,934 |
| Baseline CNN | 87.52% | 0.8324 | 23,912,294 |

### Key findings

- Transfer learning gives the best accuracy and is most efficient relative to performance
- The Improved CNN is the smallest model (~17× smaller than MobileNetV2) yet still reaches 91%
- Hardest cases are within-crop confusions, concentrated in Tomato diseases

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy · Pandas
- Matplotlib · Seaborn
- scikit-learn

## Reproducibility

- Python 3.12.13 · TensorFlow 2.19.0 · Kaggle GPU (2× Tesla T4)
- Fixed seeds (42); minor run-to-run variation expected on GPU
- To reproduce: add the dataset, set accelerator to GPU T4 ×2, and Run All

## Future Work

- Fine-tune transfer-learning backbones (unfreeze top blocks at a low learning rate)
- Grad-CAM explainability to confirm the model attends to lesions, not background
- Field-realistic data to close the domain-shift gap
- Confidence calibration and an "unknown / consult expert" path
- TFLite quantisation and a lightweight inference app
- Deduplication audit between train and val/test (PlantVillage near-duplicate check)

## License

MIT
