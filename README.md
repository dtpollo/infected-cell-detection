# 🦟 Malaria Cell Image Classifier

Deep learning project for binary classification of malaria-infected cell images using multiple neural network architectures, ensemble learning, and mobile deployment via TFLite.

---

## 📁 Repository Structure

```
malaria-cell-classifier/
├── Clas_Celulas_Malaria.ipynb     # Main notebook — full pipeline
├── requirements.txt                # Python dependencies
├── malaria_cnn_aug.tflite         # TFLite model for Android deployment
├── malariaApp.apk                 # Android app — real-time cell inference
├── malaria_dataset/
│   ├── Parasitized/               # Infected cell images (.png)
│   └── Uninfected/                # Healthy cell images (.png)
└── Model Weights/
    ├── M1_FNN.weights.h5          # Feed Forward Network weights
    ├── M2_CNN.weights.h5          # CNN weights
    ├── M3_CNN_aug.weights.h5      # CNN + Augmentation weights
    ├── M4_vgg_rf.pkl              # VGG16 + Random Forest (sklearn)
    ├── M4_vgg_extractor.pkl       # VGG16 feature extractor
    ├── M5_vgg_FNN.weights.h5      # VGG16 + Dense Network weights
    ├── ensemble_weights.json      # Optimal ensemble weights (Top 3)
    └── results_test.json          # Final test metrics for all models
```

---

## 📊 Dataset

- **1,067 images** — 500 Parasitized (label=1), 567 Uninfected (label=0)
- Source: [NIH Malaria Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- RGB format, resized to 64×64 (CNN models) or 224×224 (VGG16 models)
- Split: **Train 64% / Validation 16% / Test 20%** with stratification

---

## 🧠 Models

| Model | Input Size | Normalization | Size |
|---|---|---|---|
| Feed Forward Network | 64×64 | /255.0 | 36.51 MB |
| CNN | 64×64 | /255.0 | 7.87 MB |
| CNN + Augmentation | 64×64 | /255.0 | 7.87 MB |
| VGG16 + Random Forest | 224×224 | preprocess_input | 0.38 MB |
| VGG16 + Dense Network | 224×224 | preprocess_input | 74.01 MB |
| **Ensemble (Top 3)** | — | — | **89.74 MB** |

---

## 📈 Results (Test Set)

| Model | Accuracy | Precision | AUC |
|---|---|---|---|
| **Ensemble** | **0.9579** | 0.9505 | **0.9932** |
| VGG16 + RF | 0.9393 | 0.9579 | 0.9901 |
| VGG16 + DL | 0.9439 | 0.9400 | 0.9900 |
| CNN | 0.9533 | 0.9500 | 0.9872 |
| CNN + Aug | 0.9486 | 0.9588 | 0.9868 |
| Feed Forward | 0.5327 | 0.0000 | 0.5049 |

> **AUC** was used as the primary metric — threshold-independent and more reliable than accuracy for medical binary classification.

---

## 🏆 Ensemble Learning

Top 3 models selected by validation AUC and combined via weighted average:

```
ŷ = 0.5 × P(CNN+Aug) + 0.0 × P(CNN) + 0.5 × P(VGG16+DL)
```

Optimal weights found by grid search over validation set (66 valid combinations).

---

## 📱 Mobile Deployment

CNN+Augmentation was selected for mobile deployment:

- Exported to **TFLite**: 7.87 MB → **2.61 MB** (−66.8%)
- Deployed on **Android** using Kotlin + Android Studio
- Real-time inference on device
- Input: 64×64 RGB image, normalized /255.0

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/dtpollo/malaria-cell-classifier
cd malaria-cell-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Clas_Celulas_Malaria.ipynb
```

> ⚠️ Make sure the `Model Weights/` folder is in the same directory as the notebook before running inference cells.

---

## 📦 Requirements

See `requirements.txt`. Main dependencies:
- TensorFlow 2.x
- scikit-learn
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn

---

## 👤 Author

**Daniel Troya** — Yachay Tech University

[![GitHub](https://img.shields.io/badge/github-dtpollo-black?logo=github)](https://github.com/dtpollo)
