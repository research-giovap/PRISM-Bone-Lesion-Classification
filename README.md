# PRISM: PET/CT Radiomics Interpretable Self-attention Model

PRISM is a multicenter self-attention transformer ensemble for distinguishing **Unspecific Bone Uptakes (UBUs)** from **Prostate Cancer (PCa) bone metastases** on [¹⁸F]PSMA-1007 PET/CT, using high-dimensional radiomics and clinical features extracted from lesion segmentations.

---

## TRIPOD+AI Compliance & Reproducibility

This repository is structured to strictly comply with [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) reporting guidelines. To ensure zero data leakage and support external validation:

- **Strict modular isolation.** Every cross-validation fold contains its own isolated PyTorch weights (`model.pt`), scaling parameters (`scaler.pkl`), and explicit hyperparameter/feature definitions (`metadata.json`). Training and validation distributions are kept strictly separate.
- **Dynamic slicing and normalization.** The inference script reads fold-specific feature configurations and applies the fitted standardizer per fold. Test data is scaled using only the statistics from that fold's original training phase.

---

## Repository Structure

```
PRISM-Bone-Lesion-Classification/
│
├── README.md
├── requirements.txt
├── data/
│   └── test_data.csv          # Held-out test data (to be provided by the user)
│
├── models/                    # Level-1 base learners
│   ├── transformer_20pct/     # Ensemble using the top 20% stable features
│   │   ├── fold_1/
│   │   │   ├── model.pt       # Fold-specific PyTorch weights
│   │   │   ├── scaler.pkl     # Fold-specific fitted StandardScaler
│   │   │   └── metadata.json  # Fold-specific features & hyperparameters
│   │   └── fold_2/ ...        # (repeated for all 5 folds)
│   │
│   └── transformer_40pct/     # Ensemble using the top 40% stable features
│       └── fold_1/ ...
│
└── src/
    ├── model.py               # TabularTransformer PyTorch architecture
    └── inference.py           # Level-2 PRISM ensembling logic
```

---

## Environment Setup

Python 3.8+ is required. PyTorch, Pandas, NumPy, and Scikit-Learn are the core dependencies.

**1. Clone the repository**

```bash
git clone https://github.com/YourUsername/PRISM-Bone-Lesion-Classification.git
cd PRISM-Bone-Lesion-Classification
```

**2. Create and activate a virtual environment**

On macOS/Linux:
```bash
python -m venv prism_env
source prism_env/bin/activate
```

On Windows:
```bash
python -m venv prism_env
prism_env\Scripts\activate
```

**3. Install dependencies**

Create a `requirements.txt` in the root directory:

```
torch>=1.13.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Save your independent test data as a CSV file at `data/test_data.csv`. The file must contain:

- A unique identifier column (e.g., `Lesion_ID` or `Patient_ID`)
- Radiomics and clinical feature columns matching the names referenced in each fold's `metadata.json`
- A ground-truth target column (e.g., `Label`) with binary values: `1` for UBU (benign), `0` for PCa metastasis

> The inference script automatically reads `metadata.json` to select the relevant features per fold. Extra columns (e.g., metabolic baseline values, unrelated identifiers) are ignored.

The CSV must use **semicolons as delimiters** to match the training data format. Feature names follow PyRadiomics conventions for CT and PET modalities, covering original, LoG-filtered, and wavelet-decomposed image types (shape, first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM feature classes).

---

## Running Inference

Navigate to the `src/` directory and run the inference script:

```bash
cd src
python inference.py
```

**Under the hood:**

1. **Fold-specific initialization.** For each of the 5 folds in both the 20% and 40% transformer cohorts, the script reads `metadata.json` to identify the exact feature subset and hyperparameters.
2. **Strict scaling.** The fold-specific `scaler.pkl` is applied to standardize test data using that fold's training distribution only.
3. **Level-1 base predictions.** Each `model.pt` generates probabilistic predictions for the 20% and 40% stability subsets independently.
4. **Level-2 PRISM ensembling.** A soft-voting mechanism averages the multi-threshold predictions into a final `PRISM_Probability`.

Results are saved to `data/prism_predictions.csv`.

---

## Output Format

| Column | Description |
|---|---|
| `Patient_ID` | Unique lesion/patient identifier |
| `Prob_Transformer_20` | Level-1 probability from the 20% stable feature subset |
| `Prob_Transformer_40` | Level-1 probability from the 40% stable feature subset |
| `PRISM_Probability` | Final diagnostic probability from the Level-2 ensemble |
| `PRISM_Prediction` | Predicted class label (0 or 1) |
| `True_Label` | Ground-truth clinical label |

**Diagnostic thresholding at 0.50:**

- `PRISM_Probability ≥ 0.50` → Unspecific Bone Uptake (UBU) — benign, positive class
- `PRISM_Probability < 0.50` → PCa Bone Metastasis — malignant, negative class

---

## Model Architecture

PRISM uses a two-level stacking ensemble.

**Level 1 — TabularTransformer base learners.** Each feature token (one per radiomic feature) is independently embedded via a shared linear projection, prepended with a learnable `[CLS]` token, and passed through a standard Transformer encoder with sinusoidal positional encoding. The `[CLS]` token output is normalized and projected to a single logit, trained with binary cross-entropy loss. Two independent 5-fold cross-validated ensembles are trained: one on the top 20% most stable features (by ICC across folds), one on the top 40%.

**Level 2 — Soft voting.** The 5-fold averaged probabilities from the 20% and 40% ensembles are averaged to produce the final `PRISM_Probability`.

---

## Citation

If you use this repository or the PRISM framework in your research, please cite the corresponding paper (citation details to be added upon publication).

---

## License

MIT License

Copyright (c) 2026 Giovanni Pasini - Institute of Bioimaging and Complex Biological Systems, National Research Council (IBSBC-CNR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
