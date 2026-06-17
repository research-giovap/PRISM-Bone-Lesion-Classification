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
│   └── test_data.csv                 # Held-out test data (to be provided by the user)
│
├── models/                           # Level-1 base learners
│   ├── transformer_20pct/            # Ensemble using the top 20% stable features
│   │   ├── fold_1/
│   │   │   ├── model.pt              # Fold-specific PyTorch weights
│   │   │   ├── scaler.pkl            # Fold-specific fitted StandardScaler (joblib)
│   │   │   └── metadata.json         # Fold-specific features & hyperparameters
│   │   └── fold_2/ ...               # (repeated for all 5 folds)
│   │
│   └── transformer_40pct/            # Ensemble using the top 40% stable features
│       └── fold_1/ ...
│
└── src/
    ├── model.py                      # TabularTransformer architecture + attention extraction
    └── inference.py                  # Inference, attention mapping, and evaluation
```

---

## Environment Setup

Python 3.8+ is required.

> **Important:** `scikit-learn` is pinned to `==1.6.1` because the `scaler.pkl` files are serialized with that exact version. Installing a different version will trigger deserialization warnings or errors.

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

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
torch>=1.13.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn==1.6.1
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

---

## Data Preparation

Save your independent test data as a semicolon-delimited CSV file at `data/test_data.csv`. The file must contain:

- A `Lesion_ID` column with unique identifiers per lesion
- Radiomics and clinical feature columns matching the names in each fold's `metadata.json`
- A `label` column with values `1` (PCa metastasis) or `2` (UBU); the script remaps these automatically to `0` and `1` respectively

Feature names follow PyRadiomics conventions for CT and PET modalities, covering original, LoG-filtered (σ = 0.5–5.0 mm), and wavelet-decomposed image types, across shape, first-order, GLCM, GLRLM, GLSZM, GLDM, and NGTDM feature classes. Hyphens in feature names are normalized to underscores on load.

> Extra columns are ignored. The inference script slices only the features listed in each fold's `metadata.json`.

---

## Running Inference

Navigate to the `src/` directory and run:

```bash
cd src
python inference.py
```

**What happens:**

1. **Data loading.** `test_data.csv` is read with semicolon delimiter; hyphens in column names are replaced with underscores; raw labels `{1, 2}` are remapped to `{0, 1}`.
2. **Level-1 base predictions (×2).** For each of the 5 folds in the 20% and 40% transformer ensembles, the script loads `metadata.json`, applies the fold-specific `scaler.pkl`, runs `model.pt`, and extracts per-layer and rollout attention weights.
3. **Attention aggregation.** Attention maps are averaged across folds per lesion, then across lesions for global summaries. Per-lesion heatmaps (top-15 features by attention mass) and global heatmaps are saved to `data/attention_maps/`.
4. **Level-2 PRISM ensembling.** The 5-fold averaged probabilities from the 20% and 40% ensembles are averaged by soft voting.
5. **Evaluation.** Point-estimate metrics (AUC, Precision, Recall, F1, Accuracy) are computed on the full test set; 95% confidence intervals are derived from 1000 bootstrap resamples (seed=42).
6. **Output.** Results are written to `data/`.

---

## Output Files

| File | Contents |
|---|---|
| `data/prism_predictions.csv` | Per-lesion probabilities and predicted labels |
| `data/prism_performance_metrics.csv` | AUC, Precision, Recall, F1, Accuracy with 95% bootstrap CI for all three models |
| `data/attention_maps/<model>/Layer_<N>/Single_Lesions/` | Per-lesion attention heatmaps per layer |
| `data/attention_maps/<model>/Layer_<N>/GLOBAL_Attention_Map.png` | Global average attention heatmap per layer |
| `data/attention_maps/<model>/Layer_<N>/GLOBAL_Matrix.xlsx` | Underlying attention matrix (top-15 features) |
| `data/attention_maps/<model>/Rollout_Global/GLOBAL_Rollout_Map.png` | End-to-end attention rollout map |
| `data/attention_maps/<model>/Rollout_Global/GLOBAL_Rollout_Matrix.xlsx` | Underlying rollout matrix |
| `data/attention_maps/<model>/ALL_Features_Feature_Label_Mapping.xlsx` | Short-label to full feature name mapping |

**Columns in `prism_predictions.csv`:**

| Column | Description |
|---|---|
| `Lesion_ID` | Unique lesion identifier |
| `Prob_Transformer_20` | Level-1 probability from the 20% stable feature ensemble |
| `Prob_Transformer_40` | Level-1 probability from the 40% stable feature ensemble |
| `PRISM_Probability` | Final diagnostic probability from the Level-2 ensemble |
| `PRISM_Prediction` | Predicted class label (0 = metastasis, 1 = UBU) |
| `True_Label` | Remapped ground-truth label |

**Diagnostic threshold (0.50):**

- `PRISM_Probability ≥ 0.50` → Unspecific Bone Uptake (UBU) — benign, positive class
- `PRISM_Probability < 0.50` → PCa Bone Metastasis — malignant, negative class

---

## Model Architecture

PRISM uses a two-level stacking ensemble.

**Level 1 — TabularTransformer base learners.** Each radiomic feature is treated as a token: a shared linear projection embeds each scalar value into a `d_model`-dimensional vector. A learnable `[CLS]` token is prepended and the sequence receives a learned positional embedding. The sequence passes through a Transformer encoder (GELU activation, `batch_first=True`). The `[CLS]` token output at the final layer is passed through a LayerNorm + linear head to produce a single logit for binary classification. Two independent 5-fold cross-validated ensembles are trained: one on the top 20% most stable features (ranked by Frequency of selection a Ross folds), one on the top 40%.

**Level 2 — Soft voting.** The 5-fold averaged probabilities from the 20% and 40% ensembles are averaged to produce the final `PRISM_Probability`.

**Interpretability.** The `extract_attention_scientifically` function in `model.py` manually unrolls the encoder layer by layer, extracting raw multi-head attention weights (before residual connections and LayerNorm) at each layer. Attention rollout is computed across layers to trace end-to-end information flow from input features to the `[CLS]` token.

---

## Citation

If you use this repository or the PRISM framework in your research, please cite the corresponding paper (citation details to be added upon publication).

---

## License

MIT License

Copyright (c) 2026 Giovanni Pasini — Institute of Bioimaging and Complex Biological Systems (IBSBC-CNR)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
