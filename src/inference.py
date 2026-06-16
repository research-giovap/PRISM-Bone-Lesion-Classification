import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from model import TabularTransformer, extract_attention_scientifically

# Suppress the annoying scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator StandardScaler")

# ================================
# AESTHETICS & STYLE CONSTANTS
# ================================
PLT_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.0,
    'font.family': 'DejaVu Sans',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}

GROUP_COLORS = {
    'PET': '#C0392B',
    'CT':  '#2471A3',
}

CMAP_ALL = 'magma'
TOP_K_LESION = 15   
TOP_K_GLOBAL = 15   

# ================================
# STATISTICAL EVALUATION METRICS
# ================================
def bootstrap_ci_metrics(y_true, y_prob, model_name, n_bootstraps=1000, seed=42):
    """Computes exact point metrics on the full set, and 95% CIs via bootstrapping."""
    y_pred = (y_prob >= 0.50).astype(int)
    
    # 1. Exact Point Estimates (calculated on the full, true test set)
    point_estimates = {
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred)
    }
    
    # 2. Bootstrapping strictly for the 95% Confidence Intervals
    np.random.seed(seed)
    metrics_boot = {'AUC': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'Accuracy': []}
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]
        
        # Skip samples that only contain one class (prevents AUC errors)
        if len(np.unique(y_true_b)) < 2:
            continue
            
        y_pred_b = (y_prob_b >= 0.50).astype(int)
        
        metrics_boot['AUC'].append(roc_auc_score(y_true_b, y_prob_b))
        metrics_boot['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        metrics_boot['Recall'].append(recall_score(y_true_b, y_pred_b))
        metrics_boot['F1-score'].append(f1_score(y_true_b, y_pred_b))
        metrics_boot['Accuracy'].append(accuracy_score(y_true_b, y_pred_b))
        
    # 3. Format Output exactly as: Point Estimate (Lower-Upper)
    results = {'Model': model_name}
    for m in ['AUC', 'Precision', 'Recall', 'F1-score', 'Accuracy']:
        lower = np.percentile(metrics_boot[m], 2.5)
        upper = np.percentile(metrics_boot[m], 97.5)
        results[m] = f"{point_estimates[m]:.3f} ({lower:.3f}-{upper:.3f})"
        
    return results

# ================================
# FEATURE LABELLING UTILITIES
# ================================
def build_short_labels(feature_names):
    short_labels = []
    for name in feature_names:
        parts = name.split('_')
        mod = 'PET' if parts[0].upper() == 'PET' else 'CT'

        if 'original' in parts:
            filt = 'orig'
        elif 'log' in parts:
            try:
                sig_idx = parts.index('sigma')
                sigma = parts[sig_idx + 1] + '.' + parts[sig_idx + 2]
                filt = f'log{sigma}'
            except (ValueError, IndexError):
                filt = 'log'
        elif 'wavelet' in parts:
            try:
                wav_idx = parts.index('wavelet')
                subband = parts[wav_idx + 1]
                filt = f'wav{subband}'
            except (ValueError, IndexError):
                filt = 'wav'
        else:
            filt = parts[1][:6] if len(parts) > 1 else '?'

        feat_classes = ['firstorder', 'glcm', 'glszm', 'gldm', 'glrlm', 'ngtdm', 'shape2D', 'shape']
        fclass = '?'
        for fc in feat_classes:
            if fc in parts:
                fclass = fc[:8]
                break

        stat = parts[-1][:12]
        short_labels.append(f"{mod}|{filt}|{fclass}|{stat}")

    label_table = pd.DataFrame({
        'Short Label': short_labels,
        'Full Feature Name': feature_names
    })
    return short_labels, label_table

def get_feature_colors(feature_names):
    return [GROUP_COLORS['PET' if name.upper().startswith('PET') else 'CT'] for name in feature_names]

def apply_tick_colors(ax, colors_x, colors_y):
    for tick, col in zip(ax.get_xticklabels(), colors_x):
        tick.set_color(col)
    for tick, col in zip(ax.get_yticklabels(), colors_y):
        tick.set_color(col)

def save_label_table(label_table, save_dir, prefix=''):
    path = os.path.join(save_dir, f"{prefix}Feature_Label_Mapping.xlsx")
    label_table.to_excel(path, index=False)

# ================================
# CORE PLOTTING ENGINE
# ================================
def _compute_display_matrix(matrix, gamma=0.5, clip_percentile=98):
    nonzero = matrix[matrix > 0]
    vmax = np.percentile(nonzero, clip_percentile) if len(nonzero) > 0 else 1.0
    vmin = max(0.0, np.percentile(nonzero, 2))     if len(nonzero) > 0 else 0.0
    display = np.clip(matrix, vmin, vmax)
    display = np.power((display - vmin) / (vmax - vmin + 1e-12), gamma)
    return display, vmin, vmax

def _add_modality_legend(ax, loc='lower right'):
    patches = [
        mpatches.Patch(color=GROUP_COLORS['PET'], label='PET feature'),
        mpatches.Patch(color=GROUP_COLORS['CT'],  label='CT feature'),
    ]
    leg = ax.legend(handles=patches, loc=loc, fontsize=11, framealpha=0.92,
                    edgecolor='#aaaaaa', handlelength=1.4, handleheight=1.0, borderpad=0.6)
    ax.add_artist(leg)

def plot_heatmap_publication(matrix, short_labels, full_labels, title, filename, save_dir,
                             cmap=CMAP_ALL, gamma=0.5, clip_percentile=98, n_lesions=None, n_folds=None):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update(PLT_STYLE)

    n = len(short_labels)
    index_labels = [f"F{i+1:02d}" for i in range(n)]
    feat_colors = get_feature_colors(full_labels)

    display_mat, vmin_orig, vmax_orig = _compute_display_matrix(matrix, gamma=gamma, clip_percentile=clip_percentile)

    cell_size = 0.55
    fig_side = max(8.0, n * cell_size + 3.5)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    fig.patch.set_facecolor('white')

    sns.heatmap(display_mat, ax=ax, xticklabels=index_labels, yticklabels=index_labels, cmap=cmap,
                square=True, linewidths=0.3, linecolor='#1a1a1a', cbar=True,
                cbar_kws={"shrink": 0.72, "pad": 0.03, "aspect": 28}, vmin=0.0, vmax=1.0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Attention weight  (γ-compressed)", size=12, labelpad=10)
    
    cbar.ax.text(2.4, 1.02, f"(~{vmax_orig:.3f})", transform=cbar.ax.transAxes, fontsize=10, color='#222222', ha='center', va='bottom', fontweight='bold')
    cbar.ax.text(2.4, -0.02, f"(~{vmin_orig:.3f})", transform=cbar.ax.transAxes, fontsize=10, color='#222222', ha='center', va='top', fontweight='bold')

    ax.tick_params(axis='x', rotation=90, labelsize=11, length=3, pad=3)
    ax.tick_params(axis='y', rotation=0,  labelsize=11, length=3, pad=3)
    apply_tick_colors(ax, feat_colors, feat_colors)

    ax.set_xlabel("Provider feature  j  (key / value)", fontsize=12, labelpad=10)
    ax.set_ylabel("Collector feature  i  (query)", fontsize=12, labelpad=10)
    _add_modality_legend(ax, loc='lower right')

    subtitle_parts = []
    if n_lesions is not None: subtitle_parts.append(f"n = {n_lesions} lesions")
    if n_folds is not None: subtitle_parts.append(f"avg of {n_folds} folds")
    subtitle_parts.append(f"top {n} features by attention mass")
    
    ax.set_title(f"{title}\n{'  ·  '.join(subtitle_parts)}", fontsize=14, fontweight='bold', pad=14)

    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    mapping_df = pd.DataFrame({'Index': index_labels, 'Short Label': short_labels, 'Full Feature Name': full_labels})
    mapping_df.to_excel(os.path.join(save_dir, filename.replace('.png', '_LabelMapping.xlsx')), index=False)

def plot_heatmap_lesion(matrix, short_labels, full_labels, title, filename, save_dir, n_folds=None):
    plot_heatmap_publication(matrix, short_labels, full_labels, title=title, filename=filename, save_dir=save_dir,
                             cmap=CMAP_ALL, gamma=0.45, clip_percentile=97, n_lesions=1, n_folds=n_folds)

def plot_heatmap_global(matrix, short_labels, full_labels, title, filename, save_dir, n_lesions=None):
    plot_heatmap_publication(matrix, short_labels, full_labels, title=title, filename=filename, save_dir=save_dir,
                             cmap=CMAP_ALL, gamma=0.50, clip_percentile=98, n_lesions=n_lesions)

def plot_heatmap_rollout(matrix, short_labels, full_labels, title, filename, save_dir, n_lesions=None):
    plot_heatmap_publication(matrix, short_labels, full_labels, title=title, filename=filename, save_dir=save_dir,
                             cmap=CMAP_ALL, gamma=0.45, clip_percentile=97, n_lesions=n_lesions)

# ================================
# INFERENCE AND EXTRACTION
# ================================
def load_and_predict_fold(test_df, fold_path, device, fold_idx):
    """Loads metadata, scaler, model, runs inference, and extracts base attention."""
    with open(f"{fold_path}/metadata.json", 'r') as f:
        metadata = json.load(f)
        
    features = metadata['features']
    hparams = metadata['hyperparams']
    
    X_test_raw = test_df[features].values
    
    scaler = joblib.load(f"{fold_path}/scaler.joblib")
    X_scaled = scaler.transform(X_test_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    model = TabularTransformer(
        num_features=len(features), d_model=hparams['d_model'], nhead=hparams['nhead'],
        num_layers=hparams['num_layers'], dim_feedforward=hparams['dim_feedforward'], dropout=hparams['dropout']
    )
    
    model_file = f"{fold_path}/transformer.pt" if os.path.exists(f"{fold_path}/transformer.pt") else f"{fold_path}/model.pt"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    
    # 1. Forward pass for prediction ONLY
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    # 2. Extract attention using the scientific unrolling method
    attns = extract_attention_scientifically(model, X_tensor)
    
    # 3. Process attention for layer maps and rollout
    batch_size = X_tensor.size(0)
    seq_len = len(features) + 1
    layers_attn = []
    
    rollout = torch.eye(seq_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    for attn_tensor in attns:
        # Average across the attention heads: [Batch, Heads, Seq, Seq] -> [Batch, Seq, Seq]
        avg_heads = attn_tensor.mean(dim=1).to(device)
        layers_attn.append(avg_heads.cpu().numpy())
        
        # Accumulate Rollout via batch matrix multiplication
        rollout = torch.bmm(avg_heads, rollout)
        
    # Row-normalize rollout
    rollout = rollout / rollout.sum(dim=-1, keepdim=True)
    rollout_attn = rollout.cpu().numpy()
        
    return probs, layers_attn, rollout_attn, features

def get_ensemble_predictions(test_df, base_model_path, device, model_subset_name):
    """Runs inference across all 5 folds, aggregates attention maps dynamically, and plots them."""
    fold_probabilities = []
    n_lesions = len(test_df)
    
    # Dynamically tracking layers
    lesion_layer_sums = []
    lesion_layer_counts = []
    lesion_rollout_sums = None
    features = None
    valid_folds = 0
    
    for fold in range(1, 6):
        fold_path = f"{base_model_path}/fold_{fold}"
        if not os.path.exists(fold_path):
            continue
            
        print(f"  -> Processing {fold_path}...")
        probs, layers_attn, rollout_attn, fold_features = load_and_predict_fold(test_df, fold_path, device, fold)
        
        fold_probabilities.append(probs)
        if features is None: features = fold_features
            
        if lesion_rollout_sums is None:
            lesion_rollout_sums = np.zeros_like(rollout_attn)
            
        # Dynamically append if a new fold has MORE layers than previous folds
        for i, l_attn in enumerate(layers_attn):
            if i >= len(lesion_layer_sums):
                lesion_layer_sums.append(np.zeros_like(l_attn))
                lesion_layer_counts.append(0)
                
            lesion_layer_sums[i] += l_attn
            lesion_layer_counts[i] += 1
            
        lesion_rollout_sums += rollout_attn
        valid_folds += 1

    # Average Probs
    ensemble_probs = np.mean(fold_probabilities, axis=0)
    
    # Average Attentions based on exact valid fold counts
    num_layers = len(lesion_layer_sums)
    lesion_layer_avg = [lesion_layer_sums[i] / lesion_layer_counts[i] for i in range(num_layers)]
    lesion_rollout_avg = lesion_rollout_sums / valid_folds
    
    # ---------------------------------------------------------
    # PLOTTING AND EXPORTING
    # ---------------------------------------------------------
    out_dir = f'../data/attention_maps/{model_subset_name}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"  -> Generating plots and matrices into: {out_dir}")
    
    short_labels_all, label_table_all = build_short_labels(features)
    save_label_table(label_table_all, out_dir, prefix='ALL_Features_')
    
    lesion_ids = test_df['Lesion_ID'].values if 'Lesion_ID' in test_df.columns else test_df.index.values
    
    global_layer_sums = [np.zeros((len(features) + 1, len(features) + 1)) for _ in range(num_layers)]
    global_rollout_sum = np.zeros((len(features) + 1, len(features) + 1))
    
    # 1. Individual Lesion Plots
    for les_idx in range(n_lesions):
        les_id = lesion_ids[les_idx]
        safe_id = "".join([c for c in str(les_id) if c.isalnum() or c in ('_', '-')])
        
        for l_idx in range(num_layers):
            if lesion_layer_counts[l_idx] == 0: continue
                
            avg_matrix = lesion_layer_avg[l_idx][les_idx]
            global_layer_sums[l_idx] += avg_matrix
            
            feat_matrix = avg_matrix[1:, 1:]
            imp = feat_matrix.sum(axis=0) + feat_matrix.sum(axis=1)
            top_idx = np.argsort(imp)[::-1][:min(TOP_K_LESION, len(features))]
            
            top_mat = feat_matrix[np.ix_(top_idx, top_idx)]
            top_shorts = [short_labels_all[i] for i in top_idx]
            top_fulls = [features[i] for i in top_idx]
            
            layer_dir = os.path.join(out_dir, f"Layer_{l_idx}", "Single_Lesions")
            plot_heatmap_lesion(
                top_mat, top_shorts, top_fulls,
                title=f"Lesion: {les_id} | Layer {l_idx} | Avg of {lesion_layer_counts[l_idx]} Folds",
                filename=f"Map_L{l_idx}_{safe_id}.png", save_dir=layer_dir, n_folds=lesion_layer_counts[l_idx]
            )
            
        rollout_matrix = lesion_rollout_avg[les_idx]
        global_rollout_sum += rollout_matrix
        
    # 2. Global Layer Plots & Matrices
    for l_idx in range(num_layers):
        if lesion_layer_counts[l_idx] == 0: continue
            
        global_avg = global_layer_sums[l_idx] / n_lesions
        feat_matrix_glob = global_avg[1:, 1:]
        
        imp_glob = feat_matrix_glob.sum(axis=0) + feat_matrix_glob.sum(axis=1)
        top_idx_g = np.argsort(imp_glob)[::-1][:min(TOP_K_GLOBAL, len(features))]
        
        top_mat_g = feat_matrix_glob[np.ix_(top_idx_g, top_idx_g)]
        top_shorts_g = [short_labels_all[i] for i in top_idx_g]
        top_fulls_g = [features[i] for i in top_idx_g]
        
        g_dir = os.path.join(out_dir, f"Layer_{l_idx}")
        plot_heatmap_global(
            top_mat_g, top_shorts_g, top_fulls_g,
            title=f"Global Attention Map — Layer {l_idx}",
            filename="GLOBAL_Attention_Map.png", save_dir=g_dir, n_lesions=n_lesions
        )
        pd.DataFrame(top_mat_g, index=top_shorts_g, columns=top_shorts_g).to_excel(os.path.join(g_dir, "GLOBAL_Matrix.xlsx"))
        
    # 3. Global Rollout Plots & Matrices
    global_rollout_avg = global_rollout_sum / n_lesions
    feat_rollout_glob = global_rollout_avg[1:, 1:]
    
    imp_rollout = feat_rollout_glob.sum(axis=0) + feat_rollout_glob.sum(axis=1)
    top_idx_r = np.argsort(imp_rollout)[::-1][:min(TOP_K_GLOBAL, len(features))]
    
    top_rollout_mat = feat_rollout_glob[np.ix_(top_idx_r, top_idx_r)]
    top_rollout_shorts = [short_labels_all[i] for i in top_idx_r]
    top_rollout_fulls = [features[i] for i in top_idx_r]
    
    rollout_dir = os.path.join(out_dir, "Rollout_Global")
    plot_heatmap_rollout(
        top_rollout_mat, top_rollout_shorts, top_rollout_fulls,
        title="Global Attention Rollout — End-to-End Information Flow",
        filename="GLOBAL_Rollout_Map.png", save_dir=rollout_dir, n_lesions=n_lesions
    )
    pd.DataFrame(top_rollout_mat, index=top_rollout_shorts, columns=top_rollout_shorts).to_excel(os.path.join(rollout_dir, "GLOBAL_Rollout_Matrix.xlsx"))

    return ensemble_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

  # 1. Load the held-out test data
    test_df = pd.read_csv('../data/test_data.csv', sep=';')
    test_df.columns = test_df.columns.str.replace('-', '_')

    # True labels for metric evaluation
    y_true = test_df['label'].values
    
   
    if set(np.unique(y_true)).issubset({1, 2}):
        y_true = np.where(y_true == 2, 1, 0)

    # 2. Get L1 Ensemble Predictions & Attention for the 20% Subset
    print("\n--- Running Transformer 20% L1 Ensemble ---")
    probs_20pct = get_ensemble_predictions(test_df, '../models/transformer_20pct', device, 'transformer_20pct')
    metrics_20 = bootstrap_ci_metrics(y_true, probs_20pct, "Transformer 20% (L1)")
    
    # 3. Get L1 Ensemble Predictions & Attention for the 40% Subset
    print("\n--- Running Transformer 40% L1 Ensemble ---")
    probs_40pct = get_ensemble_predictions(test_df, '../models/transformer_40pct', device, 'transformer_40pct')
    metrics_40 = bootstrap_ci_metrics(y_true, probs_40pct, "Transformer 40% (L1)")

    # 4. PRISM Ensemble (Level 2: Soft Voting between 20% and 40%)
    prism_probabilities = (probs_20pct + probs_40pct) / 2.0
    metrics_prism = bootstrap_ci_metrics(y_true, prism_probabilities, "PRISM (L2 Ensemble)")
    
    # 5. Output Raw Predictions CSV
    results_df = pd.DataFrame({
        'Lesion_ID': test_df['Lesion_ID'] if 'Lesion_ID' in test_df.columns else test_df.index,  
        'Prob_Transformer_20': probs_20pct,
        'Prob_Transformer_40': probs_40pct,
        'PRISM_Probability': prism_probabilities,
        'PRISM_Prediction': (prism_probabilities >= 0.50).astype(int),
        'True_Label': y_true 
    })
    results_df.to_csv('../data/prism_predictions.csv', index=False)
    
    # 6. Output Statistical Metrics CSV
    metrics_df = pd.DataFrame([metrics_20, metrics_40, metrics_prism])
    metrics_df.to_csv('../data/prism_performance_metrics.csv', index=False)

    print("\n============================================================")
    print("FINAL TEST SET PERFORMANCE METRICS (1000-Bootstrap 95% CI)")
    print("============================================================")
    print(metrics_df.to_string(index=False))
    print("\nInference complete.")
    print("  -> Predictions saved to: data/prism_predictions.csv")
    print("  -> Metrics saved to:     data/prism_performance_metrics.csv")

if __name__ == '__main__':
    main()