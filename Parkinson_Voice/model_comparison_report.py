""
model_comparison_report.py
--------------------------
Generates visual charts and a detailed justification report for why
Random Forest was selected as the best model for Parkinson's Disease
detection, compared against SVM and MLP Neural Network.

Outputs:
  report/01_cv_metrics_comparison.png  — Cross-validation metrics bar chart
  report/02_holdout_metrics_comparison.png — Holdout test metrics bar chart
  report/03_radar_comparison.png — Radar/spider chart overlay
  report/04_clinical_impact.png — False negatives & clinical safety
  report/05_stability_comparison.png — Generalization stability (std dev)
  report/06_model_selection_summary.png — Final selection summary card
  report/model_justification.txt — Written justification rationale
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
REPORT_DIR = Path(__file__).parent / "report"
METADATA_PATH = Path(__file__).parent / "saved_models" / "model_metadata.json"

# Color palette
COLORS = {
    "svm": "#6366f1",            # indigo
    "random_forest": "#10b981",  # emerald green (winner)
    "mlp_neural_network": "#f59e0b",  # amber
}
MODEL_LABELS = {
    "svm": "SVM (RBF)",
    "random_forest": "Random Forest",
    "mlp_neural_network": "MLP Neural Net",
}

# Cross-validation results (from cv_comparison_results.csv)
CV_RESULTS = {
    "random_forest": {
        "accuracy": (93.30, 0.89),
        "precision": (94.57, 1.66),
        "recall": (91.79, 3.31),
        "f1": (93.09, 1.04),
        "roc_auc": (98.52, 0.64),
    },
    "svm": {
        "accuracy": (87.13, 3.65),
        "precision": (91.24, 4.01),
        "recall": (81.79, 4.13),
        "f1": (86.24, 3.92),
        "roc_auc": (93.65, 2.67),
    },
    "mlp_neural_network": {
        "accuracy": (70.53, 8.05),
        "precision": (72.80, 8.73),
        "recall": (63.93, 9.54),
        "f1": (68.05, 9.22),
        "roc_auc": (78.15, 9.49),
    },
}


def load_metadata() -> dict:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_style(fig, ax, title: str):
    """Apply consistent dark styling to a chart."""
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.set_title(title, fontsize=15, fontweight="bold", color="white", pad=14)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")


def _save(fig, name: str):
    path = REPORT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


# ── Chart 1: Cross-Validation Metrics Comparison ─────────────────────────────
def chart_cv_metrics():
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall\n(Sensitivity)", "F1-Score", "ROC-AUC"]
    model_order = ["random_forest", "svm", "mlp_neural_network"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    _apply_style(fig, ax, "5-Fold Stratified Cross-Validation — All Metrics Comparison")

    for i, model in enumerate(model_order):
        means = [CV_RESULTS[model][m][0] for m in metrics]
        stds = [CV_RESULTS[model][m][1] for m in metrics]
        bars = ax.bar(
            x + i * width, means, width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            yerr=stds,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "color": "white", "alpha": 0.7}
        )
        # Value annotations
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=COLORS[model]
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(50, 108)
    ax.legend(loc="lower right", fontsize=10, facecolor="#1e293b", edgecolor="#475569",
              labelcolor="white")
    ax.axhline(y=90, color="#475569", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(metrics) - 0.3, 90.5, "90% threshold", color="#94a3b8", fontsize=8, ha="right")

    _save(fig, "01_cv_metrics_comparison.png")


# ── Chart 2: Holdout Test Metrics ─────────────────────────────────────────────
def chart_holdout_metrics():
    meta = load_metadata()
    metrics = ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall\n(Sensitivity)", "Specificity", "F1-Score", "ROC-AUC"]
    model_order = ["random_forest", "svm", "mlp_neural_network"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    _apply_style(fig, ax, "80/20 Holdout Test — All Metrics Comparison")

    for i, model in enumerate(model_order):
        m_data = meta["models"][model]
        vals = [m_data[m] * 100 for m in metrics]
        bars = ax.bar(
            x + i * width, vals, width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color=COLORS[model]
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(50, 108)
    ax.legend(loc="lower right", fontsize=10, facecolor="#1e293b", edgecolor="#475569",
              labelcolor="white")

    _save(fig, "02_holdout_metrics_comparison.png")


# ── Chart 3: Radar / Spider Chart ────────────────────────────────────────────
def chart_radar():
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for model in ["random_forest", "svm", "mlp_neural_network"]:
        values = [CV_RESULTS[model][m][0] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill(angles, values, alpha=0.15, color=COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color="white")
    ax.set_ylim(55, 102)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels(["60%", "70%", "80%", "90%", "100%"], fontsize=8, color="#94a3b8")
    ax.yaxis.grid(True, color="#334155", linewidth=0.5)
    ax.xaxis.grid(True, color="#334155", linewidth=0.5)
    ax.spines["polar"].set_color("#475569")

    ax.set_title("Cross-Validation Performance Profile\n(Radar Comparison)",
                 fontsize=14, fontweight="bold", color="white", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=10,
              facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    _save(fig, "03_radar_comparison.png")


# ── Chart 4: Clinical Impact — False Negatives ──────────────────────────────
def chart_clinical_impact():
    meta = load_metadata()
    model_order = ["random_forest", "svm", "mlp_neural_network"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0f172a")

    # Left: False Negatives (missed PD cases)
    ax1.set_facecolor("#1e293b")
    fn_vals = [meta["models"][m]["false_negatives"] for m in model_order]
    bars1 = ax1.barh(
        [MODEL_LABELS[m] for m in model_order], fn_vals,
        color=[COLORS[m] for m in model_order],
        edgecolor="white", linewidth=0.5, height=0.5
    )
    for bar, val in zip(bars1, fn_vals):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val} missed", va="center", fontsize=11, fontweight="bold", color="#ef4444")
    ax1.set_title("False Negatives\n(Missed Parkinson's Diagnoses)", fontsize=12,
                  fontweight="bold", color="white", pad=10)
    ax1.set_xlabel("Count", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.set_xlim(0, max(fn_vals) + 4)
    ax1.invert_yaxis()

    # Right: False Positives (false alarms)
    ax2.set_facecolor("#1e293b")
    fp_vals = [meta["models"][m]["false_positives"] for m in model_order]
    bars2 = ax2.barh(
        [MODEL_LABELS[m] for m in model_order], fp_vals,
        color=[COLORS[m] for m in model_order],
        edgecolor="white", linewidth=0.5, height=0.5
    )
    for bar, val in zip(bars2, fp_vals):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val} false alarms", va="center", fontsize=11, fontweight="bold", color="#f59e0b")
    ax2.set_title("False Positives\n(Unnecessary Referrals)", fontsize=12,
                  fontweight="bold", color="white", pad=10)
    ax2.set_xlabel("Count", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.set_xlim(0, max(fp_vals) + 5)
    ax2.invert_yaxis()

    fig.suptitle("Clinical Safety Analysis — Holdout Test (N=114)",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    _save(fig, "04_clinical_impact.png")


# ── Chart 5: Generalization Stability ────────────────────────────────────────
def chart_stability():
    model_order = ["random_forest", "svm", "mlp_neural_network"]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_style(fig, ax, "Generalization Stability — Cross-Validation Standard Deviation\n(Lower is Better)")

    x = np.arange(len(metrics))
    width = 0.25

    for i, model in enumerate(model_order):
        stds = [CV_RESULTS[model][m][1] for m in metrics]
        bars = ax.bar(
            x + i * width, stds, width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, s in zip(bars, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"±{s:.2f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=COLORS[model]
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Standard Deviation (%)")
    ax.set_ylim(0, 12)
    ax.legend(loc="upper left", fontsize=10, facecolor="#1e293b", edgecolor="#475569",
              labelcolor="white")

    # Add annotation
    ax.annotate(
        "Random Forest has the lowest\nvariance across all metrics →\nmost stable generalization",
        xy=(1, CV_RESULTS["random_forest"]["f1"][1]),
        xytext=(3.2, 6),
        fontsize=10, color="#10b981", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.5", fc="#1e293b", ec="#10b981", lw=1.5)
    )

    _save(fig, "05_stability_comparison.png")


# ── Chart 6: Summary Selection Card ──────────────────────────────────────────
def chart_summary_card():
    meta = load_metadata()
    best = meta["best_model"]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.3, "WHY RANDOM FOREST?", fontsize=22, fontweight="bold",
            color="#10b981", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#064e3b", ec="#10b981", lw=2))
    ax.text(5, 8.6, "Model Selection Justification for Parkinson's Disease Detection",
            fontsize=12, color="#94a3b8", ha="center", va="center")

    # Winner box
    cv = best["cv_metrics"]
    winner_text = (
        f"5-Fold CV F1-Score:  {cv['f1_percent']:.2f}%\n"
        f"5-Fold CV Recall:    {cv['recall_percent']:.2f}%\n"
        f"5-Fold CV Accuracy:  {cv['accuracy_percent']:.2f}%\n"
        f"5-Fold CV ROC-AUC:   {cv['roc_auc_percent']:.2f}%\n"
        f"CV Accuracy Std Dev: ±0.89%"
    )
    ax.text(1.5, 6.8, "RANDOM FOREST — WINNING METRICS", fontsize=13, fontweight="bold",
            color="#10b981", va="top")
    ax.text(1.5, 6.3, winner_text, fontsize=11, color="white", va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#064e3b", ec="#10b981", lw=1.5, alpha=0.5))

    # Key reasons
    reasons = [
        ("① Highest CV F1-Score (93.09%)",
         "Balances precision and recall — the best single metric for imbalanced medical screening."),
        ("② Highest Recall / Sensitivity (91.79%)",
         "Only 3 missed PD cases on holdout vs 4 (SVM) and 10 (MLP). Critical for clinical safety."),
        ("③ Most Stable Generalization (±0.89% std)",
         "SVM had ±3.65% and MLP had ±8.05%. RF's low variance means reliable real-world performance."),
        ("④ Highest ROC-AUC (98.52%)",
         "Best class-separability across all probability thresholds."),
        ("⑤ No Feature Scaling Required",
         "Tree-based models are scale-invariant — fewer preprocessing assumptions on new voice data."),
        ("⑥ Built-in Feature Importance",
         "Provides interpretable rankings of which vocal biomarkers contribute most to diagnosis."),
    ]

    y_pos = 4.1
    for title, desc in reasons:
        ax.text(1, y_pos, title, fontsize=10.5, fontweight="bold", color="#facc15", va="top")
        ax.text(1.2, y_pos - 0.35, desc, fontsize=9.5, color="#cbd5e1", va="top")
        y_pos -= 0.85

    # Bottom attribution
    ax.text(5, -0.2, "Generated from 5-Fold Stratified CV + 80/20 Holdout Test | 23 acoustic features | 569 recordings",
            fontsize=8, color="#475569", ha="center", va="center")

    _save(fig, "06_model_selection_summary.png")


# ── Written Justification ─────────────────────────────────────────────────────
def write_justification():
    meta = load_metadata()
    cv = meta["best_model"]["cv_metrics"]

    text = textwrap.dedent(f"""\
    ================================================================================
    MODEL SELECTION JUSTIFICATION — RANDOM FOREST FOR PARKINSON'S DISEASE DETECTION
    ================================================================================

    EXPERIMENT SETUP
    ────────────────
    • 3 candidate models:  SVM (RBF kernel), Random Forest (200 trees), MLP Neural Network (64×32)
    • 23 acoustic features: voice quality (jitter, shimmer, HNR), spectral (MFCCs, centroid),
      and prosodic (pitch, speech rate) features extracted from voice recordings
    • Validation:          5-Fold Stratified Cross-Validation + 80/20 holdout test
    • Selection criteria:  Primary — CV F1-Score; Secondary — Recall (clinical safety)

    CROSS-VALIDATION RESULTS (5-Fold Stratified)
    ─────────────────────────────────────────────
    Model               Accuracy        Precision       Recall          F1-Score        ROC-AUC
    Random Forest       93.30 ±0.89%    94.57 ±1.66%    91.79 ±3.31%    93.09 ±1.04%    98.52 ±0.64%
    SVM (RBF)           87.13 ±3.65%    91.24 ±4.01%    81.79 ±4.13%    86.24 ±3.92%    93.65 ±2.67%
    MLP Neural Net      70.53 ±8.05%    72.80 ±8.73%    63.93 ±9.54%    68.05 ±9.22%    78.15 ±9.49%

    HOLDOUT TEST RESULTS (80/20 Split, N_test=114)
    ───────────────────────────────────────────────
    Model               Accuracy    Recall      F1-Score    ROC-AUC     FN (Missed PD)  FP (False Alarm)
    Random Forest       92.98%      94.64%      92.98%      98.61%      3               5
    SVM (RBF)           93.86%      92.86%      93.69%      98.58%      4               3
    MLP Neural Net      74.56%      82.14%      76.03%      80.11%      10              19

    WHY RANDOM FOREST WAS SELECTED
    ───────────────────────────────

    1. HIGHEST CROSS-VALIDATION F1-SCORE ({cv['f1_percent']:.2f}%)
       The F1-score harmonically balances precision and recall, making it the most
       appropriate primary metric for medical screening where both false positives
       and false negatives carry clinical cost. Random Forest's 93.09% CV F1 is
       6.85 percentage points higher than SVM (86.24%) and 25.04 points higher
       than MLP (68.05%).

    2. BEST CLINICAL SENSITIVITY / RECALL ({cv['recall_percent']:.2f}% CV, 94.64% holdout)
       In Parkinson's screening, a false negative (missing a PD patient) is far more
       dangerous than a false positive (unnecessary follow-up). Random Forest achieved:
       • CV Recall:  91.79% (vs SVM 81.79%, MLP 63.93%)
       • Holdout FN: Only 3 missed PD cases (vs SVM 4, MLP 10)
       This makes RF the safest model from a clinical decision standpoint.

    3. MOST STABLE GENERALIZATION (±0.89% accuracy std)
       Random Forest exhibited the lowest cross-validation variance across ALL metrics:
       • Accuracy std:  ±0.89% (vs SVM ±3.65%, MLP ±8.05%)
       • F1 std:        ±1.04% (vs SVM ±3.92%, MLP ±9.22%)
       • ROC-AUC std:   ±0.64% (vs SVM ±2.67%, MLP ±9.49%)
       Low variance indicates the model generalizes reliably to unseen voice recordings
       and is not sensitive to the specific train/test partition — essential for
       deployment in diverse clinical populations.

    4. HIGHEST ROC-AUC ({cv['roc_auc_percent']:.2f}%)
       The area under the ROC curve measures the model's ability to separate
       PD-positive and healthy classes across all probability thresholds. RF's
       98.52% indicates near-perfect discriminative power.

    5. ROBUSTNESS TO FEATURE SCALE AND MISSING DATA
       Unlike SVM and MLP which require feature standardization (StandardScaler),
       Random Forest is a tree-based ensemble that is inherently scale-invariant.
       This means:
       • No preprocessing assumptions are imposed on new, unseen voice recordings
       • The model gracefully handles the mixed-scale nature of acoustic features
         (e.g., Hz-scale pitch vs. dB-scale HNR vs. unitless MFCCs)
       • It naturally handles missing values through median imputation without
         distortion from scaling artifacts

    6. INTERPRETABILITY VIA FEATURE IMPORTANCE
       Random Forest provides built-in feature importance scores (Gini importance),
       enabling clinicians and researchers to understand which vocal biomarkers
       (jitter, shimmer, MFCC coefficients, etc.) contribute most to a PD
       prediction — critical for clinical trust and regulatory transparency.
       SVM (RBF kernel) and MLP are comparatively opaque.

    WHY NOT SVM?
    ────────────
    While SVM achieved the highest *holdout* accuracy (93.86%), its CV performance
    tells a different story: 87.13% accuracy with ±3.65% std — significantly less
    stable. The holdout result likely benefits from a favorable random split. SVM's
    CV recall of 81.79% is ~10 points lower than RF, meaning it misses more PD
    patients across different data partitions. SVM also requires feature scaling
    and offers limited interpretability with an RBF kernel.

    WHY NOT MLP?
    ────────────
    The MLP Neural Network performed substantially worse across every metric,
    with high variance (±8.05% accuracy std). With only ~569 samples, a neural
    network lacks the data volume to learn robust representations. It missed
    10 PD patients on holdout (vs RF's 3) and produced 19 false alarms (vs RF's 5).
    This is clinically unacceptable and indicates severe overfitting/underfitting.

    CONCLUSION
    ──────────
    Random Forest provides the optimal balance of:
     [+] Highest discriminative performance (F1, ROC-AUC)
     [+] Best clinical safety (highest recall, fewest missed diagnoses)
     [+] Most reliable generalization (lowest variance across folds)
     [+] Practical deployment advantages (no scaling, interpretable)

    It is the clear choice for this Parkinson's voice-based screening system.
    ================================================================================
    """)

    path = REPORT_DIR / "model_justification.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  [OK] Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating Model Comparison Report...")
    print("=" * 50)

    chart_cv_metrics()
    chart_holdout_metrics()
    chart_radar()
    chart_clinical_impact()
    chart_stability()
    chart_summary_card()
    write_justification()

    print("=" * 50)
    print(f"All charts and justification saved to: {REPORT_DIR}")

