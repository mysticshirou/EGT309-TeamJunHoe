import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, fbeta_score
import seaborn as sns
from scipy.stats import hmean

from skopt.space import Integer, Categorical, Real
from typing import Any

def read_bs_search_space(search_dict: dict[str, list]) -> dict[str, Any]:
    # Read and sort categories into respective skopt spaces
    search_space = dict()
    for key in search_dict:
        # item list should all have the same dtypes for items
        item = search_dict[key]
        assert len(set(type(x) for x in item)) == 1
        if isinstance(item[0], int):
            # Length of item list should be 2 if integer and index 0 < index 1
            assert len(item) == 2 and item[0] < item[1]
            search_space[key] = Integer(item[0], item[1])
        elif isinstance(item[0], float):
            assert len(item) == 2 and item[0] < item[1]
            search_space[key] = Real(item[0], item[1])
        elif isinstance(item[0], str):
            search_space[key] = Categorical(item)

    return search_space

def generate_report(y_test, y_prob, params):
    alpha = params.get("alpha", 0.5)

    # Thresholds from predicted probabilities
    precision, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)

    custom_scores = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        # True positives & True negatives
        tp = np.sum((y_pred_t == 1) & (y_test == 1))
        tn = np.sum((y_pred_t == 0) & (y_test == 0))
        fn = np.sum((y_pred_t == 0) & (y_test == 1))
        fp = np.sum((y_pred_t == 1) & (y_test == 0))

        # Sensitivity (recall) and Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Weighted G-Mean
        score = (sensitivity ** alpha) * (specificity ** (1 - alpha))
        custom_scores.append(score)

    custom_scores = np.array(custom_scores)
    best_idx = np.argmax(custom_scores)
    best_threshold = thresholds[best_idx]
    y_pred_best = (y_prob >= best_threshold).astype(int)

    # Metrics at best threshold
    tp = np.sum((y_pred_best == 1) & (y_test == 1))
    tn = np.sum((y_pred_best == 0) & (y_test == 0))
    fp = np.sum((y_pred_best == 1) & (y_test == 0))
    fn = np.sum((y_pred_best == 0) & (y_test == 1))

    recall_best = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_best = tn / (tn + fp) if (tn + fp) > 0 else 0
    weighted_gmean_best = custom_scores[best_idx]

    # Classification reports
    cls_report = classification_report(y_test, (y_prob >= 0.5).astype(int), output_dict=True)
    best_cls_report = classification_report(y_test, y_pred_best, output_dict=True)

    report = {
        "metrics": {
            "sensitivity (recall)": recall_best,
            "specificity": specificity_best,
            "weighted_gmean": weighted_gmean_best,
            "threshold": best_threshold,
            "alpha": alpha
        },
        "full_cls_report": cls_report,
        "best_cls_report": best_cls_report
    }

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # PR curve
    ax[0].plot(recall_vals, precision)
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall Curve")
    
    # Confusion matrix at best threshold
    cf_matrix = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cf_matrix, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_xticklabels(["Predicted No", "Predicted Yes"])
    ax[1].set_yticklabels(["Actual No", "Actual Yes"])
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    return report, fig
