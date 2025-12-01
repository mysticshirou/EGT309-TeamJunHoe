import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from skopt.space import Integer, Categorical
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
        elif isinstance(item[0], str):
            search_space[key] = Categorical(item)

    return search_space

def generate_report(y_test, y_prob, y_pred, beta):
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    # Remove last element to match thresholds length
    precision = precision[:-1]
    recall = recall[:-1]
    # F-beta computation
    beta_sq = beta * beta
    fbeta = (1 + beta_sq) * (precision * recall) / (
        (beta_sq * precision) + recall + 1e-10
    )
    # Best threshold according to F-beta
    best_idx = np.argmax(fbeta)
    best_threshold = thresholds[best_idx]
    # Predictions at best threshold
    y_pred_best = (y_prob >= best_threshold).astype(int)

    # Classification Report
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    # Custom report (no F1, no support)
    report = {
        "metrics": {
            "precision": precision[best_idx],
            "recall": recall[best_idx],
            "fbeta": fbeta[best_idx]
        },
        "full_cls_report": cls_report
    }

    # Plot precision-recall curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (β={beta})")

    return report, fig