import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

from skopt.space import Integer, Categorical, Real
from typing import Any

def read_bs_search_space(search_dict: dict[str, list]) -> dict[str, Any]:
    """
    Converting search space defined in parameters.yml into a proper BayesSearchCV search space
    
    Parameters
        search_dict: Dictionary containing (key) keyword argument name: (value) list indicating search space

    Returns
        search_space: Dictionary containing (key) keyword argument name: (value) Integer (int) / Real (float) / Categorical (string) skopt search space
    
    Usage
        For integers and floats:    Length of list in parameters.yml == 2, lower bound at index 0, upper bound at index 1
        For categorical:            No length limit, categories must match possible values for specified hyperparameter
    """
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
    """
    Generates the evaluation report

    Parameters
        y_test: Target test split
        y_prob: Output from model.predict_proba() method
        params: Parameter dictionary obtained from config

    Returns
        report: Dictionary containing evaluation metrics
        fig:    matplotlib.figure.Figure object. Contains evaluation visualisations
    """
    alpha = params.get("alpha", 0.5)

    # ROC curve values
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Compute weighted GMS across all thresholds
    custom_scores = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        # calculate metrics using their predicted and real classes
        tp = np.sum((y_pred_t == 1) & (y_test == 1))
        tn = np.sum((y_pred_t == 0) & (y_test == 0))
        fn = np.sum((y_pred_t == 0) & (y_test == 1))
        fp = np.sum((y_pred_t == 1) & (y_test == 0))

        # TP / TP + FN
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # TN / TN + FP
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Geometric Mean Score
        score = (sensitivity ** alpha) * (specificity ** (1 - alpha))
        custom_scores.append(score)

    # Gets the best threshold
    custom_scores = np.array(custom_scores)
    best_idx = np.argmax(custom_scores)
    best_threshold = thresholds[best_idx]

    # Get the predictions using the best threshold and recalculate all metrics
    y_pred_best = (y_prob >= best_threshold).astype(int)

    tp = np.sum((y_pred_best == 1) & (y_test == 1))
    tn = np.sum((y_pred_best == 0) & (y_test == 0))
    fp = np.sum((y_pred_best == 1) & (y_test == 0))
    fn = np.sum((y_pred_best == 0) & (y_test == 1))

    recall_best = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_best = tn / (tn + fp) if (tn + fp) > 0 else 0
    weighted_gmean_best = custom_scores[best_idx]

    # Final report dictionary
    report = {
        "metrics": {
            "sensitivity (recall)": float(recall_best),
            "specificity": float(specificity_best),
            "weighted_gmean": float(weighted_gmean_best),
            "roc_auc": float(roc_auc)
        },
        "parameters": {
            "threshold": float(best_threshold),
            "alpha": float(alpha)
        }
    }

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # ROC curve
    ax[0].plot(fpr, tpr)
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve (AUC = {:.4f})".format(roc_auc))

    # Confusion matrix at best threshold
    cf_matrix = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cf_matrix, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_xticklabels(["Predicted No", "Predicted Yes"])
    ax[1].set_yticklabels(["Actual No", "Actual Yes"])
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    return report, fig
