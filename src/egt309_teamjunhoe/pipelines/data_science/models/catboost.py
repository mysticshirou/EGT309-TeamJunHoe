from catboost import CatBoostClassifier, Pool
from .model_utils import read_bs_search_space
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import numpy as np
from skopt import BayesSearchCV
from egt309_teamjunhoe.pipelines.data_preprocessing.nodes import split_dataset
import yaml
import os

from .interfaces import Model
import matplotlib.pyplot as plt
import seaborn as sns

class CatBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        combined_df = X_train.copy()
        combined_df[y_train.name] = y_train

        # Split the training dataset into train and validation so we can evaluate best model 
        with open(os.path.join(os.getcwd(), "conf", "base", "parameters.yml"), "r") as yaml_file:
            split_parameters = yaml.safe_load(yaml_file)["splitting_params"]

        X_re_train, X_eval, y_re_train, y_eval = split_dataset(combined_df, split_parameters)   
        categorical_features = X_re_train.select_dtypes(include=['object']).columns.tolist()
        train_pool = Pool(X_re_train, y_re_train, cat_features=categorical_features)
        eval_pool = Pool(X_eval, y_eval, cat_features=categorical_features)
        
        if params.get("cat_boost_auto_optimize") == True:
            search_space = params.get("cat_boost_grid_search_search_space", {})
            assert len(search_space) > 0

            clf_params = params.get("cat_boost_settings", {})
            clf = CatBoostClassifier(random_state=params.get("random_state"),
                                     **clf_params)
            results = clf.grid_search(search_space, train_pool)

            clf = CatBoostClassifier(random_state=params.get("random_state"),
                                     **{**clf_params, **results["params"]})
        else:
            clf = CatBoostClassifier(random_state=params.get("random_state"),
                                    **params.get("cat_boost_settings", dict()))
        
        clf.fit(train_pool, eval_set=eval_pool, use_best_model=True)

        return clf, plt.figure()
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        categorical_features = X_test.select_dtypes(include=['object']).columns.tolist()
        test_pool = Pool(X_test, cat_features=categorical_features)

        # Probabilities for positive class
        y_prob = model.predict_proba(test_pool)[:, 1]

        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

        # Remove last element to match thresholds length
        precision = precision[:-1]
        recall = recall[:-1]

        beta = params.get("beta")

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

        # Custom report (no F1, no support)
        report = {
            "precision": precision[best_idx],
            "recall": recall[best_idx],
            "fbeta": fbeta[best_idx]
        }

        # Plot precision-recall curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve (β={beta})")

        return report, fig

