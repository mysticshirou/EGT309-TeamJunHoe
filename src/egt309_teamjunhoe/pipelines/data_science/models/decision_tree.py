from sklearn.tree import DecisionTreeClassifier, plot_tree
from skopt import BayesSearchCV
from .model_utils import read_bs_search_space, generate_report

from .interfaces import Model
from .registry import register_model
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@register_model("decision_tree")
class DecisionTree(Model):
    # DARREN
    @staticmethod
    def train(X_train, y_train, params: dict[Any, Any]) -> Any:
        if params.get("decision_tree_auto_optimize") == True:
            # Bayesion Optimiser to determine optimal decision tree parameters
            search_space = read_bs_search_space(params.get("decision_tree_bayes_search_search_space", {}))
            assert len(search_space) > 0

            tree = DecisionTreeClassifier(random_state=params.get("random_state"),
                                          class_weight="balanced")
            model = BayesSearchCV(
                tree,
                search_space,
                random_state=params.get("random_state"),
                **params.get("decision_tree_bayes_search_settings", {})
            )
            trained_model = model.fit(X_train, y_train).best_estimator_

        else:
            model = DecisionTreeClassifier(random_state=params.get("random_state"), 
                                           **params.get("decision_tree_settings", {}))
            trained_model = model.fit(X_train, y_train)
        
        return trained_model, trained_model.get_params()
    
    @staticmethod
    def eval(model, X_test, y_test, params: dict[Any, Any]) -> Any:
        # Probabilities for positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        report, fig = generate_report(y_test, y_prob, params)
        
        # Add feature importances
        importances = model.feature_importances_
        feature_names = X_test.columns.tolist()
        feat_imp = {name: float(imp) for name, imp in zip(feature_names, importances)}
        report["feature_importance"] = feat_imp
        
        return report, fig
