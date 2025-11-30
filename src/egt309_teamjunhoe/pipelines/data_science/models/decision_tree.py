from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from skopt import BayesSearchCV
from .model_utils import read_bs_search_space

from .interfaces import Model
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DecisionTree(Model):
    @staticmethod
    def train(X_train, y_train, params: dict[Any, Any]) -> Any:
        # Bayesion Optimiser to determine optimal decision tree parameters
        if params.get("decision_tree_auto_optimize") == True:
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
        else:
            model = DecisionTreeClassifier(random_state=params.get("random_state"), 
                                           **params.get("decision_tree_settings", {}))
            
        
        trained_model = model.fit(X_train, y_train)

        # Plot decision tree
        fig = plt.figure(figsize=(12, 8))
        plot_tree(trained_model.best_estimator_,
                  filled=True) 
        
        return trained_model.best_estimator_, fig
    @staticmethod
    def eval(model, X_test, y_test, params: dict[Any, Any]) -> Any:
        y_pred = model.predict(X_test)
        # Creating evaluation report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Creating classification report as matplotlib plot
        cf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ["False", "True"]
        sns.heatmap(cf_matrix, annot=True, fmt="d", ax=ax)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        return report, fig