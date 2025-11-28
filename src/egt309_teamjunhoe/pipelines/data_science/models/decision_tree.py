from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from skopt.space import Integer, Categorical
from skopt import BayesSearchCV

from .interfaces import Model
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _read_search_space(search_dict: dict[str, list]) -> dict[str, Any]:
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

class DecisionTree(Model):
    @staticmethod
    def train(X_train, y_train, params: dict[Any, Any]) -> Any:
        # Bayesion Optimiser to determine optimal decision tree parameters
        if params.get("decision_tree_auto_optimize") == True:
            search_space = _read_search_space(params.get("decision_tree_bayes_search_search_space", {}))
            assert len(search_space) > 0

            tree = DecisionTreeClassifier(random_state=params.get("random_state"))
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
        return trained_model

    @staticmethod
    def eval(model, X_test, y_test, params: dict[Any, Any]) -> Any:
        y_pred = model.predict(X_test)
        # Creating evaluation report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Creating classification report as matplotlib plot
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']], annot=True, cmap='viridis', fmt=".2f", ax=ax)
        ax.set_title('Classification Report Heatmap for Decision Tree')

        return report, fig