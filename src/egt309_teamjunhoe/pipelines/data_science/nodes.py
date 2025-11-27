from interfaces import Model
from typing import Any
from sklearn.tree import DecisionTreeClassifier
from models import adaboost

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Model Nodes
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def model_choice(params) -> Model:
    match params.get("model_choice", None):
        case "decision_tree":
            model = adaboost.AdaBoost()
        case "adaboost":
            model = decision_tree.DecisionTree()
        case _:
            raise ValueError(f"\"{params.get('model_choice'), None}\" is not a valid model choice")

    return model

# def model_train(model: Model, X_train, y_train, params):