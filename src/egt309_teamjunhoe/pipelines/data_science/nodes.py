from typing import Any
from .models import *
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Model Nodes
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def model_choice(params) -> Model:
    match params.get("model_choice", None):
        case "decision_tree":
            model = DecisionTree()
        case "ada_boost":
            model = AdaBoost()
        case _:
            raise ValueError(f"\"{params.get('model_choice'), None}\" is not a valid model choice")

    return model

def model_train(model: Model, X_train, y_train, params: dict[Any, Any]):
    trained_model = model.train(X_train, y_train, params)
    return trained_model

def model_eval(model: Model, trained_model: DecisionTreeClassifier, X_test, y_test, params: dict[Any, Any]):
    score = model.eval(trained_model, X_test, y_test, params)
    print("\033[92m" + f"Score: {score}" + "\033[0m")
    return score