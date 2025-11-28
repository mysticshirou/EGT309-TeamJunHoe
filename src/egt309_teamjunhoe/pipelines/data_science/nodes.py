from typing import Any
from .models import *

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Model Nodes
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def model_choice(params) -> Model:
    match params.get("model_choice", None):
        case "decision_tree":
            model = DecisionTree()
        case "ada_boost":
            model = AdaBoost()
        case "xg_boost":
            model = XGBoost()
        case "mlp":
            model = MLP()
        case _:
            raise ValueError(f"\"{params.get('model_choice'), None}\" is not a valid model choice")

    return model

def model_train(model: Model, X_train, y_train, params: dict[Any, Any]):
    trained_model = model.train(X_train, y_train, params)
    return trained_model

def model_eval(model: Model, trained_model: Any, X_test, y_test, params: dict[Any, Any]):
    report, plot = model.eval(trained_model, X_test, y_test, params)
    return report, plot