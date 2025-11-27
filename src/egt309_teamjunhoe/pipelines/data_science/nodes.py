from interfaces import Model
from typing import Any
from sklearn.tree import DecisionTreeClassifier
from models import adaboost

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Model Nodes
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def model_choice(params) -> Model:
    match params.get("model_choice", None):
        case "adaboost":
            model = adaboost.Adaboost()
        # case "adaboost":
        #     model = d.DecisionTree()
        case _:
            raise ValueError(f"\"{params.get('model_choice'), None}\" is not a valid model choice")

    return model

def model_train(model: Model, X_train, y_train, params: dict[Any, Any]):
    trained_model = model.train(X_train, y_train, params)
    return trained_model

def model_eval(trained_model: Model, X_test, y_test, params: dict[Any, Any]):
    score = trained_model.eval(trained_model, X_test, y_test, params)
    return score