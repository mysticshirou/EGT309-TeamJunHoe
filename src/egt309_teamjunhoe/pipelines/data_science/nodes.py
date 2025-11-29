from typing import Any
from .models import *
from kedro_datasets.pickle import PickleDataset
import datetime
import os

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

def model_save(trained_model: Any, params: dict[Any, Any]):
    # Saves a model to the save model directory, doesn't return anything
    if params.get("save_model") == True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{params.get("save_model_name", "model")}_{timestamp}.pickle"
        path = os.path.join("saved_models", filename)
        pickler = PickleDataset(filepath=path, backend="pickle")
        pickler.save(trained_model)

        print(f"Saved model to {path}")