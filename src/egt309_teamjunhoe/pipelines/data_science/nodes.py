from typing import Any
from .models import *
from kedro_datasets.pickle import PickleDataset
import datetime
import os

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Model Nodes
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def model_choice(params) -> object:
    # Obtain selected model as per configuration
    choice = params.get("model_choice")
    if choice not in MODEL_REGISTRY:
        raise ValueError(f'"{choice}" is not a valid model choice')
    return MODEL_REGISTRY[choice]()

def model_train(model: Model, X_train, y_train, params: dict[Any, Any]):
    # Train the model as per the selected model's train() method
    trained_model = model.train(X_train, y_train, params)
    return trained_model

def model_eval(model: Model, trained_model: Any, X_test, y_test, params: dict[Any, Any]):
    # Evaluate the model as per the selected model's eval() method
    report, plot = model.eval(trained_model, X_test, y_test, params)
    return report, plot

def model_save(trained_model: Any, params: dict[Any, Any]):
    # Saves a model to the save model directory, doesn't return anything
    # To save a model, set the "save_model" parameter to True inside parameters_datascience.yml and specify the saved model filename
    if params.get("save_model") == True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{params.get("save_model_name", "model")}_{timestamp}.pickle"
        path = os.path.join("saved_models", params.get("save_model_name", "model"), filename)
        pickler = PickleDataset(filepath=path, backend="pickle")
        pickler.save(trained_model)

        print(f"Saved model to {path}")