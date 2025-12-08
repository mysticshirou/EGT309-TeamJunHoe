from typing import Protocol, Any
import pandas as pd
from matplotlib.figure import Figure

# HARISH
class Model(Protocol):
    # train method should return the model and the dictionary of all model parameters
    @staticmethod
    def train(X_train, y_train, params: dict[Any, Any]) -> tuple[Any, dict]:
        """
        Docstring for train
        
        :param X_train: Feature columns used for training
        :param y_train: Target column used for training
        :param params: Used by Kedro node to automatically pass in parameters
        :type params: dict[Any, Any]
        :return: trained model and parameters used by the model
        :rtype: tuple[Any, dict[Any, Any]]
        """
        ...
    
    # eval methods should evaluate the trained model and return the metrics report and plot
    @staticmethod
    def eval(model, X_test, y_test, params: dict[Any, Any]) -> tuple[dict, Figure]: 
        """
        Docstring for eval
        
        :param model: Trained model created by train method
        :param X_test: Feature columns used for testing
        :param y_test: Target column used for testing
        :param params: Used by Kedro node to automatically pass in parameters
        :type params: dict[Any, Any]
        :return: metrics report and metrics plot
        :rtype: tuple[dict[Any, Any], Figure]
        """
        ...