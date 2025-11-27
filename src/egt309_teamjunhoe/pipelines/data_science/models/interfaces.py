from typing import Protocol, Any
import pandas as pd

class Model(Protocol):
    @staticmethod
    def train(X_train, y_train, params: dict[Any, Any]) -> Any: ...
    
    @staticmethod
    def eval(model, X_test, y_test, params: dict[Any, Any]) -> Any: ...