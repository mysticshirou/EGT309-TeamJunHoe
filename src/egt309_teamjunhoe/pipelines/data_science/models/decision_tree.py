from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from .interfaces import Model
from typing import Any

class DecisionTree(Model):
    @staticmethod
    def train(X_train, y_train, params: dict[Any, Any]) -> Any:
        tree = DecisionTreeClassifier(**params.get("decision_tree_settings", {}))
        trained_model = tree.fit(X_train, y_train)
        return trained_model

    @staticmethod
    def eval(model, X_test, y_test, params: dict[Any, Any]) -> Any:
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)