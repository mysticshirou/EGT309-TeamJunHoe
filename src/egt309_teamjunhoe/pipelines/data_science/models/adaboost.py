from .interfaces import Model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        clf = AdaBoostClassifier(**params.get("adaboost_setting", dict()))
        clf.fit(X_train, y_train)
        return clf
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)