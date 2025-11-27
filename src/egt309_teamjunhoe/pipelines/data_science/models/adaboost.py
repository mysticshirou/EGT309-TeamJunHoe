from interfaces import Model

from sklearn.ensemble import AdaBoostClassifier

class Adaboost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        clf = AdaBoostClassifier(**params.get("adaboost_setting", dict()))
        clf.fit(X_train, y_train)
        return clf
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        score = model.score(X_test, y_test)
        return score