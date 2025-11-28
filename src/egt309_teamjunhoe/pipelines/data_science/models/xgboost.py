from .interfaces import Model
from sklearn.metrics import f1_score
import xgboost as xgb

class XGBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        clf = xgb.XGBClassifier(random_state=params.get("random_state"),
                                **params.get("xgboost_setting", {}))
        clf.fit(X_train, y_train)
        return clf

    @staticmethod
    def eval(model, X_test, y_test, params):
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)
