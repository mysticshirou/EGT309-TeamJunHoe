from .interfaces import Model
from .registry import register_model
from .model_utils import generate_report, read_bs_search_space

from sklearn.metrics import make_scorer, fbeta_score
from skopt import BayesSearchCV

import xgboost as xgb
import pandas as pd


@register_model("xg_boost")
class XGBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        if params.get("xgboost_auto_optimize") is True:
            search_space = read_bs_search_space(
                params.get("xgboost_bayes_search_search_space", {})
            )
            assert len(search_space) > 0

            base_model = xgb.XGBClassifier(
                random_state=params.get("random_state")
            )

            clf = BayesSearchCV(
                base_model,
                search_space,
                random_state=params.get("random_state"),
                **params.get("xgboost_bayes_search_settings", {})
            )

            trained_model = clf.fit(X_train, y_train).best_estimator_

        else:
            clf = xgb.XGBClassifier(
                random_state=params.get("random_state"),
                **params.get("xgboost_setting", {})
            )
            trained_model = clf.fit(X_train, y_train)

        return trained_model, trained_model.get_params()

    @staticmethod
    def eval(model, X_test, y_test, params):
        # Probabilities for positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        report, fig = generate_report(y_test, y_prob, params)

        # Feature importances
        importances = model.feature_importances_
        feature_names = X_test.columns.tolist()
        feat_imp = {k: float(v) for k, v in zip(feature_names, importances)}
        report["feature_importance"] = feat_imp

        return report, fig
