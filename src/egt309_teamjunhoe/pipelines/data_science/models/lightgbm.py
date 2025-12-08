from .interfaces import Model
from .model_utils import read_bs_search_space, generate_report
from .registry import register_model
import matplotlib.pyplot as plt

import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, fbeta_score

@register_model("lightgbm")
class LightGBM(Model):
    # DARREN
    @staticmethod
    def train(X_train, y_train, params):
        if params.get("lightgbm_auto_optimize") == True:
            # Bayesion Optimiser to determine optimal hyperparameters
            search_space = read_bs_search_space(params.get("lightgbm_bayes_search_search_space", {}))
            assert len(search_space) > 0

            lgbm = lgb.LGBMClassifier(random_state=params.get("random_state"))
            clf = BayesSearchCV(
                lgbm,
                search_space,
                random_state=params.get("random_state"),
                **params.get("lightgbm_bayes_search_settings", {})
            )
            trained_model = clf.fit(X_train, y_train).best_estimator_
        else:
            clf = lgb.LGBMClassifier(random_state=params.get("random_state"),
                                     **params.get("lightgbm_setting", {}))
            trained_model = clf.fit(X_train, y_train)

        return trained_model, trained_model.booster_.dump_model()

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
