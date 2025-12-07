from .interfaces import Model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, make_scorer
from skopt import BayesSearchCV
from .model_utils import read_bs_search_space, generate_report
from .registry import register_model

import matplotlib.pyplot as plt
import seaborn as sns

@register_model("ada_boost")
class AdaBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        if params.get("ada_boost_auto_optimize") == True:
            # Bayesion Optimiser to determine optimal hyperparameters
            search_space = read_bs_search_space(params.get("ada_boost_bayes_search_search_space", {}))
            assert len(search_space) > 0

            ada = AdaBoostClassifier(random_state=params.get("random_state"))
            clf = BayesSearchCV(
                ada,
                search_space,
                random_state=params.get("random_state"),
                **params.get("ada_boost_bayes_search_settings", {})
            )
            trained_model = clf.fit(X_train, y_train).best_estimator_
        else:
            clf = AdaBoostClassifier(random_state=params.get("random_state"),
                                     **params.get("adaboost_setting", dict()))
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