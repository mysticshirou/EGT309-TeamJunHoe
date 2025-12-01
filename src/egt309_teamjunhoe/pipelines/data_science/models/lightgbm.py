from .interfaces import Model
from .model_utils import read_bs_search_space, generate_report
import matplotlib.pyplot as plt

import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, fbeta_score

class LightGBM(Model):
    @staticmethod
    def train(X_train, y_train, params):
        # Create scorer for model
        fbeta_scorer = make_scorer(fbeta_score, beta=params.get("beta"))

        if params.get("lightgbm_auto_optimize") == True:
            search_space = read_bs_search_space(params.get("lightgbm_bayes_search_search_space", {}))
            assert len(search_space) > 0

            lgbm = lgb.LGBMClassifier(random_state=params.get("random_state"),
                                      objective=fbeta_scorer)
            clf = BayesSearchCV(
                lgbm,
                search_space,
                random_state=params.get("random_state"),
                scoring=fbeta_scorer,
                **params.get("lightgbm_bayes_search_settings", {})
            )
        else:
            clf = lgb.LGBMClassifier(random_state=params.get("random_state"),
                                     objective=fbeta_scorer,
                                     **params.get("lightgbm_setting", {}))
        clf.fit(X_train, y_train)
        return clf, plt.figure()

    @staticmethod
    def eval(model, X_test, y_test, params):
        # Probabilities for positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        # Predict classes
        y_pred = model.predict(X_test)

        report, fig = generate_report(y_test, y_prob, y_pred, params.get("beta"))

        return report, fig
