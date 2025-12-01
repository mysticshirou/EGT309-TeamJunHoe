from .interfaces import Model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, make_scorer
from skopt import BayesSearchCV
from .model_utils import read_bs_search_space, generate_report

import matplotlib.pyplot as plt
import seaborn as sns

class AdaBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        if params.get("ada_boost_auto_optimize") == True:
            search_space = read_bs_search_space(params.get("ada_boost_bayes_search_search_space", {}))
            assert len(search_space) > 0

            # Create scorer for model
            fbeta_scorer = make_scorer(fbeta_score, beta=params.get("beta"))

            ada = AdaBoostClassifier(random_state=params.get("random_state"))
            clf = BayesSearchCV(
                ada,
                search_space,
                random_state=params.get("random_state"),
                scoring=fbeta_scorer,
                **params.get("ada_boost_bayes_search_settings", {})
            )
            trained_model = clf.fit(X_train, y_train).best_estimator_
        else:
            clf = AdaBoostClassifier(random_state=params.get("random_state"),
                                     **params.get("adaboost_setting", dict()))
            trained_model = clf.fit(X_train, y_train)
        
        return trained_model, plt.figure()
    
    @staticmethod
    def eval(model, X_test, y_test, params):       
        # Probabilities for positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        # Predict classes
        y_pred = model.predict(X_test)

        report, fig = generate_report(y_test, y_prob, y_pred, params.get("beta"))

        return report, fig