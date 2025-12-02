from .interfaces import Model
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from .model_utils import read_bs_search_space, generate_report
from .registry import register_model

import matplotlib.pyplot as plt
import seaborn as sns

@register_model("knn")
class KNN(Model):
    @staticmethod
    def train(X_train, y_train, params):
        search_space = read_bs_search_space(params.get("knn_bayes_search_search_space", {}))
        assert len(search_space) > 0

        # Create scorer for model
        fbeta_scorer = make_scorer(fbeta_score, beta=params.get("beta"))

        if params.get("knn_auto_optimize") == True:
            knn = KNeighborsClassifier()
            model = BayesSearchCV(
                knn,
                search_space,
                scoring=fbeta_scorer,
                random_state=params.get("random_state"),
                **params.get("knn_bayes_search_settings", {})
            )
            trained_model = model.fit(X_train, y_train).best_estimator_
        else:
            model = KNeighborsClassifier(**params.get("knn_settings", dict()))
            trained_model = model.fit(X_train, y_train)

        return trained_model, trained_model.get_params()
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        # Probabilities for positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        # Predict classes
        y_pred = model.predict(X_test)

        report, fig = generate_report(y_test, y_prob, params)

        return report, fig