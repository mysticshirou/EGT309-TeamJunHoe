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

        if params.get("knn_auto_optimize") == True:
            # Bayesion Optimiser to determine optimal hyperparameters
            knn = KNeighborsClassifier()
            model = BayesSearchCV(
                knn,
                search_space,
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
        report, fig = generate_report(y_test, y_prob, params)

        return report, fig