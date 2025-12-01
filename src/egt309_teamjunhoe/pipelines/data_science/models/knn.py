from .interfaces import Model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from .model_utils import read_bs_search_space
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
            knn = KNeighborsClassifier()
            model = BayesSearchCV(
                knn,
                search_space,
                random_state=params.get("random_state"),
                **params.get("knn_bayes_search_settings", {})
            )
        else:
            model = KNeighborsClassifier(**params.get("knn_settings", dict()))
        model.fit(X_train, y_train)
        return model, model
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Creating classification report as matplotlib plot
        cf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ["False", "True"]
        sns.heatmap(cf_matrix, annot=True, fmt="d", ax=ax)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

        return report, fig