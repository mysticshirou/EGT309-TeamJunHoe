from .interfaces import Model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns

class AdaBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        clf = AdaBoostClassifier(random_state=params.get("random_state"),
                                 **params.get("adaboost_setting", dict()))
        clf.fit(X_train, y_train)
        return clf, plt.figure()
    
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