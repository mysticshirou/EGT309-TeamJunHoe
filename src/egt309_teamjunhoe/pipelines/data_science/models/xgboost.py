from .interfaces import Model
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        clf = xgb.XGBClassifier(random_state=params.get("random_state"),
                                **params.get("xgboost_setting", {}))
        clf.fit(X_train, y_train)
        return clf, plt.figure()

    @staticmethod
    def eval(model, X_test, y_test, params):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Creating classification report as matplotlib plot
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']], annot=True, cmap='viridis', fmt=".2f", ax=ax)
        ax.set_title('Classification Report Heatmap for Decision Tree')

        return report, fig
