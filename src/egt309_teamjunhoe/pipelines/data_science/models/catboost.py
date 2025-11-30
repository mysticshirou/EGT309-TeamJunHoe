from catboost import CatBoostClassifier, Pool
from .model_utils import read_bs_search_space
from sklearn.metrics import classification_report, confusion_matrix
from skopt import BayesSearchCV
from egt309_teamjunhoe.pipelines.data_preprocessing.nodes import split_dataset
import yaml
import os

from .interfaces import Model
import matplotlib.pyplot as plt
import seaborn as sns

class CatBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        combined_df = X_train.copy()
        combined_df[y_train.name] = y_train

        # Split the training dataset into train and validation so we can evaluate best model 
        with open(os.path.join(os.getcwd(), "conf", "base", "parameters.yml"), "r") as yaml_file:
            split_parameters = yaml.safe_load(yaml_file)["splitting_params"]

        X_re_train, X_eval, y_re_train, y_eval = split_dataset(combined_df, split_parameters)   
        categorical_features = X_re_train.select_dtypes(include=['object']).columns.tolist()
        train_pool = Pool(X_re_train, y_re_train, cat_features=categorical_features)
        eval_pool = Pool(X_eval, y_eval, cat_features=categorical_features)
        
        if params.get("cat_boost_auto_optimize") == True:
            search_space = params.get("cat_boost_grid_search_search_space", {})
            assert len(search_space) > 0

            clf_params = params.get("cat_boost_settings", {})
            clf = CatBoostClassifier(random_state=params.get("random_state"),
                                     **clf_params)
            results = clf.grid_search(search_space, train_pool)

            clf = CatBoostClassifier(random_state=params.get("random_state"),
                                     **{**clf_params, **results["params"]})
        else:
            clf = CatBoostClassifier(random_state=params.get("random_state"),
                                    **params.get("cat_boost_settings", dict()))
        
        clf.fit(train_pool, eval_set=eval_pool, use_best_model=True)

        return clf, plt.figure()
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        categorical_features = X_test.select_dtypes(include=['object']).columns.tolist()
        test_pool = Pool(X_test, cat_features=categorical_features)
        
        y_pred = model.predict(test_pool)
        # Creating evaluation report
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