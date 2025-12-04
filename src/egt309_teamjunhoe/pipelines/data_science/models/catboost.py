from catboost import CatBoostClassifier, Pool
from .model_utils import generate_report
from skopt import BayesSearchCV
from egt309_teamjunhoe.pipelines.data_preprocessing.nodes import split_dataset
import yaml
import os
from .registry import register_model

from .interfaces import Model
import matplotlib.pyplot as plt
import seaborn as sns

@register_model("cat_boost")
class CatBoost(Model):
    @staticmethod
    def train(X_train, y_train, params):
        combined_df = X_train.copy()
        combined_df[y_train.name] = y_train

        # Split the training dataset into train and validation so we can evaluate best model 
        with open(os.path.join(os.getcwd(), "conf", "base", "parameters_datapreprocessing.yml"), "r") as yaml_file:
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

        return clf, clf.get_all_params()
    
    @staticmethod
    def eval(model, X_test, y_test, params):
        categorical_features = X_test.select_dtypes(include=['object']).columns.tolist()
        test_pool = Pool(X_test, cat_features=categorical_features)

        # Probabilities for positive class
        y_prob = model.predict_proba(test_pool)[:, 1]
        report, fig = generate_report(y_test, y_prob, params)

        importances = model.get_feature_importance(test_pool)
        feature_names = X_test.columns.tolist()
        feat_imp = {k: float(v) for k, v in zip(feature_names, importances)}
        report["feature_importance"] = feat_imp


        return report, fig

