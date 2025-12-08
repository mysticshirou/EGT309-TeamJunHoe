from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_choice, model_train, model_eval, model_save

# DARREN
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_choice,
                inputs=["params:model_params"],
                outputs="model_choice",
                name="model_choice",
            ),
            node(
                func=model_train,
                inputs=["model_choice", "X_train", "y_train", "params:model_params"],
                outputs=["trained_model", "training_model_parameters"],
                name="model_train",
            ),
            node(
                func=model_eval,
                inputs=["model_choice", "trained_model", "X_test", "y_test", "params:model_params"],
                outputs=["evaluation_metrics", "evaluation_graphs"],
                name="model_eval"
            ),
            node(
                func=model_save,
                inputs=["trained_model", "params:model_params"],
                outputs=None,
                name="model_save"
            )
        ]
    )