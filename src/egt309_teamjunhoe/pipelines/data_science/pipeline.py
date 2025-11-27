from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_choice, model_train, model_eval

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_choice,
                inputs=["bmarket_data", "params:cleaning_params"],
                outputs="cleaned_data",
                name="clean_dataset",
            ),
            node(
                func=encode_dataset,
                inputs=["cleaned_data", "params:encoding_params"],
                outputs="encoded_cleaned_data",
                name="encode_dataset",
            ),
            node(
                func=split_dataset,
                inputs=["encode_dataset", "params:splitting_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_dataset"
            )
        ]
    )