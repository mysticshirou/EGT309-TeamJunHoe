from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_dataset, encode_dataset, split_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_dataset,
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
                inputs=["encoded_cleaned_data", "params:splitting_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_dataset"
            )
        ]
    )