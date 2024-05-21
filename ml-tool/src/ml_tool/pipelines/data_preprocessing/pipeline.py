"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_dataframe,
                inputs="df",
                outputs="preprocessed_df",
                name="preprocess_df_node",
            ),
        ]
    )
