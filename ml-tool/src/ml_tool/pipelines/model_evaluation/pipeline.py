"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import model_selection, validation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["params:parameters"],
                outputs="model",
                name="model_selection",
            ),
            node(
                func=validation,
                inputs=["params:parameters", "X", "y", "features", "model"],
                outputs="scores",
                name="validation",
            ),
        ]
    )
