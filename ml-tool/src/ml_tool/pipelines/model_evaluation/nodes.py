"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.5
"""

from typing import Dict, List
import pandas as pd
from .models import models_dict
from .validations import Validate


def model_selection(params: Dict):

    model = models_dict[params["model"]["model_type"]]
    if params["model"]["model_params"] != "default":
        model.set_params(model_params)

    return model


def validation(params: Dict, X: pd.DataFrame, y: pd.Series, features: List, model):

    val = Validate(
        params,
        X,
        y,
        features,
        model,
    )
    val.run()

    return val.scores
