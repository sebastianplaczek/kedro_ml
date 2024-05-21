"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.5
"""

from typing import Dict
from .models import models_dict


def model_selection(model_type: str, model_params: [str, Dict]):

    model = models_dict[model_type]
    if model_params != "default":
        model.set_params(model_params)

    return model
