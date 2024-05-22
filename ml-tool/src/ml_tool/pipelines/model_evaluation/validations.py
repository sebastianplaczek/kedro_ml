from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.model_selection import KFold
from typing import Dict
import pandas as pd


def gini_normalized(y_test, y_pred):
    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
    return gini(y_test, y_pred) / gini(y_test, y_pred)


metrics = {
    # classification
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "gini": gini_normalized,
    # regression
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
}


class Validate:
    def __init__(self, params, X, y, model):
        self.params = params
        self.X = X
        self.y = y
        self.model = model

    def cross_validation(self):

        kf = KFold(n_splits=self.params["validation"]["validation_params"]["n_splits"])

        self.scores = {score: [] for score in self.params["model"]["model_scores"]}
        index = 0
        for train_index, test_index in kf.split(X):
            index += 1
            X_train, X_test = (
                X.iloc[train_index],
                X.iloc[test_index],
            )
            y_train, y_test = (
                y.iloc[train_index],
                y.iloc[test_index],
            )

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            for score_name in self.params["model"]["model_scores"]:
                metrics_function = metrics[score_name]
                score = round(metrics_function(y_test, y_pred) * 100, 2)
                self.scores[score_name].append(score)
                print(f"{score_name} : {score}")

    def choose_validation(self):
        validations_dict = {"cross_validation": self.cross_validation}
        validations_dict[self.params["validation"]["validation_type"]]
