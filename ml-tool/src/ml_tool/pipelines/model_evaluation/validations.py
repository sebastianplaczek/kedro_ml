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
from sklearn.inspection import PartialDependenceDisplay

from typing import Dict
import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.models import infer_signature


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
    def __init__(self, params, X, y, model, features):
        self.params = params
        self.X = X
        self.y = y
        self.model = model
        self.features = features

    def cross_validation(self):

        kf = KFold(n_splits=self.params["validation"]["validation_params"]["n_splits"])

        self.scores = {score: [] for score in self.params["model"]["model_scores"]}
        index = 0
        for train_index, test_index in kf.split(self.X):
            index += 1
            X_train, X_test = (
                self.X.iloc[train_index],
                self.X.iloc[test_index],
            )
            y_train, y_test = (
                self.y.iloc[train_index],
                self.y.iloc[test_index],
            )

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            for score_name in self.params["model"]["model_scores"]:
                metrics_function = metrics[score_name]
                score = round(metrics_function(y_test, y_pred) * 100, 2)
                self.scores[score_name].append(score)
                print(f"{score_name} : {score}")

            self.create_avg_metrics()

            if self.params["save_charts"]["roc"]:
                self.plot_roc(X_test, y_test, index)
            if self.params["save_charts"]["confusion_matrix"]:
                self.plot_conf_matrix(y_test, y_test, index)
            if self.params["save_charts"]["feature_importance"]:
                self.feature_importance(index)
            if self.params["save_charts"]["partial_dependence"]["plot"]:
                self.partial_dependence_plot(index)
        if self.params["save_charts"]["metrics"]:
            self.plot_metrics()

    def choose_validation(self):
        validations_dict = {"cross_validation": self.cross_validation}
        validations_dict[self.params["validation"]["validation_type"]]()

    def plot_roc(self, X_test, y_test, iteration):
        y_probs = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(self.model_output_path + "//" + f"roc_curve_{iteration}.png")

    def plot_conf_matrix(self, y_test, y_pred, iteration):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.savefig(self.model_output_path + "//" + f"confusion_matrix_{iteration}.png")

    def feature_importance(self, iteration):

        plt.figure(figsize=(10, 6))
        plt.barh(self.features, self.model.feature_importances_)
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance Plot")

        plt.savefig(
            self.model_output_path + "//" + f"feature_importance_{iteration}.png"
        )

    def partial_dependence_plot(self, iteration):
        categorical_features = self.params["save_charts"]["partial_dependence"][
            "cat_features"
        ]
        continuous_features = self.params["save_charts"]["partial_dependence"][
            "cont_features"
        ]

        categorical_indices = [
            self.X.columns.get_loc(col) for col in categorical_features
        ]

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Partial Dependence Plots")
        PartialDependenceDisplay.from_estimator(
            estimator=self.model,
            X=self.X,
            features=list(range(self.X.shape[1])),  # plot all features
            categorical_features=categorical_indices,  # categorical features indices
            random_state=5,
            ax=ax,
        )
        plt.savefig(
            self.model_output_path + "//" + f"partial_dependence_{iteration}.png"
        )

    def create_avg_metrics(self):
        self.avg_metrics = {}
        for score_name, score_list in self.scores.items():
            self.avg_metrics[score_name] = np.mean(score_list)

    def plot_metrics(self):
        plt.figure(figsize=(10, 6))
        for score_name, score_list in self.scores.items():
            avg = np.mean(score_list)
            splits = [str(x) for x in range(len(score_list))]
            plt.scatter(splits, score_list, label=f"{score_name}")
            plt.plot(splits, [avg for x in splits], label=f"{score_name}_avg")
        plt.ylabel("score")
        plt.xlabel("validation split number")
        plt.title(f"Validation metrics")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(self.model_output_path + "//" + "metrics.png")

    def create_folder_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Utworzono folder: {path}")
        else:
            print(f"Folder {folder_name} już istnieje.")

    def run_mlflow(self):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment(f"{self.params['model']['model_name']}")
        with mlflow.start_run():
            if self.params["model"]["model_params"] != "default":
                mlflow.log_params(self.params["model"]["model_params"])

            mlflow.log_metrics(self.avg_metrics)
            signature = infer_signature(self.X, self.model.predict(self.X))
            model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature,
                # input_example=self.X,
                # registered_model_name="tracking-quickstart",
            )

    def run(self):
        self.now = str(datetime.now()).replace(" ", "_").replace(":", "")
        current_directory = os.getcwd()
        folders = ["data", "output", self.now]
        self.model_output_path = os.path.join(current_directory, *folders)
        self.create_folder_if_not_exists(self.model_output_path)
        self.choose_validation()

        if self.params["mlflow"]:
            self.run_mlflow()
