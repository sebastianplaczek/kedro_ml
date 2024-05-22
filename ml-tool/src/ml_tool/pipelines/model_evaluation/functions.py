def plot_roc(X_test, y_test, iteration):
        y_probs = model.predict_proba(X_test)[:, 1]
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
        plt.barh(self.columns, self.model.feature_importances_)
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance Plot")

        # Zapisz wykres do pliku graficznego
        plt.savefig(
            self.model_output_path + "//" + f"feature_importance_{iteration}.png"
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
        plt.xlabel("score")
        plt.ylabel("validation split number")
        plt.title(f"Validation metrics")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(self.model_output_path + "//" + "metrics.png")