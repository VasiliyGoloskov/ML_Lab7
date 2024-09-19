from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_class: type = DecisionTreeRegressor,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators
        self.models: list = []
        self.gammas: list = []
        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.early_stopping_rounds: int = early_stopping_rounds
        self.plot: bool = plot
        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=True)
        x_subsample = x[indices]
        y_subsample = y[indices]

        model = self.base_model_class(**self.base_model_params)
        model.fit(x_subsample, y_subsample)

        new_predictions = model.predict(x)

        gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        self.models.append(model)
        self.gammas.append(gamma * self.learning_rate)

    def fit(self, x_train, y_train, x_valid, y_valid):
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if self.early_stopping_rounds is not None:
                if len(self.history['valid_loss']) > self.early_stopping_rounds:
                    if valid_loss > min(self.history['valid_loss'][-self.early_stopping_rounds:]):
                        print("Early stopping")
                        break

        if self.plot:
            import matplotlib.pyplot as plt
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['valid_loss'], label='Valid Loss')
            plt.legend()
            plt.show()

    def predict_proba(self, x):
        probabilities = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            probabilities += gamma * model.predict(x)

        prob_class_1 = self.sigmoid(probabilities)
        prob_class_0 = 1 - prob_class_1

        return np.vstack((prob_class_0, prob_class_1)).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        weighted_importances = np.array([gamma * model.feature_importances_ for gamma, model in zip(self.gammas, self.models)])

        return np.average(weighted_importances, axis=0)


