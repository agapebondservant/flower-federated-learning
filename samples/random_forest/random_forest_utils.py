import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

from flwr.common import NDArrays

from federatedlearning import model_utils

import logging
logging.basicConfig(level=logging.INFO)


class RandomForestUtils(model_utils.ModelUtils):
    def get_model_parameters(self, model) -> NDArrays:
        """Returns the parameters of a sklearn RandomForestClassifier model."""
        params = [
            model.classes_,
            model.n_classes_,
            model.n_features_in_,
        ]
        if model.oob_score:
            params.append(model.oob_score_)
        logging.debug(f"Model parameters: {params}")
        return params

    def set_model_params(self, model, params: NDArrays):
        """Sets the parameters of a sklearn RandomForestClassifier model."""
        logging.debug(f"Setting model parameters: {params}")
        model.classes_ = params[0]
        model.n_classes_ = params[1]
        model.n_features_in_ = params[2]
        if model.oob_score:
            model.oob_score_ = params[3]
        return model

    def set_initial_params(self, model, x: np.ndarray, y: np.ndarray):
        """Sets initial parameters as zeros; Required since model params are uninitialized
        until model.fit is called.
        """
        logging.debug(f"Setting initial params: x={x}, y={y}")
        base_model = self._build_initial_model(model, x, y)
        self.set_model_params(model, self.get_model_parameters(base_model))

    def _build_initial_model(self, model, x: np.ndarray, y: np.ndarray):
        model.fit(x, y)
        return model

    def evaluate(self, model, parameters, config, x_test, y_test):  # type: ignore
        self.set_model_params(model, parameters)
        y_pred = model.predict(x_test)
        report = pd.DataFrame(classification_report(y_pred, y_test, output_dict=True)).transpose()
        scores = {}
        for label in model.classes_:
            scores[f"precision_{label}"] = report[['precision']].loc[str(label)]
            scores[f"recall_{label}"] = report[['recall']].loc[str(label)]
            scores[f"f1_{label}"] = report[['f1-score']].loc[str(label)]
        loss = log_loss(y_test, model.predict_proba(x_test))
        logging.info(f"Evaluation results: loss={loss}, count={len(x_test)}, scores={scores}")
        return loss, scores
