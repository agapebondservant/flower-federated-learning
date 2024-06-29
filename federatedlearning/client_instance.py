import warnings
import flwr as fl
from federatedlearning import model_utils
import logging
logging.basicConfig(level=logging.INFO)


class ClientInstance(fl.client.NumPyClient):
    def __init__(self, model, modelutil: model_utils.ModelUtils, x_train, x_test, y_train, y_test):
        self.model = model
        self.modelutil = modelutil
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self, config):  # type: ignore
        return self.modelutil.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        logging.debug(f"Starting fit() in server...")
        self.modelutil.set_model_params(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        logging.info(f"Training finished for round {config['server_round']}.")
        return self.modelutil.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        logging.debug(f"Starting evaluate() in server...")
        return self.modelutil.evaluate(self.model, parameters, config, self.X_test, self.y_test)

    def start(self, server_address: str):  # type: ignore
        logging.info(f"Starting flower client...")
        # Start Flower client
        fl.client.start_client(
            server_address=server_address, client=self.to_client()
        )
