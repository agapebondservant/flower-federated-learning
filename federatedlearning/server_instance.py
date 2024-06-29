import flwr as fl
from typing import Dict
from federatedlearning import model_utils
import logging
logging.basicConfig(level=logging.INFO)


class ServerInstance(fl.client.NumPyClient):
    def __init__(self, model, modelutil: model_utils.ModelUtils, x_test, y_test):
        self.model = model
        self.modelutil = modelutil
        self.X_test = x_test
        self.y_test = y_test

    def fit_round(self, server_round: int) -> Dict:
        """Send round number to client."""
        logging.debug(f"Starting fit_round() in server...")
        return {"server_round": server_round}

    def get_evaluate_fn(self):  # type: ignore
        """Return an evaluation function for server-side evaluation."""
        # The `evaluate` function will be called after every round
        def evaluate(server_round, parameters: fl.common.NDArrays, config):
            logging.debug(f"Starting evaluate() in server...")
            return self.modelutil.evaluate(self.model, parameters, config, self.X_test, self.y_test)

        return evaluate

    def start(self, server_address: str, num_rounds: int, min_available_clients: int):  # type: ignore
        logging.info(f"Starting flower server...")
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=min_available_clients,
            evaluate_fn=self.get_evaluate_fn(),
            on_fit_config_fn=self.fit_round,
        )
        # Start Flower server
        fl.server.start_server(
            server_address=server_address,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
        )
