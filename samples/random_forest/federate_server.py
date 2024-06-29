from sklearn.ensemble import RandomForestClassifier

import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine
from samples.random_forest import random_forest_utils
from federatedlearning import server_instance
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Start Flower server
if __name__ == "__main__":
    # Load Data
    logging.info("Loading data...")
    cnx = create_engine(os.getenv('TRAINING_DB_URI_FULL'))
    logging.debug(f"Training db is {os.getenv('TRAINING_DB_URI_FULL')}")
    df = pd.read_sql_query(f"select * from \"rf_credit_card_transactions_training\"", cnx)
    X, y = df[["time_elapsed", "amt", "lat", "long"]].to_numpy(), df[["is_fraud"]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    logging.info(f"Data loaded: len(X_train)={len(X_train)}, len(y_train)={len(y_train)}")

    # Create Model
    logging.info("Creating base model...")
    model = RandomForestClassifier(
        random_state=1,
        n_estimators=5,
        class_weight='balanced',
        warm_start=True,  # prevent refreshing weights when fitting
    )
    logging.info(f"Base model created: {model}")

    # Start server
    helper = random_forest_utils.RandomForestUtils()
    helper.set_initial_params(model, X_test, y_test)
    server = server_instance.ServerInstance(model, helper, X_test, y_test)
    server.start(os.getenv('FLOWER_SERVER_ADDRESS'),
                 num_rounds=int(os.getenv('FLOWER_NUM_ROUNDS')),
                 min_available_clients=int(os.getenv('FLOWER_MIN_AVAILABLE_CLIENTS')))
