import warnings
from dotenv import load_dotenv
import os
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
from sqlalchemy import create_engine

from samples.random_forest import random_forest_utils
from federatedlearning import client_instance

logging.basicConfig(level=logging.INFO)

load_dotenv()

if __name__ == "__main__":

    # Load Data
    cnx = create_engine(os.getenv('TRAINING_DB_URI_FULL'))
    logging.debug(f"Training db is {os.getenv('TRAINING_DB_URI_FULL')}")
    df = pd.read_sql_query(f"select * from \"rf_credit_card_transactions_training\"", cnx)
    X, y = df[["time_elapsed", "amt", "lat", "long"]].to_numpy(), df[["is_fraud"]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create Model
    model = RandomForestClassifier(
        random_state=1,
        n_estimators=5,
        class_weight='balanced',
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Start client
    helper = random_forest_utils.RandomForestUtils()
    helper.set_initial_params(model, X_test, y_test)
    client = client_instance.ClientInstance(model, helper, X_train, X_test, y_train, y_test)
    client.start(os.getenv('FLOWER_SERVER_ADDRESS'))
