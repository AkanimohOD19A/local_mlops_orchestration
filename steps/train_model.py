# 2. Train Model and store Model Artifact
#    3. Load Data
#    4. Split Data into train and test
#    5. Train Regressors*
#    6. Set Experiment Tracking parameters for ZenML/Mlflow
#    7. Export Model

# Load Libraries
import os
from zenml import step, Model
from zenml.client import Client
from zenml.logger import get_logger
import pandas as pd
from typing import Union, Annotated
from datetime import date
import joblib
# ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.client import MlflowClient
from mlflow.models import infer_signature

experiment_tracker = Client().active_stack.experiment_tracker
model_dir = "./model_dir"
logging = get_logger(__name__)
today = date.today()


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
        data: pd.DataFrame,
        model_name: Union[str, None]
) -> Annotated[Model, "Trained_Model"]:
    y = data['Salary']
    X = data[['Experience Years']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.7,
                                                        random_state=234)

    model = None
    if model_name == "linearRegression" or model_name is None:
        model = LinearRegression()
    elif model_name == "decisionTree":
        model = DecisionTreeRegressor()
    elif model_name == "randomForest":
        model = RandomForestRegressor()

    mlflowClient = MlflowClient()

    # Log the model type
    mlflow.log_param("model_type", model_name)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    signature = infer_signature(X_test, y_pred)

    # Log Metrics
    mlflow.log_metric("rmse", rmse)

    # Log metadata directly to the MlFlow Model Registry
    registered_model_name = "salary_prediction_regression-model"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name=registered_model_name,
    )

    # Get the model metadata from the run
    run_id = mlflow.active_run().info.run_id
    latest_versions = mlflowClient.search_model_versions(f"name='{registered_model_name}'")
    model_version = max(int(v.version) for v in latest_versions)

    logging.info(f"Run ID: {run_id}, Model Version: {model_version}")

    # Set registered model tag
    mlflowClient.set_registered_model_tag(
        name=registered_model_name,
        key="staged-to-prod",
        value="regression"
    )

    # Set model version tag
    tag_key = "staging"
    mlflowClient.set_model_version_tag(
        name=registered_model_name,
        version=str(model_version),
        key=tag_key,
        value=rmse
    )

    # Create a ZenML Model object
    zenml_model = Model(
        name=registered_model_name,
        model=model,
        metadata={"rmse": str(rmse),
                  "stage": tag_key}
    )

    ## Log metadata directly to the zenml_model object
    zenml_model.log_metadata({"rmse": str(rmse), "stage": str(tag_key)})
    zenml_model.set_stage(tag_key, force=True)

    # Export Model to Local Directory
    model_dir_name = f"{rmse}_{model_name}_{today}.pkl"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_dir_name)

    # Dump model
    joblib.dump(model, model_path)

    # Log the model path as an artifact
    # mlflow.log_artifact(model_path)
    logging.info(f"Successfully Trained {model_name} \n"
                 f"and stored trained model to Model Register: {model_path}")

    return zenml_model
