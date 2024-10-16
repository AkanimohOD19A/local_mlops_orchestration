import mlflow
from mlflow import MlflowClient
from zenml.logger import get_logger
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Get the Mlflow client
logger = get_logger(__name__)


def get_model_by_tag(tag, model_name):
    logger.info(f"Fetching Model: {model_name} by Tag: {tag}")
    mlflowClient = MlflowClient()
    models = mlflowClient.search_model_versions(
        f"name='{model_name}'"
    )

    sorted_models = sorted(models, key=lambda x: x.version, reverse=True)

    latest_model = None
    rmse = None
    tag = str(tag)

    for model in sorted_models:
        if tag in model.tags:
            latest_model = model
            rmse = model.tags[tag]
            break

    if latest_model:
        print(f"Latest model with '{tag}' tag:")
        print(f"Version: {latest_model.version}")
        print(f"Tags: {latest_model.tags}")
        print(f"Run ID: {latest_model.run_id}")
        print(f"Staged Value: {rmse}")

        try:
            v = latest_model.version
            v_rmse = float(rmse)
            print(f"{v_rmse}")
            loaded_model = mlflow.pyfunc.load_model(
                f"models:/{model_name}/{latest_model.version}"
            )

            logger.info(f"Completed Fetch - Loaded Model: {model_name} | {v} with RMSE: {v_rmse}")

            return loaded_model, v_rmse, v
        except ValueError as e:
            print(f"{e}")
    else:
        print(f"No model found with '{tag}' tag")



