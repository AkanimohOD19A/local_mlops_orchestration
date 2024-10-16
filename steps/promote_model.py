from zenml import step, Model, log_model_metadata
from typing import Dict, Annotated
from zenml.logger import get_logger
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from dotenv import load_dotenv
import os
import mlflow
from mlflow import MlflowClient
from .fetch_tag import get_model_by_tag

# Get the Mlflow client
logger = get_logger(__name__)
load_dotenv()
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

mlflow.set_tracking_uri(mlflow_tracking_uri)
@step
def promote_model(
        model: Model
) -> Annotated[bool, "is_promoted"]:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflowClient = MlflowClient()

    # Search for model with the "staging" tag
    model_name = model.name
    tag_key = "staging"

    # Fetch Staging model
    staging_model, staging_rmse, staging_version = get_model_by_tag(
        tag=tag_key,
        model_name=model_name
    )

    logger.info(f"Model {staging_model} \n"                
                f"RMSE: {staging_rmse} \n"                
                f"Version: {staging_version}")

    #Fetch Production model
    # Initiate with None & Inf
    # production_model = None
    # production_rmse = float('inf')
    try:
        production_model, production_rmse, production_version = get_model_by_tag(
            tag="production",
            model_name=model_name
        )

        # Determine the new tag based on RMSE comparison
        if production_model:
            # Compare RMSEs
            if staging_rmse < production_rmse: # Staging is better, promote it to production
                # Change tag "production" - "archived"
                mlflowClient.delete_model_version_tag(
                    name=model_name,
                    version=str(production_version),
                    key="production"
                )
                mlflowClient.set_model_version_tag(
                    name=model_name,
                    version=str(production_version),
                    key="archived",
                    value=str(production_rmse)
                )
                # Change tag "staging" - "production"
                mlflowClient.delete_model_version_tag(
                    name=model_name,
                    version=str(staging_version),
                    key=tag_key
                )
                mlflowClient.set_model_version_tag(
                    name=model_name,
                    version=str(staging_version),
                    key="production",
                    value=str(staging_rmse)
                )
                print(f"Promoted staging model to production. Archived previous production model.")
                return True
            else:  # Production is better, archive staging
                mlflowClient.delete_model_version_tag(
                    name=model_name,
                    version=str(staging_version),
                    key=tag_key
                )
                mlflowClient.set_model_version_tag(
                    name=model_name,
                    version=str(staging_version),
                    key="archived",
                    value=str(staging_rmse)
                )
                return False
    except Exception as e:
        logger.warning(f"Error fetching  previous production model: {str(e)}")
        try:
            logger.info(f"Attempting: Auto Promotion of Staging to Production")
            mlflowClient.delete_model_version_tag(
                name=model_name,
                version=str(staging_version),
                key=tag_key
            )
            mlflowClient.set_model_version_tag(
                name=model_name,
                version=str(staging_version),
                key="production",
                value=str(staging_rmse)
            )
            logger.info(f"Completed: Auto Promotion of Staging to Production")
            return True

        except Exception as e:
            logger.warning(f"Error: Auto Promotion of Staging to Production")