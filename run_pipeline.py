from commctrl import DTN_FORMAT

from pipelines.training_pipeline import simple_ml_pipeline
import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

tracking_uri = get_tracking_uri()

## Click Options
LR = "linearRegression"
DT = "decisionTree"
RF = "randomForest"

@click.command()
@click.option(
    "--modelname",
    "-m",
    type=click.Choice([LR, DT, RF]),
    default=LR,
    help="Optionally you can choose any of the models to train "
         "By default the Linear Regression Model will be used ",
)
def execute_pipe(modelname: str):
    # simple_ml_pipeline(modelname)
    simple_ml_pipeline(modelname)

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{tracking_uri}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )


if __name__ == "__main__":
    execute_pipe()