from zenml import pipeline
from steps.load_data import ingest_data
from steps.train_model import train_model
from steps.promote_model import promote_model

@pipeline
def simple_ml_pipeline(modelname: str):
    dataset = ingest_data()
    model = train_model(dataset, modelname)
    is_promoted = promote_model(model)

    return is_promoted


