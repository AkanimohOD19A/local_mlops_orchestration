import streamlit as st
import mlflow
import pickle
import os
import pandas as pd
from steps.fetch_tag import get_model_by_tag

# mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri("file:C:/Users/buasc/PycharmProjects/mlops_mastery/local_stores/mlruns")
# st.write(mlflow_tracking_uri)
## Fetches Production Artifacts
@st.cache_resource
def load_production_model():
    model_name = "salary_prediction_regression-model"

    try:
        production_model, production_rmse, production_version = get_model_by_tag(
            tag="production",
            model_name=model_name
        )
        st.success(f"Successfully fetched production model from model registry | RMSE: {production_rmse}")

        return production_model
    except Exception as e:
        st.error("No Production Model found!")
        try:
            local_production_model = "model_dir/6311.262456108905_linearRegression_2024-10-16.pkl"
            with open(local_production_model, "rb") as file:
                production_model = pickle.load(file)
            st.warning("Loaded from Reserves")

            return production_model
        except Exception as e:
            st.error("No model in reserves!")


# Call the load production function
production_model = load_production_model()

## A User Input widget
st.title("SALARY Prediction Application")
## Write Prediction
work_experience = st.number_input("Experience Years",
                                  min_value=0,
                                  value=5,
                                  step=1)

if st.button("Predict"):
    if production_model is not None:
        # Create a Dataframe
        input_data = pd.DataFrame([[float(work_experience)]], columns=["Experience Years"])
        prediction = production_model.predict(input_data)

        st.write(f"Prediction: {prediction[0]}")
    else:
        st.error("No model available for prediction")
