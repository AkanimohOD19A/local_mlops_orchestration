## Local MLOPs: ZenML, MLFlow and Streamlit

A simple Proof-Of-Concept that explores an orchestration pipeline that builds a ML Model and model deployment based off a promotion logic - Machine Learning Development and Ops - then retrieved for User Inference.

Libraries:
1. ZenML for orchestration
2. MLFlow for handling tag/aliasing deployed models
3. Streamlit for User interface

## Data
Dataset on Salary earnings and it is publicly hosted on github.

## Required Steps (ZenML Set-up)
Install zenml:
```commandline
pip install "zenml[server]"
```

Integrate mlflow: to register an MLflow Experiment Tracker and add it to your stack
```
zenml integrate mlflow -y
```

Start Afresh
`zenml clean -y`

Initiate Zenmlwith `zenml init`

Create a local runs directory for the artifact_store
`mkdir local_runs`
Create artifact_store
artifact-store: local
`zenml artifact-store register custom_local_store --flavor local --path="C:\Users\buasc\PycharmProjects\mlops_mastery\local_stores"`
Create experiment-tracker w/ mlflow flavour
`zenml experiment-tracker register custom_mlflow_experiment_tracker --flavor=mlflow`

SET Stack
w/. default orchestrator, custom: experiment tracker and artifact store
`zenml stack register local_mlops_stack -o default -e custom_mlflow_experiment_tracker -a custom_local_store  --set`

View Stack w/: `zenml stack list`


Our Steps:
1. Load Data
   - - 1. Ingest Data from Host
   - - 2. Create Local dir for Data
2. Train Model and store Model Artifact
   - - 1. Load Data
   - - 2. Split Data into train and test
   - - 3. Train Regressors*
   - - 4. Set Experiment Tracking parameters for ZenML/Mlflow
   - - 5. Export Model
3. Promotion Logic (Deploys Model)
4. User Interface - Streamlit

![image](https://github.com/user-attachments/assets/798922d6-c6fe-453a-86e2-0b26ad776588)

![image](https://github.com/user-attachments/assets/a542002b-a786-445b-979e-b5566b45c3f5)

RUN:
`python run_pipeline.py -m <model_name>`

Streamlit:
`streamlit run app.py`
![image](https://github.com/user-attachments/assets/897521b3-8c7a-4599-95a1-fa885edcb476)


