# Load Dependencies
import os
from zenml import step
import pandas as pd
from typing import Annotated

df_pth = "https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv"
local_pth = "./datasets/SalaryData.csv"


@step(enable_cache=True)
def ingest_data() -> Annotated[pd.DataFrame, "Salary_Data"]:
    if os.path.exists(local_pth):
        df = pd.read_csv(local_pth)
    else:
        os.makedirs("./datasets", exist_ok=True)
        df = pd.read_csv(df_pth)
        df.to_csv("./datasets/SalaryData.csv", index=False)

    return df
