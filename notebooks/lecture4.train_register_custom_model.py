# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.custom_model import MarvelModelWrapper
from importlib.metadata import version
from dotenv import load_dotenv
from mlflow import MlflowClient
import os

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "main"})
marvel_characters_v = version("marvel_characters")

# for this one you need to run "uv build" first for a wheel
# Wheel packages the marvel_characters module for distribution to Databricks workers.
# Without it, imports like "from marvel_characters.config import ProjectConfig" fail
# because worker nodes don't have access to your local project files.
# Git/GitHub only syncs code to your development environment, not to execution workers.
# The wheel is logged with MLflow and installed on workers during model deployment.
code_paths=[f"../dist/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]

# COMMAND ----------
client = MlflowClient()
# we first saved basic model, and then we proceed with a custom model
wrapped_model_version = client.get_model_version_by_alias(
    name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic",
    alias="latest-model")
# Initialize model with the config path

# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
X_test = test_set[config.num_features + config.cat_features]

# COMMAND ----------
pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom"
wrapper = MarvelModelWrapper()
wrapper.log_register_model(wrapped_model_uri=f"models:/{wrapped_model_version.model_id}",
                           pyfunc_model_name=pyfunc_model_name,
                           experiment_name=config.experiment_name_custom,
                           input_example=X_test[0:1],
                           tags=tags,
                           code_paths=code_paths)

# COMMAND ----------
# unwrap and predict
loaded_pufunc_model = mlflow.pyfunc.load_model(f"models:/{pyfunc_model_name}@latest-model")
# by unwrapping we get back to our basic model
unwraped_model = loaded_pufunc_model.unwrap_python_model()

# COMMAND ----------
unwraped_model.predict(context=None, model_input=X_test[0:1])
# COMMAND ----------
# another predict function with uri
# this one can be run only in databricks. It is done for checking,
# if you get any errors in serving env. You need to run entire notebook
# in databricks the way it was shown during lecture 2.

# predictions = mlflow.models.predict(
#     f"models:/{pyfunc_model_name}@latest-model",
#     X_test[0:1]
# )


loaded_pufunc_model.predict(X_test[0:1])
# COMMAND ----------
