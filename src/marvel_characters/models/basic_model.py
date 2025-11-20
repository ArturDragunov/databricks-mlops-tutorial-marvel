"""Basic model implementation for Marvel character classification.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict (Alive).
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
import pandas as pd
from delta.tables import DeltaTable
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from marvel_characters.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for Marvel character survival prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.marvel_character_model_basic"
        self.tags = tags.to_dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        
        Spark->Pandas->X/y split & Delta table + Data Version
        """
        logger.info("🔄 Loading data from Databricks tables...")
        # in lecture2 we already saved train and test datasets to DeltaTable.
        # now we load them using pyspark and transform to pandas
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        # eval_data is for mlflow evaluation step below
        self.eval_data = self.test_set[self.num_features + self.cat_features + [self.target]]

        # Delta tables are like Git for data - they track every version of your table.
        # Each time you modify/save the table, Delta creates a new version.
        #                                                     mlops_dev.marvel_characters.train_set
        # mlops_dev.marvel_characters we created manually initially, and train set was added in previous lectures
        # we take train_delta_table from the same place as train_set_spark. It's just used for diff purposes.
        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.train_set")
        # we are going to the history and take the most recent version
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")

        class CatToIntTransformer(BaseEstimator, TransformerMixin):
            """Transformer that encodes categorical columns as integer codes for LightGBM.

            Unknown categories at transform time are encoded as -1.
            
            We define class inside other class's method, because this way we don't need to deal\
                with private packages otherwise. Better way is to use pyfunc to deal with these dependencies.
            """

            def __init__(self, cat_features: list[str]) -> None:
                """Initialize the transformer with categorical feature names."""
                self.cat_features = cat_features
                self.cat_maps_ = {}

            def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
                """Fit the transformer to the DataFrame X."""
                self.fit_transform(X) # method defined below
                return self

            def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
                """Fit and transform the DataFrame X. It learns categorical features in data and
                transforms them into integers. In case unknown feature appears, it gets -1."""
                X = X.copy()
                for col in self.cat_features:
                    c = pd.Categorical(X[col])
                    # Build mapping: {category: code}
                    self.cat_maps_[col] = dict(zip(c.categories, range(len(c.categories)), strict=False))
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

            def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                """Transform the DataFrame X by encoding categorical features as integers."""
                X = X.copy()
                for col in self.cat_features:
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

        preprocessor = ColumnTransformer(
            transformers=[("cat", CatToIntTransformer(self.cat_features), self.cat_features)], remainder="passthrough"
        ) # we use sklearn Pipeline to define chain of steps
        self.pipeline = Pipeline( # lgbm has sklearn compatible format
            steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the model using MLflow. We set experiment and we start mlflow run.
        There inside we log our train/test sets (mlflow.log_input()), we 
        log the model (mlflow.sklearn.log_model()), and at the end we evaluate
        our logged model using default mlflow parameters.
        
        We later can find our logged model in databricks experiments section.
        """
        mlflow.set_experiment(self.experiment_name)
        # with run tags it will be possible to search for the run later
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id


            # The signature defines your model's input/output schema for MLflow tracking.
            # signature = infer_signature(
            # model_input=self.X_train, 
            # model_output=self.pipeline.predict(self.X_train)
            # )
            # It captures:
            # Input schema: column names and data types (e.g., APPEARANCES: long, ALIGN: string)
            # Output schema: prediction type (e.g., long for binary classification)

            # Why It's Needed:
            # When you deploy the model, MLflow uses the signature to:

            # Validate inputs at inference time (reject requests with wrong columns/types)
            # Document the API (show users what data format to send)
            # Enable model serving (auto-generate REST API endpoints with proper schema validation)

            # Without it, you'd have no runtime checks that incoming data matches what the model expects.
            signature = infer_signature(model_input=self.X_train,
                                        model_output=self.pipeline.predict(self.X_train))


            # MLflow logs which exact version of the data trained your model.
            # If someone updates the train_set table tomorrow,
            # you can still trace back that your model used version 3, not version 4
            #  Essential for:
            #     Debugging ("did the data change?")
            #     Reproducibility ("retrain on the exact same data")
            #     Auditing ("what data was this model trained on?")
            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.train_data_version, # ← logged for reproducibility. taken from DeltaTable
            )
            mlflow.log_input(train_dataset, context="training")
            
            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")
            
            # we use mlflow.sklearn, because Pipeline is from sklearn even though we use LGBM
            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model", # model folder (can be any name)
                signature=signature,
                input_example=self.X_test[0:1],
            )
            # evaluation metrics are computed separately and saved to result
            result = mlflow.models.evaluate(
                self.model_info.model_uri,
                self.eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )
            self.metrics = result.metrics

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-score.
        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        latest_model_uri = f"models:/{latest_model_version.model_id}"

        result = mlflow.models.evaluate(
            latest_model_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics
        if self.metrics["f1_score"] >= metrics_old["f1_score"]:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            # lightgbm-pipeline-model is the same as artifact_path in log_model
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags, # tags make it easy to trace it back to the code version we used
            # Idea is that by connecting git hash with tags we can match exactly the code with the model
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            # by setting this alias, it will be easier to retrieve the latest model later
            alias="latest-model", 
            version=latest_version,
        )
        return latest_version
