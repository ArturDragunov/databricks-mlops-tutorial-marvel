from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pathlib import Path
from marvel_characters.config import Tags
import re
import tempfile

def adjust_predictions(predictions: np.ndarray | list[int]) -> dict[str, list[str]]:
    """Adjust predictions to human-readable format."""
    return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}


class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for LightGBM model. This whole class is just for 1 additional step:
    for adjust_predictions() function."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the LightGBM model."""
        model_artifact = context.artifacts.get("lightgbm-pipeline")
        if model_artifact is None:
            raise RuntimeError("Artifact 'lightgbm-pipeline' not found")
    
        # Normalize path to handle Windows/Linux differences
        model_uri = Path(model_artifact).as_posix()
        self.model = mlflow.sklearn.load_model(model_uri)

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        """Predict the survival of a character."""
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)

    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ) -> None:
        """Log and register the model.

        :param wrapped_model_uri: URI of the wrapped model
        :param pyfunc_model_name: Name of the PyFunc model
        :param experiment_name: Name of the experiment
        :param tags: Tags for the model
        :param code_paths: List of code paths
        :param input_example: Input example for the model
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")
            # Use a looser Python pin so the serving build can resolve
            # an available Python patch version (e.g. 3.12.x) in the repo.
            conda_env = _mlflow_conda_env(
                additional_conda_deps=["python=3.12"],
                additional_pip_deps=additional_pip_deps,
            )

            signature = infer_signature(model_input=input_example, model_output={"Survival prediction": ["alive"]})
            
            # Download and normalize path
            downloaded = mlflow.artifacts.download_artifacts(wrapped_model_uri)
            local_path_str = Path(downloaded).as_posix()
            
            # Save locally first
            with tempfile.TemporaryDirectory() as tmpdir:
                save_dir = Path(tmpdir) / "pyfunc_model"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                mlflow.pyfunc.save_model(
                    path=str(save_dir.as_posix()),
                    python_model=self,
                    artifacts={"lightgbm-pipeline": local_path_str},
                    signature=signature,
                    code_paths=code_paths,
                    conda_env=conda_env,
                )
                
                # Fix MLmodel file
                mlmodel_path = save_dir / "MLmodel"
                if mlmodel_path.exists():
                    text = mlmodel_path.read_text(encoding="utf-8")
                    text = text.replace("\\", "/")
                    mlmodel_path.write_text(text, encoding="utf-8")
                
                # Register from local
                model_uri = f"file://{save_dir.as_posix()}"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=pyfunc_model_name,
                    tags=tags.to_dict(),
                )
        
        client = MlflowClient()
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version