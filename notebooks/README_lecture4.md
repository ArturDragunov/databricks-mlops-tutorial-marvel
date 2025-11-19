# Lecture 4 README: MLflow Model Training & Registration Flow

## Overview
Lecture 4 demonstrates two MLflow workflows:
1. **Basic Model**: Train & register sklearn Pipeline with LightGBM
2. **Custom Model**: Wrap basic model in custom class for formatted output

---

## Part 1: Basic Model (`train_register_basic_model.py`)

### Flow Diagram
```
Config Load → Data Load → Feature Prep → Train → Log to MLflow → Register in UC
```

### Step-by-Step

**1. Setup (Lines 1-29)**
```python
# Connect to Databricks MLflow
mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load config and initialize
config = ProjectConfig.from_yaml("../project_config_marvel.yml", env="dev")
tags = Tags(git_sha="abcd12345", branch="main")  # for experiment tracking
basic_model = BasicModel(config=config, tags=tags, spark=spark)
```

**2. Load Data (Line 31-32)**
```python
basic_model.load_data()
```
- Loads train/test from Delta tables
- Extracts Delta table versions for lineage tracking
- Converts PySpark → Pandas
- Splits into X_train, y_train, X_test, y_test

**3. Prepare Features (Line 32)**
```python
basic_model.prepare_features()
```
- Creates sklearn Pipeline:
  - `CatToIntTransformer`: Encodes categorical features as integers for LightGBM
  - `LGBMClassifier`: The actual model
- Unknown categories get encoded as -1

**4. Train (Line 35)**
```python
basic_model.train()
```
- Fits Pipeline on training data
- No explicit evaluation here - happens in next step

**5. Log to MLflow (Line 38)**
```python
basic_model.log_model()
```
Creates MLflow run with:
- **Signature**: Input/output schema for API validation
- **Dataset lineage**: Links to Delta table versions
- **Model artifact**: Serialized Pipeline
- **Metrics**: F1, accuracy, etc. via `mlflow.evaluate()`

Stores in `self.run_id` and `self.model_info` for later use

**6. Inspect Logged Model (Lines 41-61)**
```python
# Load by model_id
logged_model = mlflow.get_logged_model(basic_model.model_info.model_id)
model = mlflow.sklearn.load_model(f"models:/{basic_model.model_info.model_id}")

# Or load by run_id
run_id = mlflow.search_runs(...).run_id[0]
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# Inspect logged datasets
run = mlflow.get_run(basic_model.run_id)
training_input = next((x for x in run.inputs.dataset_inputs if x.tags[0].value == 'training'), None)
training_source = mlflow.data.get_source(training_input)
training_source.load()  # retrieves actual training data from Delta
```

**7. Register Model (Line 63)**
```python
basic_model.register_model()
```
- Saves model to Unity Catalog: `catalog.schema.marvel_character_model_basic`
- Sets alias `"latest-model"` for easy retrieval
- Returns version number

**8. Search Models (Lines 66-75)**
```python
# Search by name (supported)
v = mlflow.search_model_versions(filter_string=f"name='{basic_model.model_name}'")

# Search by tag (NOT supported)
v = mlflow.search_model_versions(filter_string="tags.git_sha='abcd12345'")  # fails
```

---

## Part 2: Custom Model (`train_register_custom_model.py`)

### Flow Diagram
```
Load Basic Model → Create Wrapper → Log Wrapper → Register Wrapper → Test Predictions
```

### Step-by-Step

**1. Setup (Lines 1-28)**
```python
config = ProjectConfig.from_yaml(...)
tags = Tags(git_sha="abcd12345", branch="main")
marvel_characters_v = version("marvel_characters")

# Build wheel first: uv build
code_paths = [f"../dist/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]
```

**2. Retrieve Basic Model (Lines 30-35)**
```python
client = MlflowClient()
wrapped_model_version = client.get_model_version_by_alias(
  name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic",
  alias="latest-model"
)
```
Gets the URI of the basic model to wrap

**3. Load Test Data (Lines 37-39)**
```python
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
X_test = test_set[config.num_features + config.cat_features]
```

**4. Log & Register Wrapper (Lines 41-48)**
```python
wrapper = MarvelModelWrapper()
wrapper.log_register_model(
  wrapped_model_uri=f"models:/{wrapped_model_version.model_id}",
  pyfunc_model_name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom",
  experiment_name=config.experiment_name_custom,
  input_example=X_test[0:1],
  tags=tags,
  code_paths=code_paths
)
```

**What happens inside `log_register_model()` (custom_model.py):**

```python
# 1. Prepare conda environment with wheel
additional_pip_deps = [f"code/{whl_name}"]  # includes marvel_characters wheel
conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

# 2. Log wrapper as PyFunc model
model_info = mlflow.pyfunc.log_model(
  python_model=self,  # MarvelModelWrapper instance
  artifacts={"lightgbm-pipeline": wrapped_model_uri},  # links to basic model
  signature=infer_signature(...),
  code_paths=code_paths,  # includes wheel
  conda_env=conda_env
)

# 3. Register in Unity Catalog
registered_model = mlflow.register_model(
  model_uri=model_info.model_uri,
  name=pyfunc_model_name,
  tags=tags.to_dict()
)

# 4. Set alias for easy retrieval
client.set_registered_model_alias(
  name=pyfunc_model_name,
  alias="latest-model",
  version=latest_version
)
```

**5. Test Predictions (Lines 50-58)**
```python
# Load wrapper model
loaded_pyfunc_model = mlflow.pyfunc.load_model(f"models:/{pyfunc_model_name}@latest-model")

# Method 1: Unwrap and call directly
unwrapped_model = loaded_pyfunc_model.unwrap_python_model()
unwrapped_model.predict(context=None, model_input=X_test[0:1])
# Returns: {"Survival prediction": ["alive"]}

# Method 2: Call via MLflow interface
loaded_pyfunc_model.predict(X_test[0:1])
# Same output: {"Survival prediction": ["alive"]}
```

---

## How the Wrapper Works (custom_model.py)

### Class Structure
```python
class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
```

**`load_context()`** - Called once when model loads
```python
def load_context(self, context: PythonModelContext):
  # Load basic sklearn model from MLflow
  self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline"])
```
- `context.artifacts` contains the URI specified in `log_model()`
- Loads the basic model into `self.model`

**`predict()`** - Called for each inference request
```python
def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
  predictions = self.model.predict(model_input)  # [0, 1, 1]
  return adjust_predictions(predictions)  # {"Survival prediction": ["dead", "alive", "alive"]}
```

**`adjust_predictions()`** - Helper function
```python
def adjust_predictions(predictions):
  return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}
```

---

## Key Concepts

### Two Models in Unity Catalog
1. **`marvel_character_model_basic`**: Raw sklearn Pipeline, outputs [0, 1]
2. **`marvel_character_model_custom`**: Wrapper around basic, outputs {"Survival prediction": ["dead", "alive"]}

### Why Separate Models?
- **Basic**: ML logic, training, evaluation
- **Custom**: API interface, output formatting, business logic
- Can update formatting without retraining
- Can wrap multiple models with same interface

### Model URIs
```python
# By model ID
f"models:/{model_id}"

# By name and alias
f"models:/{catalog}.{schema}.{model_name}@{alias}"

# By run ID and artifact path
f"runs:/{run_id}/lightgbm-pipeline-model"
```

### Artifacts in Wrapper
```python
artifacts={"lightgbm-pipeline": wrapped_model_uri}
```
Links wrapper to basic model. When wrapper loads, it can access the basic model via `context.artifacts["lightgbm-pipeline"]`

### Code Paths & Wheel
```python
code_paths=[f"../dist/marvel_characters-{version}-py3-none-any.whl"]
```
Bundles custom code with model so `from marvel_characters.config import Tags` works on worker nodes during deployment

---

## Execution Order
```
1. train_register_basic_model.py
   → Creates & registers basic model
   
2. train_register_custom_model.py
   → Loads basic model
   → Wraps it
   → Registers wrapper as new model
```