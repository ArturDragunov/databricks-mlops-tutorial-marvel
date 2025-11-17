"""Configuration file for the Marvel characters project."""

from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.

    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    experiment_name_basic: str | None
    experiment_name_custom: str | None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]

            return cls(**config_dict)

# So this line:
# return cls(**config_dict)

# is equivalent to:

# return ProjectConfig(
#     num_features = config_dict["num_features"],
#     cat_features = config_dict["cat_features"],
#     target = config_dict["target"],
#     catalog_name = config_dict["catalog_name"],
#     schema_name = config_dict["schema_name"],
#     parameters = config_dict["parameters"],
#     experiment_name_basic = config_dict["experiment_name_basic"],
#     experiment_name_custom = config_dict["experiment_name_custom"]
# )

# Pydantic’s BaseModel automatically maps keyword arguments to the model’s fields,
# turning the YAML content into a fully instantiated object of type ProjectConfig.
# If the YAML contains extra keys not in the model,
# Pydantic will raise a validation error unless model_config.extra = "allow" is set.


class Tags(BaseModel):
    """Model for MLflow tags."""

    git_sha: str
    branch: str
    run_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert the Tags instance to a dictionary."""
        tags_dict = {}
        tags_dict["git_sha"] = self.git_sha
        tags_dict["branch"] = self.branch
        if self.run_id is not None:
            tags_dict["run_id"] = self.run_id
        return tags_dict
