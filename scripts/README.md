Completely the same logic as we had in MSD -> you define scripts, which go one by one in a sequential pattern.
You create a Databricks workflow, and run those tasks in a job.
Here though we are using mlflow for tracking, logging and registering.
This is the flow for retraining/recreating a model and redeploying it to production.
- process_data.py is based on L2;
- train_register_custom_model.py is based on L3/L4
- deploy_model is based on L6