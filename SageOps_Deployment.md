# SageMaker deployment — concise summary (with full Path B table & glossary)

## 1) Core concepts (quick definitions)

- **Training vs. inference:** Training builds a model; **inference** runs that model to make predictions.
- **Inference container:** A Docker image (prebuilt or custom) that knows how to load your model and serve predictions via an HTTP endpoint.
- **Model artifacts:** The serialized files produced by training (for example, `.pt`, `.pkl`) packaged as `model.tar.gz`, typically stored in **Amazon Simple Storage Service (Amazon S3)** so endpoints and jobs can pull them easily.
- **Amazon Elastic Container Registry (Amazon ECR) image:** Your Docker image stored in Amazon ECR so **Amazon SageMaker** can pull it for training or serving.
- **Endpoint & endpoint configuration:** An **endpoint** is the live HTTPS prediction service; the **endpoint configuration** defines instance type (or serverless memory), model variants, and traffic weights.

---

## 2) Path A (Studio/SDK flow) — nutshell

Use the **Amazon SageMaker** Python SDK (or console) to:
1. Package inference (prebuilt container or custom **Amazon ECR** image).
2. Upload **model artifacts** to **Amazon S3** (`model.tar.gz`).
3. Create a **SageMaker Model** (bind S3 artifacts + container image).
4. Create an **Endpoint Configuration** (instance type or serverless settings; optional variants/weights).
5. Deploy an **Endpoint** (real-time, serverless, or asynchronous).
6. (Optional) Add autoscaling, shadow tests, blue/green rollout, and monitoring.

---

## 3) Serving options — when to use what

- **Real-time endpoint (provisioned instances):** Always-on, low latency; best for steady interactive traffic.
- **Serverless endpoint:** Scales to zero; ideal for spiky or low-traffic APIs where cold starts are acceptable.
- **Asynchronous inference:** Queued, longer-running jobs with inputs/outputs in **Amazon S3**; not for interactive latency.
- **Amazon SageMaker Batch Transform (batch job, not an endpoint):** Bulk or scheduled scoring from **Amazon S3** to **Amazon S3** that shuts down when done.
- **Amazon SageMaker Processing jobs:** Great for extract-transform-load (**ETL**), evaluation, and report assembly around inference.

**Daily report from yesterday’s data?** Prefer **Amazon SageMaker Batch Transform** for scoring + a **Processing** step to aggregate and render the report, orchestrated by a **SageMaker Pipeline** and scheduled with **Amazon EventBridge**.

---

## 4) Path B (“SageOps”: production Continuous Integration/Continuous Deployment (CI/CD))

“SageOps” is an opinionated way to run MLOps on **Amazon SageMaker** using **SageMaker Pipelines**, **SageMaker Projects**, **SageMaker Model Registry**, infrastructure as code, approvals, safe rollouts, and monitoring—typically across separate development, staging, and production accounts.

### Path B summary table (steps, services, purposes)

| # | Step name | AWS services used | Purpose |
|---|---|---|---|
| 0 | Bootstrap & foundations | **AWS Identity and Access Management (IAM)**, **AWS Key Management Service (AWS KMS)**, **Amazon Virtual Private Cloud (Amazon VPC)**, **AWS CloudFormation**, **AWS Cloud Development Kit (AWS CDK)**, Terraform | Establish permissions, encryption, networking, and infrastructure as code. |
| 1 | Repos & project scaffold | **Amazon SageMaker Projects**, **AWS CodeCommit**, **AWS CodePipeline** | Create repositories and CI/CD wiring between Git and SageMaker. |
| 2 | Build training image (optional) | **Amazon Elastic Container Registry (Amazon ECR)**, **AWS CodeBuild** | Containerize training code/dependencies and store images in Amazon ECR. |
| 3 | Data ingest & validation | **Amazon Simple Storage Service (Amazon S3)**, **AWS Glue**, **Amazon Athena**, **Amazon SageMaker Processing**, **Amazon SageMaker Feature Store** (optional) | Land, clean, and validate data; compute/persist features. |
| 4 | Train | **Amazon SageMaker Training**, Amazon S3 | Run training jobs; produce `model.tar.gz` in Amazon S3. |
| 5 | Evaluate & report | Amazon SageMaker Processing, Amazon S3 | Score validation data and write metrics (for example, `evaluation.json`). |
| 6 | Metric gate | **Amazon SageMaker Pipelines** | Stop or continue the pipeline based on metric thresholds. |
| 7 | Register model | **Amazon SageMaker Model Registry** | Version the model with metrics/metadata; set status to *PendingManualApproval*. |
| 8 | Approval gate | **AWS CodePipeline** (manual approval), Console/API | Human (or policy) flips status to *Approved* before release. |
| 9 | Build inference image (optional) | Amazon ECR, AWS CodeBuild | Package custom serving container (or use a prebuilt image). |
| 10 | Deploy to staging | **Amazon SageMaker Endpoints** (real-time/serverless/asynchronous) or **Amazon SageMaker Batch Transform**, **Amazon CloudWatch** | Stand up staging deployment and capture logs/metrics. |
| 11 | Guardrails & rollout tests | **Amazon SageMaker Deployment Guardrails** (shadow, canary, linear, blue/green), **Amazon CloudWatch Alarms** | Safely test new model and shift traffic gradually. |
| 12 | Promote to production (multi-account) | Amazon SageMaker Model Registry (cross-account), AWS CodePipeline, **AWS Security Token Service (AWS STS)**, AWS KMS, Amazon S3 | Move the Approved package to the production account and deploy. |
| 13 | Monitoring & drift | **Amazon SageMaker Model Monitor**, Amazon CloudWatch, **Amazon EventBridge**, **Amazon Simple Notification Service (Amazon SNS)** | Track data/quality/latency; schedule checks; alert on issues. |
| 14 | Lineage & audit | **Amazon SageMaker Lineage**, **AWS CloudTrail**, AWS CodePipeline history | Ensure end-to-end traceability and auditability. |
| 15 | Rollback & cleanup | Amazon SageMaker Endpoints (weights/variants), blue/green rollback, AWS CodePipeline | Revert quickly and retire unused resources. |

---

## 5) Service glossary (one-liners)

- **AWS Identity and Access Management (IAM):** Fine-grained access control for users, roles, and policies.  
- **AWS Key Management Service (AWS KMS):** Manage encryption keys for data at rest/in transit.  
- **Amazon Virtual Private Cloud (Amazon VPC):** Private networking, subnets, routing, and security groups.  
- **AWS CloudFormation / AWS Cloud Development Kit (AWS CDK):** Define and provision infrastructure as code.  
- **AWS CodeCommit:** Managed Git repositories.  
- **AWS CodePipeline:** Orchestrate CI/CD stages (build, test, approve, deploy).  
- **AWS CodeBuild:** Build/packaging service for code and container images.  
- **Amazon Elastic Container Registry (Amazon ECR):** Private Docker image registry.  
- **Amazon Simple Storage Service (Amazon S3):** Durable object storage for data and model artifacts.  
- **AWS Glue / Amazon Athena:** Data catalog and ETL (Glue), serverless SQL over **Amazon S3** (Athena).  
- **Amazon SageMaker Processing:** Run Python/containers for ETL, evaluation, and report jobs.  
- **Amazon SageMaker Feature Store:** Centralized, online/offline feature storage.  
- **Amazon SageMaker Training:** Managed training jobs for built-in or custom algorithms.  
- **Amazon SageMaker Pipelines:** Native ML pipelines (steps, parameters, conditions, caching).  
- **Amazon SageMaker Model Registry:** Versioned catalog of models with metadata/metrics/approval state.  
- **Amazon SageMaker Endpoints:** Managed HTTPS inference (real-time/serverless/asynchronous).  
- **Amazon SageMaker Batch Transform:** On-demand batch inference that reads/writes to **Amazon S3** and shuts down.  
- **Amazon SageMaker Deployment Guardrails:** Safe rollout strategies (shadow, canary, linear, blue/green) and automated rollbacks.  
- **Amazon SageMaker Model Monitor:** Scheduled data/quality/bias drift checks on production traffic.  
- **Amazon SageMaker Lineage:** Track lineage from data/code to artifacts and endpoints.  
- **Amazon CloudWatch / Amazon CloudWatch Alarms:** Logs, metrics, dashboards, and alarm-based automation.  
- **Amazon EventBridge:** Event bus and scheduling (for example, nightly pipeline triggers).  
- **Amazon Simple Notification Service (Amazon SNS):** Pub/sub notifications (alerts, approvals).  
- **AWS Security Token Service (AWS STS):** Temporary cross-account credentials for promotions.  
- **AWS CloudTrail:** Audit API activity across AWS services.

## 6) Appendix
### A. Path A Python Code
```python
import boto3
import sagemaker
from sagemaker import get_execution_role

# 1. Set up session and role
session = sagemaker.Session()
role = get_execution_role()  # SageMaker execution role with S3/ECR permissions
region = session.boto_region_name

# 2. Location of model artifacts in S3
model_artifact = "s3://my-bucket/path/to/model.tar.gz"

# 3. Choose a SageMaker-provided inference container (here: sklearn)
#    These URIs differ by region; use sagemaker.image_uris to fetch.
from sagemaker import image_uris
container_uri = image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1"  # sklearn version
)

# 4. Create a SageMaker Model object
from sagemaker.model import Model

model = Model(
    image_uri=container_uri,   # inference container image
    model_data=model_artifact, # S3 path to model.tar.gz
    role=role                  # IAM role
)

# 5. Deploy the model to an endpoint
#    Here we choose a real-time endpoint with ml.m5.large instance
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="my-sklearn-endpoint"
)

# ---- At this point you have a live HTTPS endpoint ----

# 6. Make predictions
test_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = predictor.predict(test_data)
print("Prediction:", prediction)

# 7. Clean up (to avoid charges)
predictor.delete_endpoint()
```

### B. Path A Python Code
#### B1. Train.py (toy example using sklearn; generates its own data)
```python
# train.py
# - Runs inside a SageMaker SKLearn training container
# - Trains a simple LogisticRegression and saves to /opt/ml/model/model.joblib

import os, json
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def main():
    # Create synthetic train data (replace with reading from S3 if you have data)
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Persist model to the model artifacts directory expected by SageMaker
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))

    # (Optional) Save simple training metadata
    with open(os.path.join(model_dir, "training_meta.json"), "w") as f:
        json.dump({"n_samples": len(y)}, f)

if __name__ == "__main__":
    main()
```

#### B2. evaluate.py (loads the model artifact from the training step and writes metrics)
```python
# evaluate.py
# - Runs inside a Processing job (SKLearnProcessor)
# - Receives: model.tar.gz from the training step
# - Produces: evaluation.json with metrics (accuracy) in the "evaluation" output

import os, json, tarfile, joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

def load_model_from_tar(tar_path):
    # Unpack model.tar.gz produced by training
    extract_dir = "/opt/ml/processing/model"
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)
    # We saved model as model.joblib in train.py
    return joblib.load(os.path.join(extract_dir, "model.joblib"))

def main():
    # Inputs wired by the ProcessingStep:
    #  - /opt/ml/processing/input/model/model.tar.gz
    model_tar = "/opt/ml/processing/input/model/model.tar.gz"
    clf = load_model_from_tar(model_tar)

    # Make synthetic test data (replace with your real validation set)
    X_test, y_test = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=123)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    # SageMaker Model Metrics expects a JSON; keep it simple
    report = {
        "metrics": {
            "accuracy": {
                "value": acc
            }
        }
    }

    # Write the evaluation report to the "evaluation" output
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f)

if __name__ == "__main__":
    main()
```
#### B3. build_register_pipeline.py
```python
# build_register_pipeline.py
# Creates a SageMaker Pipeline:
#   1) Train (SKLearn Estimator)
#   2) Evaluate (Processing) -> writes evaluation.json
#   3) If accuracy >= threshold, Register to Model Registry as PendingManualApproval
#
# Then starts an execution. Your CI would run this file on each commit or schedule.

import boto3, sagemaker, os
from sagemaker.session import Session
from sagemaker import image_uris
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep

sm_session = sagemaker.Session()
region = sm_session.boto_region_name
role = sagemaker.get_execution_role()
sm_client = boto3.client("sagemaker", region_name=region)

# ======== CONFIG ========
PROJECT_NAME = "demo-sageops-sklearn"
PIPELINE_NAME = f"{PROJECT_NAME}-pipeline"
MODEL_PACKAGE_GROUP = f"{PROJECT_NAME}-pkg-group"
MODEL_APPROVAL_STATUS = "PendingManualApproval"  # gate by default
ACCURACY_THRESHOLD = 0.90                        # simple example gate
instance_type = "ml.m5.large"
# ========================

# Ensure the Model Package Group exists
try:
    sm_client.create_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageGroupDescription="Demo group for sklearn model versions"
    )
    print(f"Created Model Package Group: {MODEL_PACKAGE_GROUP}")
except sm_client.exceptions.ResourceInUse:
    print(f"Model Package Group already exists: {MODEL_PACKAGE_GROUP}")

# Pipeline parameters (overridable at runtime from CI/CD)
p_model_package_group = ParameterString(name="ModelPackageGroupName", default_value=MODEL_PACKAGE_GROUP)
p_acc_threshold       = ParameterFloat(name="AccuracyThreshold", default_value=ACCURACY_THRESHOLD)

# ---- Step 1: Train ----
sk_estimator = SKLearn(
    entry_point="train.py",              # local file in repo
    role=role,
    instance_type=instance_type,
    instance_count=1,
    framework_version="1.2-1",          # sklearn container tag
    sagemaker_session=sm_session,
    base_job_name=f"{PROJECT_NAME}-train"
)
train_step = TrainingStep(
    name="TrainModel",
    estimator=sk_estimator
    # No inputs because train.py generates synthetic data for demo purposes.
)

# ---- Step 2: Evaluate ----
# We run evaluate.py in a Processing job and feed it the model.tar.gz from the training step
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type=instance_type,
    instance_count=1,
    base_job_name=f"{PROJECT_NAME}-eval",
    sagemaker_session=sm_session
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=processor,
    code="evaluate.py",
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/input/model"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    property_files=[evaluation_report],
)

# ---- Step 3: Register (PendingManualApproval) if metrics pass ----
# Build ModelMetrics object pointing to the evaluation.json produced above
metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f"{eval_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri}/evaluation.json",
        content_type="application/json"
    )
)

# The model object to register — uses the same SKLearn container for inference
register_step = RegisterModel(
    name="RegisterToRegistry",
    estimator=sk_estimator,                                  # uses estimator image for inference
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],     # allowlist of deployable types
    transform_instances=["ml.m5.large"],
    model_package_group_name=p_model_package_group,
    model_metrics=metrics,
    approval_status=MODEL_APPROVAL_STATUS
)

# Gate: only register if accuracy >= threshold
cond = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=eval_step.name,
        property_file=evaluation_report,
        json_path="metrics.accuracy.value"
    ),
    right=p_acc_threshold
)
gate_step = ConditionStep(
    name="AccuracyGate",
    conditions=[cond],
    if_steps=[register_step],
    else_steps=[]
)

# ---- Assemble & run the Pipeline ----
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[p_model_package_group, p_acc_threshold],
    steps=[train_step, eval_step, gate_step],
    sagemaker_session=sm_session
)

# Create or update the pipeline in SageMaker, then start an execution
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print(f"Started pipeline execution: {execution.arn}")

```
#### B4. promote_and_deploy.py
```python
# promote_and_deploy.py
# - Option A: Manually approve the most recent Pending model (for demo)
# - Option B: Find latest Approved version and deploy it to an endpoint
# - Option C: (Optional) Canary/linear traffic shifting once two variants exist

import boto3, sagemaker, time
from sagemaker.model import ModelPackage

PROJECT_NAME = "demo-sageops-sklearn"
MODEL_PACKAGE_GROUP = f"{PROJECT_NAME}-pkg-group"
ENDPOINT_NAME = f"{PROJECT_NAME}-endpoint"
INSTANCE_TYPE = "ml.m5.large"

sm = boto3.client("sagemaker")
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# -------- Option A: (Demo) Approve the newest Pending version ----------
pending = sm.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="PendingManualApproval",
    SortBy="CreationTime", SortOrder="Descending", MaxResults=1
)["ModelPackageSummaryList"]

if pending:
    pending_arn = pending[0]["ModelPackageArn"]
    print(f"Approving newest Pending model: {pending_arn}")
    sm.update_model_package(
        ModelPackageArn=pending_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription="Approved via demo script"
    )
else:
    print("No Pending models to approve (skipping Option A).")

# -------- Option B: Deploy the latest Approved model ----------
approved = sm.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime", SortOrder="Descending", MaxResults=1
)["ModelPackageSummaryList"]

if not approved:
    raise RuntimeError("No Approved models exist in the registry. Approve one first.")

approved_arn = approved[0]["ModelPackageArn"]
print(f"Latest Approved model: {approved_arn}")

# Use the high-level ModelPackage class to deploy directly
pkg = ModelPackage(
    role=role,
    model_package_arn=approved_arn,
    sagemaker_session=sess
)

predictor = pkg.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    endpoint_name=ENDPOINT_NAME
)
print(f"Deployed endpoint: {ENDPOINT_NAME}")

# Try a quick test call (your input schema will vary)
print("Sample prediction (dummy):", predictor.predict([[0.1]*20]))

# -----------------------------------------------------------------------
# -------- Option C: Canary / Linear rollout (traffic shifting) ---------
# This section assumes you already have an endpoint with TWO variants:
#   - "Blue"  = old model version
#   - "Green" = new model version (the one we just deployed as another variant)
# If you only have a single-variant endpoint, you'd first create a new
# endpoint config referencing BOTH variants, then call update_endpoint.
# Below we show ONLY the **traffic shifting** API once both variants exist.

def set_weights(endpoint_name, blue_weight, green_weight):
    """Update weights between Blue and Green variants."""
    sm.update_endpoint_weights_and_capacities(
        EndpointName=endpoint_name,
        DesiredWeightsAndCapacities=[
            {"VariantName": "Blue",  "DesiredWeight": float(blue_weight)},
            {"VariantName": "Green", "DesiredWeight": float(green_weight)},
        ],
    )
    print(f"Shifted traffic -> Blue: {blue_weight}  Green: {green_weight}")

# Example CANARY: start 10% to Green, bake, then go 100%
# set_weights(ENDPOINT_NAME, blue_weight=90.0, green_weight=10.0)
# time.sleep(300)  # bake time while you watch metrics/alarms
# set_weights(ENDPOINT_NAME, blue_weight=0.0,  green_weight=100.0)

# Example LINEAR: ramp up by 20% every 5 minutes
# for g in [20, 40, 60, 80, 100]:
#     set_weights(ENDPOINT_NAME, blue_weight=100-g, green_weight=g)
#     time.sleep(300)

# If anything looks bad, you can instantly roll back:
# set_weights(ENDPOINT_NAME, blue_weight=100.0, green_weight=0.0)
```