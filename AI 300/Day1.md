# AI-300: Week 1, Lesson 1

Day 1

11–12 November 2026

# Azure Machine Learning Architecture

Today's learning objectives

By the end of this lesson, you should be able to:

1. Explain Azure Machine Learning and its architecture.
2. Understand workspaces and their role.
3. Distinguish compute instances, compute clusters and inference compute.
4. Explain datastores and data assets.
5. Understand environments and their relationship with reproducible training.
6. Identify how these components work together in a machine learning workflow.

   Study allocation

   4 hours

   Learning approach

   Theory + architecture + practical exercises

## PART 1. Understanding Azure Machine Learning

### 1.1 What is Azure Machine Learning?

[Becoming an Azure Data Scientist Associate: How to pass DP-100 exam | trojrobert](https://images.openai.com/static-rsc-4/FSs96-l6DDkhgtG8MmRr5NHrICUzjVepo8xgxm0HTNMzEnEHYJ45Cm9SITJjKHQHGGFmKbZ6-Mq0zb2n1o94DHFnOgm2mNEi1x_UJTIjzFYn1YBukVI8opwlGlk-WALoGZKUxtSvz_02_5pRTfhTgk-r2-QAuDt31RWbIsqvTGw?purpose=inline)

[trojrobert.github.io](https://trojrobert.github.io/becoming-an-azure-data-scientist-associate-how-to-pass-dp-100-exam/?utm_source=chatgpt.com)

Azure Machine Learning (Azure ML) is Microsoft's cloud-based platform for managing the machine learning lifecycle.

It provides infrastructure and services for:

- Preparing and managing datasets.
- Developing machine learning models.
- Training models using scalable computing resources.
- Tracking experiments and evaluating model performance.
- Registering and managing models.
- Deploying models for inference.
- Monitoring models in production.

Think of it as an environment in which an organisation can manage the entire journey of a machine learning model, from raw data to production deployment.

Consider a banking example.

Suppose NCBA wants to develop a credit default prediction model.

The objective is to predict whether a borrower is likely to default on a loan.

A typical workflow would look like this:

## Credit risk machine learning lifecycle

1\. Data collection

Loan applications, repayment histories, customer information

2\. Data preparation

Cleaning, feature engineering and transformation

3\. Model training

Train logistic regression, random forest or gradient boosting models

4\. Model evaluation

AUC, precision, recall, calibration and validation

5\. Model deployment

Expose predictions through an online endpoint or batch scoring

6\. Production monitoring

Monitor data drift, model performance and operational health

Azure ML provides services to support these stages.

The important thing to understand is that Azure ML is not simply a place where you write Python code. It is a platform for managing the complete machine learning lifecycle.

### 1.2 Why do we need a platform like Azure ML?

Imagine building a credit risk model using only your local computer.

You could use Python, pandas and scikit-learn. You could train the model and save it as a `.pkl` file.

But consider what happens when:

- Your dataset grows from 100,000 to 100 million records.
- Your local computer cannot handle the training workload.
- Another data scientist needs to reproduce your experiment.
- You need to deploy the model for thousands of prediction requests.
- You need to know which dataset and parameters produced a particular model.
- Your organisation needs to control who can access the model and its data.

These are problems of scalability, reproducibility, collaboration, deployment and governance.

Azure ML addresses these challenges through a collection of interconnected resources.

Our first major architectural concept is the workspace.

## PART 2. The Azure Machine Learning workspace

### 2.1 What is a workspace?

A workspace is the central organisational and management resource in Azure Machine Learning.

It acts as a container for your machine learning work.

It brings together:

- Your experiments and training jobs.
- Your datasets and data assets.
- Your compute resources.
- Your datastores.
- Your environments.
- Your trained models.
- Your reusable components.

Think of it as a central project environment for a machine learning team.

[How to build an end-to-end Azure Machine Learning workflow | by Eva Jurado Cortés | Data Science + AI at Microsoft | Medium](https://images.openai.com/static-rsc-4/UjTbPMspUTON3wai9FL8zlTQK1us6oY2vMYU7KiNWfm8vmfNd0j1clcIhJcrBmW5Q5YhPA1KyllUChxqecyvKofQ_WfJQRcT9gctR-fGM8Y4N-BTjBtMzqn139KlkMmxDZcAUJ8qaJqxIgde6Ww0d0eXqopIjc_pQtR455f00Gk?purpose=inline)

[medium.com](https://medium.com/data-science-at-microsoft/how-to-build-an-end-to-end-azure-machine-learning-workflow-bbce316907d1?utm_source=chatgpt.com)

For example, imagine NCBA has a machine learning team developing three models:

1. Credit default prediction.
2. Fraud detection.
3. Customer churn prediction.

The team could organise these projects within an Azure ML workspace, depending on its access, security, cost and isolation requirements.

Each workspace has its own resources, configurations and assets.

### 2.2 What does a workspace actually manage?

There are three important infrastructure resources you need to understand.

| Resource  | Purpose                                           |
| --------- | ------------------------------------------------- |
| Workspace | Central management of ML activities and assets    |
| Compute   | Provides processing power to execute ML workloads |
| Datastore | Maintains connection information for data storage |

There are also four important asset types:

| Asset       | Purpose                                                   |
| ----------- | --------------------------------------------------------- |
| Data        | Identifies and versions data used in ML jobs              |
| Environment | Defines software dependencies and execution configuration |
| Model       | Stores a trained model and its associated metadata        |
| Component   | Packages reusable steps in an ML workflow                 |

Important distinction: Resources provide the infrastructure for your machine learning activities. Assets are the reusable, trackable objects you create and manage within that infrastructure.

This distinction is directly relevant to your examination.

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn+1

### 2.3 Workspace architecture: a practical example

Suppose you are developing a credit default model.

Your workspace might look like this:

Azure Machine Learning Workspace

credit-risk-workspace

Compute

CPU/GPU resources for training and development

Datastores

Connections to Azure Storage

Data assets

Versioned credit-risk datasets

Environments

Python and ML dependencies

Models

Trained and versioned models

Components

Reusable processing and training steps

One thing to remember: the workspace does not necessarily physically contain all your underlying data. For example, your actual credit records may reside in Azure Data Lake Storage. The workspace can maintain references to the relevant storage and registered assets.

### 2.4 Workspace and its associated Azure resources

When creating an Azure ML workspace, you may also need associated Azure resources.

Common associated resources include:

- Azure Storage Account: Stores workspace artefacts, job outputs and related files.
- Azure Key Vault: Stores secrets and credentials.
- Azure Container Registry (ACR): Stores container images used by ML workloads.
- Application Insights: Supports monitoring and diagnostics.

Depending on the configuration, Azure ML can create these resources automatically or use existing supported resources.

Don't confuse these associated resources with the workspace's own logical assets.

For example:

- A datastore is an Azure ML resource that stores connection information.
- The storage account is the actual Azure storage infrastructure.
- A data asset is a versioned reference to data used by your ML workflows.

This distinction is particularly useful when exam questions describe a storage problem and ask which Azure ML object should be configured.

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn+1

## PART 3. Compute resources

This is one of the most important topics in today's lesson.

### 3.1 What is compute?

In machine learning, compute refers to the processing resources used to execute workloads.

These resources include:

- CPU.
- RAM.
- GPU.
- Storage and associated processing infrastructure.

When you train a model, your computer needs to process data, calculate parameters, optimise a loss function and produce model artefacts.

Azure ML allows you to execute these workloads using cloud-based compute resources rather than relying entirely on your local machine.

### 3.2 The five compute types you should know

Azure ML provides several compute options. For your examination, pay particular attention to these five:

[Learn To Create The Compute Resources In The Azure Machine Learning Workspace](https://images.openai.com/static-rsc-4/aqXEh4tkcebMDFMpEjqAG6amxVaj8HDw565J7wDHvXibuYTwL2HMgRcxhy_PBg163vrSi09Jbfr74Q6Tfik3uDSMMVa34BUASDe6SM0mMppViBdpgDWQ1N-LkaZhyo-z77HZrlTDvOTafGWtQcyMp3fildBrOI_i7JCpAJSqWwU?purpose=inline)

1\. Compute instance

Individual development

A managed cloud workstation for an individual data scientist.

Used for notebooks, experimentation, development and smaller training or testing workloads.

[Compute Target | Azure Machine Learning](https://images.openai.com/static-rsc-4/wKpSEyE2TLxI7dx4Itjad86k4A6RfghVRjFH7k8FLSbE-VsNbV7UgTU2a0h9pY0mMu2-QUNc10SU2Y_jnNA1qzTThCNeQ7aKXPrTjeOla_Emrxivcx7tX1NabTmOwgkpPd_KHx9Zyw_DqA2O8D1jJAvdEQvcPRT3nDbhpNdocNQ?purpose=inline)

2\. Compute cluster

Scalable training

A managed collection of CPU or GPU nodes that can scale according to workload requirements.

Used for larger training jobs, parallel workloads and hyperparameter tuning.

[Einrichten von Automated ML für Tabellendaten im Studio - Azure Machine Learning | Microsoft Learn](https://images.openai.com/static-rsc-4/7wUWoCSCEfpkpI_uRMBPEedBJTYRiLbQZuZSA1SCixwUgXdzCwu0ZsvbAUvgLa45X0euV6-E93nqaygOjlgXDq-umG3CLTcYGrP1ePQ2vTegqLuwwA9Q5FdltkjoJxJYrsuJHQPbNSWUogvmISvoZoFyUQ2UFTxErUKBkMF483E?purpose=inline)

3\. Serverless compute

On-demand execution

Allows you to run supported ML jobs without provisioning and managing your own compute cluster.

Azure ML manages the compute lifecycle.

[AKS — Azure Kubernetes Service Overview | by Anoop L | Medium](https://images.openai.com/static-rsc-4/2DLFVBlT1W6gPf8ePUI_9Ei5QNySD7U6lYssKmF-6latJ5Zp_AfskgAYZK3mz0lftHQEVM5qmnqaK-jOnVc3hAdYueM1SVqbWtH-Yc5h-x0IA33hfFZo9GPqCWvpG33q8iFvAZ1RnUOpFojbp1rlCl6TZHpe3vfjE8ydbRcMdCI?purpose=inline)

4\. Inference cluster

Model serving

An attached Azure Kubernetes Service (AKS) cluster used to host deployed models through Kubernetes-based inference.

Relevant when using AKS-based inference architectures.

[Planejar o isolamento de rede - Azure Machine Learning | Microsoft Learn](https://images.openai.com/static-rsc-4/oYR2GoFGYWG_KN6CsfFnQm_PT-h40yV4UMWqNZ27PhoFVWySD9VlzmXXU5mFooKiQBo2KD1bOFNANLkdwUO37qAytSCZBV7sEEqfvHzYWFwWmOe8I2FyGpuJsO_SMx5_GUGzAOz6i_b4lh6PCi8CHshMUuZsxjAhlSA9gF7nbWw?purpose=inline)

5\. Attached compute

Existing infrastructure

Connects external compute resources to your Azure ML workspace.

Useful when an organisation already has infrastructure it wants to use.

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn+1

### 3.3 Compute instance vs compute cluster

This is a distinction you must understand rather than memorise.

Imagine you are working on the credit default prediction project.

You want to:

- Explore your dataset.
- Write Python code.
- Plot distributions.
- Test feature engineering techniques.
- Run a small experiment.

A compute instance is appropriate. It provides an interactive development environment.

Now imagine you have developed a training script and need to train 20 different model configurations using different hyperparameters.

A compute cluster is more appropriate because it can provide multiple compute nodes and scale to handle the jobs.

The difference:

| Feature         | Compute instance             | Compute cluster          |
| --------------- | ---------------------------- | ------------------------ |
| Primary purpose | Development                  | Workload execution       |
| Typical user    | Individual data scientist    | ML team/jobs             |
| Execution       | Interactive                  | Job-based                |
| Scaling         | Individual machine           | Multiple nodes           |
| Best suited for | Notebooks and exploration    | Training and tuning      |
| Cost behaviour  | Can incur cost while running | Can scale down when idle |

### 3.4 What is autoscaling?

Suppose your training workload requires 10 compute nodes.

You configure your compute cluster with:

- Minimum nodes: 0
- Maximum nodes: 10

When you submit a training job, Azure ML can provision nodes as needed, subject to your configuration and available quota.

When the workload finishes, the cluster can scale down to zero.

Why is this useful?

You don't have to pay for idle cluster compute capacity when there are no jobs running, although other associated charges may still apply.

Exam scenario:

An organisation wants to run occasional training jobs that require substantial compute power. It wants to minimise costs when no training jobs are running.

What should it use?

Answer: A compute cluster configured with autoscaling and a minimum node count of zero.

### 3.5 Choosing the right compute

| Scenario                                           | Appropriate choice    |
| -------------------------------------------------- | --------------------- |
| Develop a model using Jupyter notebooks            | Compute instance      |
| Train a large model across multiple nodes          | Compute cluster       |
| Execute a supported job without managing a cluster | Serverless compute    |
| Host models on Kubernetes                          | AKS inference cluster |
| Reuse existing external compute infrastructure     | Attached compute      |

A useful exam habit: identify whether the question is asking about interactive development, scalable job execution, serverless execution or production inference. That usually narrows down the answer significantly.

## PART 4. Datastores

### 4.1 What is a datastore?

A datastore is an Azure ML resource that securely stores the connection information needed to access data in an underlying storage service.

This is an important definition.

A datastore is not the actual dataset. It is not the storage account itself. It is the connection configuration that allows Azure ML to access the storage.

Think about this banking scenario.

NCBA has historical loan repayment data stored in Azure Data Lake Storage Gen2.

Your machine learning model needs access to that data.

Rather than repeatedly writing storage connection details into your Python scripts, you register a datastore in Azure ML.

The datastore provides the connection to the underlying storage.

## How a datastore works

Azure ML Workspace

Datastore

Connection information and authentication configuration

Azure Data Lake Storage Gen2

Actual files and datasets

### 4.2 Supported storage services

For the Azure ML SDK v2 and CLI v2 concepts in our syllabus, know these:

| Storage service              | Typical use                                       |
| ---------------------------- | ------------------------------------------------- |
| Azure Blob Storage           | Object and unstructured data storage              |
| Azure Files                  | Managed file shares                               |
| Azure Data Lake Storage      | Large-scale analytics storage                     |
| Azure Data Lake Storage Gen2 | Analytics workloads using hierarchical namespaces |

The key idea is that Azure ML datastores can connect to different supported storage backends.

### 4.3 Why use datastores?

Consider what happens when an organisation has multiple data scientists.

Without properly configured datastores, developers may independently manage storage connection information.

This creates problems:

- Repeated configuration.
- Inconsistent access.
- Credential management risks.
- Difficult maintenance.
- Poor governance.

Datastores centralise storage connection configuration within the Azure ML workspace.

### 4.4 Authentication and security

A datastore needs a way to authenticate when accessing storage.

Depending on the configuration, authentication can use:

- Microsoft Entra ID credentials.
- Managed identities.
- Account keys or other supported credentials.

For production environments, managed identities are particularly important.

What is a managed identity?

It is an identity managed by Azure that allows an Azure resource to authenticate to other services without requiring you to manually manage credentials in application code.

For example:

Your Azure ML compute resource needs to access a storage account.

Instead of hardcoding a storage account key into your training script, you can configure the appropriate managed identity and grant it the necessary permissions.

This reduces the need to handle long-lived secrets manually.

We will study identity and access control in greater detail tomorrow.

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn+1

## PART 5. Data assets

This is where datastore and data asset concepts become particularly important.

### 5.1 What is a data asset?

A data asset is a registered, reusable and versioned reference to data used in Azure Machine Learning.

Suppose you have a CSV file containing historical loan applications.

You want to use this dataset for multiple experiments.

Rather than treating it as an anonymous file every time, you register it as a data asset.

For example:

Data asset: credit-risk-data

Asset type

URI file

Version

1

Format

CSV

Location

Azure Data Lake Storage

Illustrative example, not an actual registered resource.

Later, the data is updated with additional repayment records.

Instead of overwriting the original registered version, you can register a new version.

credit-risk-data

1. Version 1

   Original dataset: January–June 2026
2. Version 2

   Updated dataset: January–September 2026
3. Version 3

   Updated dataset: January–October 2026

This gives you traceability.

For instance, if a model was trained using version 1, you can identify the dataset version used in that experiment.

### 5.2 Three data asset types

Azure ML supports three important data asset types.

| Type         | Meaning                                                           | Example                                                      |
| ------------ | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| `uri_file`   | Reference to a single file                                        | `loans.csv`                                                  |
| `uri_folder` | Reference to a folder containing files                            | `/credit-data/`                                              |
| `mltable`    | A structured definition of how to load and interpret tabular data | Multiple CSV files with a defined schema and transformations |

Let's understand these individually.

A. URI file

Use this when your data is represented by a single file.

Example:

`credit_applications.csv`

B. URI folder

Use this when your data consists of multiple files within a directory.

Example:

`/credit-data/2026/`

This could contain several monthly CSV files.

C. MLTable

This is particularly interesting.

MLTable allows you to define how data should be loaded and interpreted.

For example, suppose your data is distributed across 12 monthly CSV files.

An MLTable definition can specify:

- Which files to read.
- How to parse them.
- Which columns to use.
- How the data should be transformed or interpreted.

This is useful when building reusable and repeatable data preparation workflows.

### 5.3 The critical difference: datastore vs data asset

Let's make sure this is completely clear.

Imagine the following:

You have a storage account called `ncbamlstorage`.

Inside it, there is a container containing:

`credit_applications.csv`

You register a datastore called `credit-storage` that provides access to that container.

Then you register a data asset called `credit-risk-data`, pointing to the CSV file.

Datastore: credit-storage

How Azure ML connects to storage

Data asset: credit-risk-data

Which data to use, and which version

Training job

Consumes the registered dataset

Remember:

- Datastore = Where and how to access the storage.
- Data asset = What data your ML workflow references.

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn+1

## PART 6. Environments

### 6.1 Why are environments necessary?

This concept should be familiar if you have worked with Python, virtual environments, Docker or dependency management.

Imagine you develop a credit default prediction model using:

- Python 3.11
- pandas 2.x
- scikit-learn 1.5
- NumPy
- MLflow

Your model works perfectly.

You share your training script with another data scientist.

They execute it using Python 3.13 and different package versions.

Suddenly, the code produces errors or different results.

Why?

Because your execution environments are different.

An Azure ML environment defines the software dependencies and configuration required to execute an ML workload.

It can specify:

- Python version.
- Installed libraries.
- Dependencies.
- Docker image.
- Environment variables.
- Other software configuration.

### 6.2 Types of Azure ML environments

There are two major categories you should understand.

1\. Curated environments

These are predefined environments supplied by Azure ML.

They contain commonly used frameworks, packages and configurations.

Examples include environments prepared for:

- PyTorch.
- TensorFlow.
- Scikit-learn.

The advantage is convenience. You don't need to build everything from scratch.

2\. Custom environments

These are environments that you configure yourself.

You can define your own dependencies using:

- A Docker image.
- A base Docker image and a Conda YAML file.
- Other supported environment configurations.

For example, you might need an environment with a specific combination of:

- Python 3.11.
- pandas.
- scikit-learn.
- XGBoost.
- MLflow.

You create a custom environment that captures those dependencies.

### 6.3 Why environment versioning matters

Suppose your original model was trained using:

`credit-risk-env:1`

Six months later, you update the environment:

`credit-risk-env:2`

The new version contains updated libraries.

If you need to reproduce your original training run, the first environment version provides a reference to the software configuration used at the time.

This is important for:

- Reproducibility.
- Auditability.
- Collaboration.
- Troubleshooting.
- Production consistency.

### 6.4 Environment vs datastore vs compute

These three concepts are frequently confused.

| Concept     | Question it answers                              |
| ----------- | ------------------------------------------------ |
| Compute     | Where will the code execute?                     |
| Environment | What software will the code execute with?        |
| Datastore   | How will the code access the underlying storage? |

Consider this example:

You want to train a random forest model.

- Compute cluster: Provides the processing power.
- Environment: Provides Python and scikit-learn.
- Datastore: Provides access to the training data.
- Data asset: Identifies the specific training dataset.

All four work together to execute your training job.

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn

## PART 7. Putting everything together

Now let's combine all five concepts into one architecture.

Imagine you have been assigned to develop a credit risk model.

The following is the relationship between the Azure ML resources and assets.

[Azure Machine Learning – Time With AI](https://images.openai.com/static-rsc-4/xv0IVgvZ-AYFA33YhrjCUdcyaDuEU6SEdsKVjmeRdwDTUzkdm2cBSF7woPI1wIBuS2XNWUStx5lYIjxhpMxW6p-pLFfeV2P1EyDrjdWsViKZI-8WF_QQV1Ic8aMgUnly3dJRxvEhY4hgIO3FN3Lp-tDZAaFv4kH9lHil8eaApEc?purpose=inline)

[timewithai.wordpress.com](https://timewithai.wordpress.com/2020/03/26/azure-machine-learning/?utm_source=chatgpt.com)

Here is the logical sequence:

1. Create an Azure ML workspace to manage the project.
2. Register a datastore to connect to Azure Data Lake Storage.
3. Register a data asset pointing to the credit-risk dataset.
4. Create an environment containing the required Python packages.
5. Provision a compute cluster to execute training.
6. Submit a training job using the data asset, environment and compute.
7. Track the job and eventually register the resulting model.

Notice how each component has a distinct responsibility.

This is the architectural foundation for the rest of Week 1.

You will later learn how MLflow tracks the training run, how hyperparameter optimisation improves the model, how registries manage assets and how endpoints deploy models.

## PART 8. Exam-focused knowledge consolidation

Before testing you, let's identify some common examination traps.

### 8.1 Concepts Microsoft expects you to distinguish

| If the question describes...                         | Think of...        |
| ---------------------------------------------------- | ------------------ |
| Centralised management of ML resources and assets    | Workspace          |
| Connection information for cloud storage             | Datastore          |
| A specific, reusable and versioned dataset reference | Data asset         |
| A managed interactive development machine            | Compute instance   |
| Scalable CPU/GPU training                            | Compute cluster    |
| Execution without managing your own cluster          | Serverless compute |
| Python libraries and Docker configuration            | Environment        |
| Reusable workflow steps                              | Component          |
| A trained model and its files                        | Model              |

### 8.2 Common mistakes

1. Assuming a datastore contains the actual data. It stores connection information.
2. Assuming a data asset is the physical storage account. It is a registered reference to data.
3. Using a compute instance when the scenario requires scalable distributed training.
4. Assuming an environment provides processing power. Compute provides processing power.
5. Assuming a workspace is just a notebook environment. It manages much more than notebooks.
6. Confusing a custom environment with a custom compute resource. They solve completely different problems.

## PART 9. Knowledge assessment

Now we test whether you actually understand the architecture.

I want you to answer these questions without consulting the notes above. Don't worry about getting everything right. Your mistakes will determine what we revisit.



Assessment 1

&#x20;Foundational understanding



8 examination-style questions

0/8 answered

Question 1. A data scientist needs an interactive cloud-based environment to develop Python notebooks and explore a dataset. Which compute resource should they use?

A. Compute cluster

B. Compute instance

C. Inference cluster

D. Datastore

Question 2. An organisation stores its loan records in Azure Data Lake Storage Gen2. It wants Azure ML to securely connect to this storage without hardcoding credentials in its scripts. What should it configure?

A. Data asset

B. Environment

C. Datastore

D. Model registry

Question 3. A training dataset is updated every month. The team needs to retain references to previous dataset versions to reproduce historical experiments. What should it use?

A. Compute instances

B. Versioned data assets

C. Application Insights

D. Azure Container Registry

Question 4. A data scientist needs to train 30 different model configurations and wants compute resources to scale from zero when idle. What is the most appropriate choice?

A. Compute instance

B. Compute cluster with autoscaling

C. Azure Files

D. Custom environment

Question 5. A training script requires Python 3.11, pandas, scikit-learn and XGBoost. Which Azure ML asset defines these dependencies?

A. Datastore

B. Data asset

C. Environment

D. Workspace

Question 6. Which statement best describes an Azure ML workspace?

A. It is exclusively a cloud storage account.

B. It is a central resource for managing ML resources, jobs and assets.

C. It is a replacement for Python.

D. It is exclusively used to deploy models.

Question 7. Which of the following is an Azure ML asset rather than an infrastructure resource?

A. Compute

B. Datastore

C. Environment

D. Workspace

Question 8. Your team needs to use an existing external compute resource rather than provision a new Azure ML compute cluster. What should it consider?

A. Attached compute

B. Compute instance

C. Data asset

D. Curated environment

Submit answers

## PART 10. Applied architecture exercise

This is more important than memorising definitions.

You are the ML engineer responsible for implementing NCBA's credit default prediction solution.

Your organisation has:

- Historical loan repayment records in Azure Data Lake Storage Gen2.
- A data science team using Python and scikit-learn.
- A requirement to retain historical dataset versions.
- Training jobs that occasionally require substantial CPU resources.
- A need to reproduce previous experiments.

Your task: Design the Azure ML architecture.

Explain which resources and assets you would use for each of these:

1. Central management of the project.
2. Secure connection to historical loan records.
3. Versioning of the training dataset.
4. Defining Python and scikit-learn dependencies.
5. Executing scalable training workloads.
6. Reproducing a previous training experiment.

I want you to explain why you selected each component, not simply name it.

## PART 11. Your reference materials

These are the official Microsoft resources that underpin today's lesson. Use them as supplementary references, not as a replacement for our teaching.

1. [AI-300 official examination study guide](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-300?utm_source=chatgpt.com). Pay attention to the skills measured and their weighting.
2. [How Azure Machine Learning works: Resources and assets](https://learn.microsoft.com/azure/machine-learning/concept-azure-machine-learning-architecture?utm_source=chatgpt.com). This is our main technical reference.
3. [Azure Machine Learning workspaces](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2\&utm_source=chatgpt.com). Useful for workspace architecture and associated resources.
4. [Azure ML compute instances](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2\&utm_source=chatgpt.com). Useful for understanding development compute.

The official examination guide assigns 15–20% to designing and implementing MLOps infrastructure. Today's lesson establishes the foundation for that domain, but we will also revisit these concepts in later lessons.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://learn.microsoft.com\&sz=32)

Microsoft Learn+1



## Today's completion criteria

Day 1 learning checklist

Your progress

0/9

Explain what Azure Machine Learning does.

Distinguish workspace resources from assets.

Explain the role of a workspace.

Distinguish compute instances from compute clusters.

Explain datastores and their authentication.

Distinguish datastores from data assets.

Explain uri_file, uri_folder and MLTable.

Distinguish curated and custom environments.

Design the credit risk ML architecture.

## How we will study going forward

I suggest we follow this structure for every four-hour session:

| Activity                            | Time    |
| ----------------------------------- | ------- |
| Detailed teaching and explanations  | 90 min  |
| Architecture and practical examples | 45 min  |
| Guided implementation or exercises  | 60 min  |
| Examination questions and review    | 45 min  |
| Total                               | 4 hours |

One scheduling adjustment: your plan labels today, 12 November, as the second day, but the first topic was scheduled for 11 November. We'll complete the first topic now and then continue with workspace configuration. We won't skip material just to catch up with the calendar.

My expectation: Don't rush through these lessons. I want you to be able to look at an unfamiliar Azure architecture scenario and reason your way to the correct solution, even if Microsoft changes the wording of the examination question.

Start by answering the eight assessment questions and then attempt the architecture exercise. I'll mark your answers, identify misconceptions, and teach you anything you haven't fully grasped before we move on.

We have an examination to pass, but more importantly, we're building an engineer who understands what they're deploying.