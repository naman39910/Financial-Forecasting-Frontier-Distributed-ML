# Financial Forecasting Frontier – Distributed Machine Learning

A distributed data engineering and machine learning project focused on banking customer analytics and term deposit subscription prediction using Apache Spark, Hadoop, Hive, and Python.

## Project Overview

The **Financial Forecasting Frontier** project demonstrates how large-scale banking data can be processed, analyzed, and used for predictive modelling through distributed computing technologies.

The project includes:
- Distributed data processing using Apache Spark.
- Banking data analysis and exploratory data analysis (EDA).
- Machine learning using Random Forest.
- Hadoop HDFS and Hive integration.
- MapReduce-style data analysis.
- Spark Structured Streaming for simulated real-time analytics.
- Data parallelism and concurrent analytical processing.

## Objectives

1. Analyze banking customer information.
2. Identify patterns influencing term deposit subscriptions.
3. Build a machine learning model to predict customer subscription outcomes.
4. Demonstrate distributed data processing using Spark and Hadoop.
5. Implement simulated streaming analytics.
6. Explore parallel execution and performance monitoring.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming and data analysis |
| Apache Spark | Distributed data processing |
| PySpark MLlib | Machine learning |
| Hadoop HDFS | Distributed storage |
| Apache Hive | SQL-based data querying |
| MapReduce | Distributed processing concepts |
| PostgreSQL | Hive Metastore database |
| Docker Compose | Container orchestration |
| Jupyter Notebook | Interactive development |
| Pandas | Data manipulation |
| Scikit-learn concepts | Machine learning workflow |

## System Architecture

```text
                 Banking Dataset
                     bank.csv
                        |
          +-------------+-------------+
          |                           |
     Apache Spark                 Hadoop HDFS
          |                           |
    Data Processing               Data Storage
          |                           |
     Exploratory                     Hive
    Data Analysis                 SQL Queries
          |
    Machine Learning
          |
    Random Forest
          |
   Subscription Prediction
          |
   Streaming Analytics
          |
   Data Parallelism
          |
    Analytical Results
```

**Note:** Spark processing scripts read the dataset from the Docker-mounted local `/data/bank.csv` path. HDFS and Hive are demonstrated through separate storage and querying workflows.

## Repository Structure

```text
Financial-Forecasting-Frontier-Distributed-ML/
│
├── README.md
├── docker-compose.yml
├── .gitignore
│
├── docs/
│   └── Financial Forecasting Frontier.docx
│
├── Notebook/
│   └── colab_notebooks/
│
├── data_parallelism/
│   └── data_parallelism.py
│
├── hadoop_hive/
│   ├── hive-site.xml
│   ├── hive_queries.sql
│   └── setup_hdfs.sh
│
├── mapreduce/
│   ├── mr_runner.py
│   └── generate_mr_scripts.py
│
├── spark_ml/
│   └── spark_ml.py
│
├── spark_processing/
│   └── spark_processing.py
│
└── spark_streaming/
    ├── spark_streaming.py
    └── stream_simulator.py
```

## Dataset

The project uses a banking marketing dataset named:

`bank.csv`

The dataset contains customer information and marketing campaign details.

### Important Features

| Feature | Description |
|---|---|
| age | Customer age |
| job | Customer occupation |
| marital | Marital status |
| education | Education level |
| balance | Average yearly balance |
| housing | Housing loan status |
| loan | Personal loan status |
| contact | Contact communication type |
| month | Last contact month |
| duration | Last contact duration |
| campaign | Number of contacts during campaign |
| pdays | Days since previous contact |
| previous | Number of previous contacts |
| poutcome | Previous campaign outcome |
| y | Term deposit subscription (yes/no) |

**Target Variable:** `y`

- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe.

### Dataset Setup

The `data/` directory is excluded from Git using `.gitignore`. Therefore, place your dataset locally:

```text
project-root/
└── data/
    └── bank.csv
```

Use a dataset you are authorized to use. The dataset is not automatically downloaded by Docker Compose.

## Project Modules

### 1. Exploratory Data Analysis – Apache Spark

File: `spark_processing/spark_processing.py`

This module performs distributed data analysis using PySpark.

Key activities:
- Load and inspect banking data.
- Analyze customer demographics.
- Filter customers based on account balance.
- Calculate average balance by occupation.
- Analyze subscription trends by marital status.
- Calculate subscription rates by education.
- Analyze default rates by occupation.
- Group customers by age categories.
- Generate aggregated analytical results.

### 2. Machine Learning – Random Forest

File: `spark_ml/spark_ml.py`

This module builds a classification model to predict whether a customer will subscribe to a term deposit.

**Machine Learning Workflow:**

1. Load banking dataset.
2. Inspect missing values.
3. Handle outliers using the IQR method for balance.
4. Transform categorical variables using StringIndexer.
5. Apply OneHotEncoder.
6. Assemble features using VectorAssembler.
7. Scale features using StandardScaler.
8. Train Random Forest Classifier.
9. Evaluate model performance.
10. Perform hyperparameter tuning using CrossValidator.

**Model Configuration:**
- Algorithm: Random Forest Classifier
- Number of Trees: 100
- Maximum Depth: 5
- Training Data: 80%
- Testing Data: 20%
- Cross Validation: 3 Folds

**Evaluation Metrics:**
- Accuracy
- AUC
- F1 Score
- Confusion Matrix

Actual model performance depends on the dataset and execution environment.

### 3. Hadoop HDFS and Hive

Directory: `hadoop_hive/`

This module demonstrates distributed storage and SQL-based analysis.

Key activities:
- Store banking data in Hadoop HDFS.
- Create HDFS directories.
- Configure Hive Metastore.
- Connect Hive with PostgreSQL.
- Execute analytical SQL queries using Hive.

### 4. MapReduce Processing

Directory: `mapreduce/`

This module demonstrates MapReduce-style analytical processing using Python.

Analysis includes:
- Average balance by occupation.
- Housing loan distribution by education.
- Monthly contacts by subscription status.
- Average call duration by campaign outcome.
- Customer age and balance analysis.

**Note:** The `mr_runner.py` script performs local Python-based aggregations. It is not itself a distributed Hadoop MapReduce execution. The separate script `generate_mr_scripts.py` prepares Hadoop Streaming commands.

### 5. Spark Structured Streaming

Directory: `spark_streaming/`

This module demonstrates simulated streaming analytics using Apache Spark Structured Streaming.

Key activities:
- Read customer records in batches.
- Generate event timestamps.
- Process incoming CSV files.
- Perform streaming aggregations.
- Apply window-based analytics.
- Handle late-arriving data using watermarking.
- Generate model predictions through the streaming workflow.

**Note:** The streaming simulator reads the banking CSV dataset and generates batches. It does not connect to a live banking transaction system.

### 6. Data Parallelism

File: `data_parallelism/data_parallelism.py`

This module explores parallel data processing using Apache Spark.

Key activities:
- Repartition data across Spark partitions.
- Perform parallel aggregations.
- Cache frequently used datasets.
- Execute analytical tasks concurrently.
- Train a Random Forest model.
- Monitor CPU and memory utilization.

## Prerequisites

Install the following tools:

- Docker Desktop
- Docker Compose
- Git
- Python 3 (for local Python scripts)
- Minimum 8 GB RAM recommended
- At least 4 CPU cores recommended

Docker should have sufficient memory and disk resources to run Spark, Hadoop, Hive, PostgreSQL, and Jupyter simultaneously.

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/naman39910/Financial-Forecasting-Frontier-Distributed-ML.git

cd Financial-Forecasting-Frontier-Distributed-ML
```

### Step 2: Add the Dataset

Create a folder named `data` in the project root and place `bank.csv` inside it.

```text
data/bank.csv
```

### Step 3: Start Docker Services

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

### Step 4: Access the Services

| Service | URL | Purpose |
|---|---|---|
| Spark Master | http://localhost:8080 | Spark cluster monitoring |
| HDFS NameNode | http://localhost:9870 | HDFS management |
| Hive UI | http://localhost:10002 | Hive interface |
| Jupyter Notebook | http://localhost:8888 | Interactive notebooks |

The configured Jupyter token is `banking123`.

### Step 5: Configure HDFS

Run the setup script:

```bash
bash hadoop_hive/setup_hdfs.sh
```

This script prepares HDFS directories and uploads the banking dataset.

### Step 6: Execute Hive Queries

Copy the SQL file into the Hive container:

```bash
docker cp hadoop_hive/hive_queries.sql hive:/data/hive_queries.sql
```

Connect to Hive:

```bash
docker exec -it hive beeline -u 'jdbc:hive2://localhost:10000'
```

Execute the SQL file:

```sql
!run /data/hive_queries.sql
```

## Running Spark Modules

### 1. Spark Data Processing

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/app/spark_processing/spark_processing.py
```

### 2. Machine Learning

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/app/spark_ml/spark_ml.py
```

### 3. Data Parallelism

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/app/data_parallelism/data_parallelism.py
```

The scripts may require additional Python packages inside the relevant container, depending on the execution environment.

## Streaming Execution

The streaming simulator and Spark streaming application must both access the same dataset and streaming directory.

The current Docker Compose configuration mounts the project `data/` directory at different paths in different containers. The simulator uses absolute `/data/...` paths, so its container must have the project data directory mounted at `/data` before execution.

Once the mount paths are aligned, start the Spark streaming application:

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
--master spark://spark-master:7077 \
/app/spark_streaming/spark_streaming.py
```

Run the simulator in a container where `/data/bank.csv` is available:

```bash
python3 spark_streaming/stream_simulator.py
```

The simulator generates CSV batches for Spark Structured Streaming to process.

## Running MapReduce Analysis

Run the local Python MapReduce-style analysis:

```bash
python mapreduce/mr_runner.py
```

## Expected Learning Outcomes

Through this project, the following concepts are demonstrated:

- Distributed computing fundamentals.
- Apache Spark architecture.
- DataFrame transformations and aggregations.
- Machine learning pipeline development.
- Classification model evaluation.
- Hadoop distributed storage.
- Hive SQL querying.
- Streaming data processing.
- Data partitioning and parallel execution.
- Containerized data engineering environments.

## Security Considerations

This project is configured as a local learning and demonstration environment.

Before using it in any shared or production environment:

- Replace hardcoded passwords and tokens.
- Do not expose Docker service ports publicly.
- Configure appropriate Hive authentication.
- Review HDFS permissions.
- Use secure secret management.
- Apply suitable network and access controls.

The credentials in the current Docker configuration are demonstration credentials only.

## Future Enhancements

- Integrate Kafka for event-based streaming.
- Deploy the project on a cloud platform.
- Add model comparison with Gradient Boosted Trees and Logistic Regression.
- Implement model tracking using MLflow.
- Build an interactive dashboard using Power BI.
- Add automated model retraining.
- Implement CI/CD for deployment.
- Improve security and authentication.

## Author

**Naman Koolwal**

GitHub: https://github.com/naman39910

## License

This project is intended for educational and learning purposes. Add a LICENSE file if you want to specify reuse and distribution terms.
