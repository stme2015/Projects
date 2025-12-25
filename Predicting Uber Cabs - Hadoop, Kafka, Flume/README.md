# Uber Cab Prediction System

A real-time big data analytics system for predicting Uber cab locations using Apache Hadoop, Kafka, and Flume with K-Means clustering.

## Overview

This project implements a streaming data pipeline that processes historical Uber trip data to predict optimal cab locations. The system uses:
- **Apache Spark MLlib** for K-Means clustering model training
- **Apache Kafka** for real-time data streaming
- **Apache Flume** for data ingestion into HDFS
- **Spark Streaming** for real-time predictions

## Architecture

The system follows a Lambda architecture pattern:
1. **Batch Layer**: Historical data processing and model training
2. **Speed Layer**: Real-time streaming predictions using Kafka and Spark Streaming
3. **Serving Layer**: HDFS storage for both batch and streaming results

See `uber-case-architecture.pdf` for detailed architecture diagram.

## Technical Stack

- **Apache Spark 2.x** - Distributed computing and ML
- **Apache Kafka** - Message streaming
- **Apache Flume** - Data ingestion
- **Hadoop HDFS** - Distributed storage
- **Python 3.6+** - Data processing and ML
- **PySpark MLlib** - Machine learning library

## Project Structure

```
├── 1_Sample_Data_Preparation.ipynb    # Data sampling and preparation
├── 2_Find_K_Value.ipynb               # Elbow method for optimal K
├── 3_Build_Model.ipynb                # K-Means model training
├── 4_Visualization.ipynb              # Results visualization
├── Spark_streaming.py                 # Real-time streaming processor
├── kafkaloader.sh                     # Kafka data loader script
├── flume_properties                   # Flume configuration
└── uber-case-architecture.pdf         # System architecture
```

## Workflow

### 1. Data Preparation
- Sample 2% of historical Uber trip data (~3.3M records)
- Extract latitude and longitude features
- Store in HDFS for model training

### 2. Model Training
- Use Elbow method to determine optimal K value (K=8)
- Train K-Means clustering model on historical data
- Save model to HDFS for real-time predictions
- WSSSE: 68,813.00

### 3. Real-time Streaming
- Ingest streaming data via Flume → HDFS
- Load data into Kafka topics
- Spark Streaming consumes Kafka messages
- Apply trained model for real-time predictions
- Store predictions back to HDFS

## Key Features

- **Scalable**: Handles millions of records using distributed computing
- **Real-time**: Sub-second latency for streaming predictions
- **Fault-tolerant**: Leverages Hadoop ecosystem reliability
- **Accurate**: K-Means clustering with optimized K value

## Model Performance

- **Clusters**: 8 optimal zones identified
- **Features**: Latitude, Longitude
- **Training Data**: 3.3M historical trips
- **WSSSE**: 68,813.00

## Prerequisites

- Hadoop cluster (HDP 2.x or higher)
- Apache Kafka broker
- Apache Flume agent
- Python 3.6+ with PySpark
- Jupyter Notebook (for development)

## Configuration

### Flume Setup
Edit `flume_properties` with your HDFS paths and agent names.

### Kafka Setup
Configure broker list and topic name in `kafkaloader.sh`.

### Spark Streaming
Update HDFS paths in `Spark_streaming.py` for model location and output directory.

## Usage

### Training Phase
```bash
# 1. Run Jupyter notebooks in order
jupyter notebook 1_Sample_Data_Preparation.ipynb
jupyter notebook 2_Find_K_Value.ipynb
jupyter notebook 3_Build_Model.ipynb
```

### Streaming Phase
```bash
# 1. Start Flume agent
flume-ng agent -n CustomerLoanDataAgent_2182B45 -f flume_properties

# 2. Load data to Kafka
./kafkaloader.sh /path/to/data broker-list:9092 topic-name

# 3. Start Spark Streaming
spark-submit Spark_streaming.py zookeeper:2181 topic-name
```

## Results

The system successfully:
- Identifies 8 optimal cab zones in NYC area
- Processes streaming data with <1s latency
- Predicts cab locations with high accuracy
- Stores predictions for further analysis

## Future Enhancements

- Add weather and traffic data for improved predictions
- Implement demand forecasting
- Real-time dashboard for visualization
- Dynamic cluster adjustment based on time/day

## License

Educational/Research Use
