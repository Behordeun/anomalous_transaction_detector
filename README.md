
# Detect Anomalous Transactions in Unstructured Financial Records

## Implementation Report & Deliverables

## Implementation Report

This project delivers a robust, production-ready pipeline for detecting anomalous transactions in unstructured financial logs. The solution is designed to address real-world challenges in fraud detection, including noisy data, evolving fraud patterns, and the need for interpretable, actionable results. Key components and highlights include:

- Modular parsing logic handles multiple log formats (CSV, XLSX) using regular expressions and custom heuristics.
- Graceful handling of malformed, incomplete, or edge-case records maximizes data utility.
- Diagnostic logging and reporting provide transparency on parsing success/failure rates.
- Parsing logic is extensible and documented for future log formats.

### 1. Data Parsing & Cleaning

- Modular parsing logic handles multiple log formats (CSV, XLSX) using regular expressions and custom heuristics.
- Graceful handling of malformed, incomplete, or edge-case records maximizes data utility.
- Diagnostic logging and reporting provide transparency on parsing success/failure rates.
- Parsing logic is extensible and documented for future log formats.
- Modular parsing logic handles multiple log formats (CSV, XLSX) using regular expressions and custom heuristics.
- Graceful handling of malformed, incomplete, or edge-case records maximizes data utility.
- Diagnostic logging and reporting provide transparency on parsing success/failure rates.
- Parsing logic is extensible and documented for future log formats.

- Extraction and transformation of numeric and categorical features: transaction amount, device, location, time, user behavior.
- Computation of user-level statistics (median, std, z-score) and behavioral novelty (new device/location, time gaps).
- Integration of geospatial calculations and currency normalization for richer context.
- Mode-based filling for missing categorical values, with diagnostics on NaN counts and missing columns.

### 2. Feature Engineering

- Extraction and transformation of numeric and categorical features: transaction amount, device, location, time, user behavior.
- Computation of user-level statistics (median, std, z-score) and behavioral novelty (new device/location, time gaps).
- Integration of geospatial calculations and currency normalization for richer context.
- Mode-based filling for missing categorical values, with diagnostics on NaN counts and missing columns.
- Extraction and transformation of numeric and categorical features: transaction amount, device, location, time, user behavior.
- Computation of user-level statistics (median, std, z-score) and behavioral novelty (new device/location, time gaps).
- Integration of geospatial calculations and currency normalization for richer context.
- Mode-based filling for missing categorical values, with diagnostics on NaN counts and missing columns.

- Unsupervised models (Isolation Forest, Local Outlier Factor) applied to engineered features.
- Aggregate model scores for robust anomaly labeling; contamination rate is tunable via UI.
- Top-N anomalies are flagged with clear, human-readable explanations based on feature heuristics.
- Model performance and parameters are monitored and tunable.

### 3. Anomaly Detection & Modeling

- Unsupervised models (Isolation Forest, Local Outlier Factor) applied to engineered features.
- Aggregate model scores for robust anomaly labeling; contamination rate is tunable via UI.
- Top-N anomalies are flagged with clear, human-readable explanations based on feature heuristics.
- Model performance and parameters are monitored and tunable.
- Unsupervised models (Isolation Forest, Local Outlier Factor) applied to engineered features.
- Aggregate model scores for robust anomaly labeling; contamination rate is tunable via UI.
- Top-N anomalies are flagged with clear, human-readable explanations based on feature heuristics.
- Model performance and parameters are monitored and tunable.

- Interactive Streamlit app for end-to-end analysis, including file upload, process controls, and dynamic reporting.
- Visualizations include anomaly frequency over time, device/location/user-level anomaly frequency, transaction amount distributions, and top anomalies table.
- Each flagged anomaly is explained using feature-based heuristics (e.g., unusually large amount, new device/location, unusual time gap).
- Diagnostic exports and summary statistics support stakeholder communication and manual validation.

### 4. Visualization & Interpretability

- Interactive Streamlit app for end-to-end analysis, including file upload, process controls, and dynamic reporting.
- Visualizations include anomaly frequency over time, device/location/user-level anomaly frequency, transaction amount distributions, and top anomalies table.
- Each flagged anomaly is explained using feature-based heuristics (e.g., unusually large amount, new device/location, unusual time gap).
- Diagnostic exports and summary statistics support stakeholder communication and manual validation.
- Interactive Streamlit app for end-to-end analysis, including file upload, process controls, and dynamic reporting.
- Visualizations include anomaly frequency over time, device/location/user-level anomaly frequency, transaction amount distributions, and top anomalies table.
- Each flagged anomaly is explained using feature-based heuristics (e.g., unusually large amount, new device/location, unusual time gap).
- Diagnostic exports and summary statistics support stakeholder communication and manual validation.

- Enables proactive identification and investigation of anomalous transactions in large, unstructured datasets.
- Enhances fraud detection, operational efficiency, regulatory compliance, customer trust, and strategic insights.
- Recommendations for deployment include integration with real-time monitoring, feedback loops, supervised model extension, and regular retraining.

### 5. Business Impact

- Enables proactive identification and investigation of anomalous transactions in large, unstructured datasets.
- Enhances fraud detection, operational efficiency, regulatory compliance, customer trust, and strategic insights.
- Recommendations for deployment include integration with real-time monitoring, feedback loops, supervised model extension, and regular retraining.
- Enables proactive identification and investigation of anomalous transactions in large, unstructured datasets.
- Enhances fraud detection, operational efficiency, regulatory compliance, customer trust, and strategic insights.
- Recommendations for deployment include integration with real-time monitoring, feedback loops, supervised model extension, and regular retraining.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Objectives](#objectives)
- [Pipeline Steps](#pipeline-steps)
- [Deliverables](#deliverables)
- [Usage](#usage)
- [Installation &amp; Environment](#installation--environment)



## Evaluation Rubric

- [Business Impact](#business-impact)

## Overview

This project provides a comprehensive, production-ready pipeline for detecting anomalous transactions in semi-structured financial logs. It is designed to address real-world challenges in fraud detection, including noisy data, evolving fraud patterns, and the need for interpretable results. The solution leverages Python, pandas, scikit-learn, and Plotly for scalable data processing, advanced analytics, and interactive visualizations. The modular architecture allows for easy extension to new data sources, features, and models.

## Dataset Description

The dataset comprises raw transaction logs with minimal structure, simulating real-world financial system or user-facing event logs. Each row contains a log entry with information such as timestamp, user ID, transaction type, amount, device, and location. The logs are intentionally varied in format to test the robustness of the parsing logic. Example formats include:

```text
"2023-05-14 14:05:31 | user: 1023 | txn: withdrawal of £500 from ATM near Liverpool | device: Samsung Galaxy S10 | location: 53.4084,-2.9916"
```

The solution is designed to handle malformed records, missing fields, and multiple log patterns, ensuring generalization to new data sources.

## Objectives

1. **Data Understanding & Parsing**

   - Develop robust parsing logic to convert diverse, noisy log formats into a structured DataFrame.
   - Gracefully handle malformed, incomplete, or edge-case records to maximize data utility.
   - Document parsing rules and provide extensibility for new log formats.
2. **Feature Engineering**

   - Extract and transform numeric and categorical features, including transaction amount, device, location, time, and user behavior.
   - Compute user-level statistics (median, std, z-score) and behavioral novelty (new device/location, time gaps).
   - Integrate geospatial calculations and currency normalization for richer context.
3. **Anomaly Detection**

   - Apply unsupervised models (Isolation Forest, Local Outlier Factor) to engineered features.
   - Aggregate model scores for robust anomaly labeling.
   - Tune model parameters and contamination rates for optimal precision/recall.
4. **Evaluation & Interpretability**

   - Visualize clusters, outliers, and anomaly distributions using interactive charts.
   - Flag top-N anomalies with clear, human-readable explanations based on feature heuristics.
   - Provide a manual validation plan and recommendations for operational integration.

## Pipeline Steps

- **Parsing & Cleaning:**

  - Multiple log formats are parsed using regular expressions and custom heuristics to extract timestamp, user, transaction type, amount, location, and device.
  - Malformed or incomplete records are filtered or imputed to maintain data integrity.
  - Parsing logic is modular and documented for easy extension.
- **Feature Engineering:**

  - Features include currency conversion, time-based fields (hour, weekday, month), device/location novelty, user statistics (median, std, z-score), and geospatial calculations (distance, speed).
  - Behavioral features capture user transaction patterns and anomalies in device/location usage.
  - All features are validated for completeness and relevance.
- **Modeling:**

  - Categorical features are one-hot encoded; numeric features are standardized.
  - Isolation Forest and Local Outlier Factor are trained on the feature matrix to detect anomalies.
  - Aggregate scores are computed for robust anomaly detection, with thresholds set by contamination rate.
  - Model performance is monitored and parameters are tunable.
- **Visualization:**

  - Interactive charts (Plotly) display:
    - Distribution of transaction amounts by anomaly label
    - Amount vs. anomaly score
    - Anomaly counts by transaction type
  - Visualizations support exploratory analysis and stakeholder communication.
- **Interpretability:**

  - Each flagged anomaly is explained using feature-based heuristics (e.g., unusually large amount, new device/location, unusual time gap).
  - Explanations are provided in both tabular and visual formats for transparency.

## Deliverables


### Project Deliverables

1. **Jupyter Notebook**
   - Step-by-step, documented code for the entire pipeline
   - Interactive visualizations for exploratory analysis and stakeholder review

2. **Standalone Parser Module**
   - Modular Python code for extracting structured features from raw, unstructured logs
   - Extensible logic for new log formats and robust error handling

3. **Output Files**
   - Parsed logs, engineered features, top anomalies, and diagnostic reports in CSV format
   - Visual charts (PNG, HTML) for business and technical reporting

4. **Streamlit App**
   - Interactive UI for file upload, process controls, dynamic reporting, and visualization
   - Downloadable results and diagnostic summaries

5. **Example Scripts**
   - Batch processing and integration scripts for command-line and automated workflows

6. **Documentation**
   - This README.md with methodology, usage instructions, business context, and technical details
   - Inline docstrings and comments throughout the codebase

7. **Evaluation Rubric & Manual Validation Plan**
   - Criteria for assessing pipeline performance, interpretability, and business value
   - Recommendations for manual review and operational integration

---

## Usage

### Jupyter Notebook

Open and run the notebook cell-by-cell to follow the pipeline, inspect intermediate results, and interact with visualizations.

### Command Line

For batch processing or integration, run:

```bash
python analysis.py --input data/synthetic_dirty_transaction_logs.csv --output_dir output
```

### Output

Results (parsed logs, features, anomaly scores, top anomalies, charts) are saved in the `output/` directory for further analysis or reporting.

## Installation & Environment

1. Clone the repository and navigate to the project folder:




   ```bash
   git clone <repo-url>
   cd <project-folder>
   ```

2. Install dependencies (Python 3.8+ recommended):




   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter Notebook:




   ```bash
   jupyter notebook
   ```

4. (Optional) Create and activate a virtual environment for isolation:




   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

5. For large datasets, ensure sufficient memory and disk space.

## Business Impact

This solution delivers measurable business value by enabling financial institutions to proactively identify and investigate anomalous transactions in large, unstructured datasets. By leveraging advanced unsupervised learning and robust feature engineering, the pipeline:

- **Enhances Fraud Detection:** Flags high-risk transactions that may evade traditional rule-based systems, improving early detection of emerging fraud patterns and reducing financial losses. The use of ensemble anomaly scores increases sensitivity to novel fraud tactics.
- **Optimizes Operational Efficiency:** Prioritizes cases for manual review, allowing fraud teams to focus resources on the most suspicious events and streamline investigation workflows. Automated explanations support rapid triage and escalation.
- **Supports Regulatory Compliance:** Provides transparent, interpretable explanations for flagged anomalies, facilitating audit trails and compliance with financial regulations **(e.g., Anti-Money Laundry (AML), Know Your Customer (KYC))**. The pipeline can be extended to support periodic reporting and regulator queries.
- **Improves Customer Trust:** Minimizes false positives and ensures legitimate transactions are not unnecessarily blocked, maintaining a positive customer experience. The system can be tuned to balance risk and service quality.
- **Enables Strategic Insights:** Surfaces behavioral trends and outlier patterns that inform risk management strategies, product design, and policy updates. Insights from anomaly clusters can guide new controls and targeted interventions.

**Recommendations for Deployment and Next Steps:**

- Integrate the pipeline with real-time transaction monitoring systems for continuous anomaly detection.
- Collaborate with fraud analysts to refine feature heuristics and feedback loops.
- Extend the solution to incorporate supervised models as labeled data becomes available.
- Monitor model drift and retrain regularly to adapt to changing fraud patterns.
- Document and automate the pipeline for reproducibility and scalability across business units.

---
