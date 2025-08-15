"""
Anomaly Detection System for Financial Transaction Analysis.

This module provides comprehensive anomaly detection capabilities for financial
transaction logs using multiple detection algorithms including Isolation Forest,
rule-based detection, sequence modeling, and embedding-based approaches.
"""

import argparse
import csv
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from parsing_utils import parse_datetime, parse_log

# Constants
NO_VALID_LOGS_ERROR = "No valid logs could be parsed. Check input data format."
CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}
REQUIRED_COLUMNS = ["currency", "type", "location", "device", "weekday"]
NUMERIC_COLUMNS = [
    "amount_value",
    "time_diff_hours",
    "user_amount_median",
    "user_amount_std",
    "amount_z_user",
]
CATEGORICAL_COLUMNS = ["currency", "type", "location", "device", "weekday"]
COLOR_MAP = {"Normal": "#2ca02c", "Anomaly": "#d62728"}
AMOUNT_Z_THRESHOLD = 3
TIME_PERCENTILE = 0.95
RARE_TRANSITION_THRESHOLD = 3
PCA_PERCENTILE = 98
RULE_THRESHOLD = 3000

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def convert_amount(val: Union[str, int, float]) -> Tuple[float, Optional[str]]:
    """Convert currency-prefixed string into numeric value and currency code.

    Args:
        val: Raw amount string possibly prefaced by a currency symbol.

    Returns:
        Tuple of numeric amount (rounded to 2 decimal places) and currency code.

    Examples:
        "€304.0" -> (304.00, "EUR")
        "$1215.74" -> (1215.74, "USD")
        "£2428.72" -> (2428.72, "GBP")
    """
    if pd.isna(val):
        return (np.nan, None)

    if isinstance(val, (int, float)):
        return (round(float(val), 2), None)

    val_str = str(val).strip()
    if not val_str:
        return (np.nan, None)

    currency = None
    number_str = val_str

    # Check for currency symbol at beginning or end
    if val_str and val_str[0] in CURRENCY_SYMBOLS:
        currency = CURRENCY_SYMBOLS[val_str[0]]
        number_str = val_str[1:]
    elif val_str and val_str[-1] in CURRENCY_SYMBOLS:
        currency = CURRENCY_SYMBOLS[val_str[-1]]
        number_str = val_str[:-1]

    # Remove thousand separators
    number_str = number_str.replace(",", "")

    try:
        amount = round(float(number_str), 2)
        return (abs(amount), currency)  # Handle negative amounts
    except (ValueError, TypeError):
        return (np.nan, currency)


def _safe_convert_amount(val: Union[str, int, float]) -> Tuple[float, str]:
    """Safely convert amount with error handling."""
    try:
        result = convert_amount(val)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        logging.error(f"Invalid convert_amount result for {val}: {result}")
        return (np.nan, "UNKNOWN")
    except Exception as exc:
        logging.error(f"Exception in amount conversion for {val}: {exc}")
        return (np.nan, "UNKNOWN")


def _add_basic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add basic temporal and currency features."""
    dataframe = dataframe.copy()
    dataframe["dt"] = dataframe["timestamp"].apply(parse_datetime)

    converted = dataframe["amount"].apply(_safe_convert_amount)
    dataframe["amount_value"] = converted.apply(lambda x: x[0])
    dataframe["currency"] = converted.apply(
        lambda x: x[1] if x[1] is not None else "UNKNOWN"
    )

    # Add temporal features
    dataframe["hour"] = dataframe["dt"].dt.hour
    dataframe["weekday"] = dataframe["dt"].dt.weekday
    dataframe["day_of_month"] = dataframe["dt"].dt.day
    dataframe["month"] = dataframe["dt"].dt.month
    dataframe["type"] = dataframe["type"].str.lower()

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in dataframe.columns:
            dataframe[col] = "UNKNOWN"
        mode_val = dataframe[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
        dataframe[col] = dataframe[col].fillna(fill_val)

    return dataframe


def _add_sequential_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add sequential features based on user transaction history."""
    dataframe = dataframe.sort_values(["user", "dt"])
    dataframe["prev_dt"] = dataframe.groupby("user")["dt"].shift(1)
    dataframe["time_diff_hours"] = (
        (dataframe["dt"] - dataframe["prev_dt"]).dt.total_seconds() / 3600
    ).round(2)

    median_time_diff = dataframe["time_diff_hours"].median()
    dataframe["time_diff_hours"] = dataframe["time_diff_hours"].fillna(median_time_diff)

    dataframe["prev_device"] = dataframe.groupby("user")["device"].shift(1)
    dataframe["prev_location"] = dataframe.groupby("user")["location"].shift(1)
    dataframe["is_new_device"] = (
        dataframe["device"] != dataframe["prev_device"]
    ).astype(int)
    dataframe["is_new_location"] = (
        dataframe["location"] != dataframe["prev_location"]
    ).astype(int)

    return dataframe


def _add_user_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add user-level statistical features."""
    dataframe["user_amount_median"] = (
        dataframe.groupby("user")["amount_value"]
        .transform(lambda x: x.expanding().median().shift(1))
        .round(2)
    )
    dataframe["user_amount_std"] = (
        dataframe.groupby("user")["amount_value"]
        .transform(lambda x: x.expanding().std(ddof=1).shift(1))
        .round(2)
    )

    global_median = dataframe["amount_value"].median()
    global_std = dataframe["amount_value"].std()
    dataframe["user_amount_median"] = dataframe["user_amount_median"].fillna(
        global_median
    )
    dataframe["user_amount_std"] = dataframe["user_amount_std"].fillna(global_std)

    return dataframe


def _add_z_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add z-score for amount relative to user's historical pattern."""
    dataframe["amount_z_user"] = (
        (
            (dataframe["amount_value"] - dataframe["user_amount_median"])
            / dataframe["user_amount_std"].replace(0, np.nan)
        )
        .fillna(0)
        .round(2)
    )

    return dataframe


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute additional features from parsed logs."""
    dataframe = _add_basic_features(dataframe)
    dataframe["location_norm"] = (
        dataframe["location"].astype(str).str.strip().str.lower()
    )
    dataframe = _add_sequential_features(dataframe)
    dataframe = _add_user_statistics(dataframe)
    dataframe = _add_z_score(dataframe)

    # Round numeric columns
    for col in NUMERIC_COLUMNS:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].round(2)

    # Clean up temporary columns
    temp_cols = ["prev_dt", "prev_device", "prev_location"]
    dataframe.drop(
        columns=[col for col in temp_cols if col in dataframe.columns], inplace=True
    )

    return dataframe


def prepare_features_for_model(
    dataframe: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]
) -> Tuple[np.ndarray, OneHotEncoder, StandardScaler]:
    """Encode categorical variables and scale numeric ones.

    Args:
        dataframe: DataFrame containing all engineered features.
        categorical_cols: Column names to one-hot encode.
        numeric_cols: Column names to scale.

    Returns:
        Tuple of encoded feature matrix, fitted encoder, and scaler.
    """
    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_matrix = encoder.fit_transform(dataframe[categorical_cols])

    # Standardize numeric columns
    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(dataframe[numeric_cols])

    # Combine feature matrices
    features = np.hstack([num_matrix, cat_matrix])

    return features, encoder, scaler


def fit_isolation_forest(
    features: np.ndarray, contamination: float = 0.02
) -> IsolationForest:
    """Train Isolation Forest on feature matrix.

    Args:
        features: Feature matrix.
        contamination: Proportion of outliers in dataset.

    Returns:
        Fitted IsolationForest model.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)
    return model


def score_anomalies(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Compute anomaly scores (higher values indicate more anomalous).

    Args:
        model: Trained isolation forest.
        features: Encoded feature matrix.

    Returns:
        Array of anomaly scores.
    """
    raw_scores = model.decision_function(features)
    return -raw_scores  # Negate so higher values = more anomalous


def _build_explanation(row: pd.Series, time_threshold: float) -> str:
    """Build explanation for anomalous transaction."""
    reasons = []

    if row["amount_z_user"] > AMOUNT_Z_THRESHOLD:
        reasons.append("amount far above user average")
    if row["is_new_device"] == 1:
        reasons.append("first time using device")
    if row["is_new_location"] == 1:
        reasons.append("unseen location")
    if row["time_diff_hours"] > time_threshold:
        reasons.append("unusual time gap since last txn")

    if reasons:
        return "; ".join(reasons)

    return (
        "Anomaly detected: This transaction does not match typical patterns "
        "for this user, but no single feature stands out. It may be due to "
        "a combination of subtle changes or factors not directly captured "
        "by the main rules. Please review this event in context."
    )


def explain_anomalies(dataframe: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Generate explanations for most anomalous events.

    Args:
        dataframe: DataFrame with anomaly scores and features.
        top_n: Number of anomalies to explain.

    Returns:
        DataFrame containing top anomalies with explanations.
    """
    time_threshold = dataframe["time_diff_hours"].quantile(TIME_PERCENTILE)

    df_sorted = (
        dataframe.sort_values("anomaly_score", ascending=False).head(top_n).copy()
    )
    df_sorted["explanation"] = df_sorted.apply(
        lambda row: _build_explanation(row, time_threshold), axis=1
    )

    return df_sorted


def _create_histogram(dataframe: pd.DataFrame, output_dir: str, method: str) -> None:
    """Create amount distribution histogram."""
    amount_col = "amount_value" if "amount_value" in dataframe.columns else "amount"
    if amount_col in dataframe.columns:
        fig = px.histogram(
            dataframe,
            x=amount_col,
            color="anomaly_status",
            title=f"Transaction Amount Distribution - {method.title()}",
            color_discrete_map=COLOR_MAP,
            barmode="group",
        )
        fig.write_html(os.path.join(output_dir, "amount_histogram.html"))


def _create_scatter_plots(
    dataframe: pd.DataFrame, output_dir: str, method: str
) -> None:
    """Create scatter plot visualizations."""
    amount_col = "amount_value" if "amount_value" in dataframe.columns else "amount"

    # Amount vs anomaly score
    if amount_col in dataframe.columns:
        fig1 = px.scatter(
            dataframe,
            x=amount_col,
            y="anomaly_score",
            color="anomaly_status",
            title=f"Amount vs Anomaly Score - {method.title()}",
            color_discrete_map=COLOR_MAP,
            hover_data=["user", "type"],
        )
        fig1.write_html(os.path.join(output_dir, "amount_vs_score_scatter.html"))

    # Temporal analysis
    if "timestamp" in dataframe and amount_col in dataframe.columns:
        fig2 = px.scatter(
            dataframe,
            x="timestamp",
            y=amount_col,
            color="anomaly_status",
            title=f"Temporal Transaction Analysis - {method.title()}",
            color_discrete_map=COLOR_MAP,
            hover_data=["user", "type"],
        )
        fig2.write_html(os.path.join(output_dir, "temporal_scatter.html"))


def _create_box_plots(dataframe: pd.DataFrame, output_dir: str, method: str) -> None:
    """Create box plot visualizations."""
    amount_col = "amount_value" if "amount_value" in dataframe.columns else "amount"
    if amount_col not in dataframe.columns:
        return

    box_plot_configs = [
        (
            "type",
            "amount_by_type_boxplot.html",
            "Amount Distribution by Transaction Type",
        ),
        ("device", "amount_by_device_boxplot.html", "Amount Distribution by Device"),
    ]

    # Add location box plot
    location_col = (
        "location_norm" if "location_norm" in dataframe.columns else "location"
    )
    if location_col in dataframe:
        box_plot_configs.append(
            (
                location_col,
                "amount_by_location_boxplot.html",
                "Amount Distribution by Location",
            )
        )

    # Add user box plot if reasonable number of users
    if "user" in dataframe and dataframe["user"].nunique() <= 20:
        box_plot_configs.append(
            ("user", "amount_by_user_boxplot.html", "Amount Distribution by User")
        )

    for col, filename, title in box_plot_configs:
        if col in dataframe:
            fig = px.box(
                dataframe,
                x=col,
                y=amount_col,
                color="anomaly_status",
                title=f"{title} - {method.title()}",
                color_discrete_map=COLOR_MAP,
            )
            if col in ["device", location_col, "user"]:
                fig.update_xaxes(tickangle=45)
            fig.write_html(os.path.join(output_dir, filename))


def _create_grouped_bar_charts(
    dataframe: pd.DataFrame, output_dir: str, method: str
) -> None:
    """Create grouped bar chart visualizations."""
    bar_chart_configs = [
        ("type", "anomaly_by_type_grouped.html", "Anomaly Counts by Transaction Type"),
        ("device", "device_usage_grouped.html", "Device Usage Patterns"),
        ("user", "user_anomaly_grouped.html", "User-level Anomaly Frequency"),
    ]

    # Add location bar chart
    location_col = (
        "location_norm" if "location_norm" in dataframe.columns else "location"
    )
    if location_col in dataframe:
        bar_chart_configs.append(
            (
                location_col,
                "location_anomaly_grouped.html",
                "Anomaly Frequency by Location",
            )
        )

    for col, filename, title in bar_chart_configs:
        if col in dataframe:
            counts = (
                dataframe.groupby([col, "anomaly_status"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                counts,
                x=col,
                y="count",
                color="anomaly_status",
                title=f"{title} - {method.title()}",
                color_discrete_map=COLOR_MAP,
                barmode="group",
            )
            if col in ["device", location_col, "user"]:
                fig.update_xaxes(tickangle=45)
            fig.write_html(os.path.join(output_dir, filename))


def _create_time_series(dataframe: pd.DataFrame, output_dir: str, method: str) -> None:
    """Create time series visualization."""
    if "dt" not in dataframe:
        return

    time_series = dataframe.copy()
    time_series["date"] = time_series["dt"].dt.date
    ts_counts = (
        time_series.groupby(["date", "anomaly_status"]).size().reset_index(name="count")
    )

    fig = px.line(
        ts_counts,
        x="date",
        y="count",
        color="anomaly_status",
        title=f"Daily Anomaly Frequency - {method.title()}",
        markers=True,
        color_discrete_map=COLOR_MAP,
    )
    fig.write_html(os.path.join(output_dir, "anomaly_timeseries.html"))


def create_visualisations(
    dataframe: pd.DataFrame, output_dir: str, method: str = "isolation_forest"
) -> None:
    """Generate comprehensive diagnostic plots.

    Args:
        dataframe: DataFrame containing anomaly scores, labels, and features.
        output_dir: Directory to write visualization files.
        method: Detection method name for file naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataframe["anomaly_status"] = dataframe["anomaly_label"].map(
        {0: "Normal", 1: "Anomaly"}
    )

    _create_histogram(dataframe, output_dir, method)
    _create_scatter_plots(dataframe, output_dir, method)
    _create_box_plots(dataframe, output_dir, method)
    _create_grouped_bar_charts(dataframe, output_dir, method)
    _create_time_series(dataframe, output_dir, method)


def load_raw_data(input_path: str) -> pd.DataFrame:
    """Load raw transaction logs from CSV or XLSX file."""
    try:
        _, ext = os.path.splitext(input_path.lower())
        if ext == ".csv":
            dataframe = pd.read_csv(input_path)
        elif ext in [".xlsx", ".xls"]:
            dataframe = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        logging.info(
            "Loaded raw data from %s with %d rows.", input_path, len(dataframe)
        )
        return dataframe
    except Exception as exc:
        logging.error("Failed to read input file: %s", exc)
        raise RuntimeError(f"Failed to read input file: {exc}") from exc


def _safe_parse_log(log: str) -> Optional[Dict]:
    """Safely parse a single log entry."""
    try:
        return parse_log(log)
    except Exception as exc:
        logging.warning("Failed to parse log: %s | Error: %s", log, exc)
        return None


def parse_logs(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse raw logs into structured DataFrame."""
    df_parsed = df_raw["raw_log"].apply(_safe_parse_log)
    df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None])

    if df_parsed.empty:
        logging.error(NO_VALID_LOGS_ERROR)
        raise ValueError(NO_VALID_LOGS_ERROR)

    logging.info("Parsed %d valid logs.", len(df_parsed))
    return df_parsed


def save_parsed_logs(df_parsed: pd.DataFrame, output_dir: str) -> None:
    """Save parsed logs to CSV file."""
    output_path = os.path.join(output_dir, "parsed_logs.csv")
    df_parsed.to_csv(output_path, index=False, header=True)
    logging.info("Saved parsed logs to %s", output_path)


def prepare_model_data(
    df_features: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Prepare data for modeling with validation."""
    # Log diagnostic information
    nan_counts = df_features[categorical_cols + numeric_cols].isna().sum()
    missing_cols = [
        col for col in categorical_cols + numeric_cols if col not in df_features.columns
    ]

    logging.info("NaN counts before dropna: %s", nan_counts.to_dict())
    if missing_cols:
        logging.error("Missing columns before dropna: %s", missing_cols)

    df_features.dropna(subset=categorical_cols + numeric_cols, inplace=True)
    features, _, _ = prepare_features_for_model(
        df_features, categorical_cols, numeric_cols
    )

    return features, df_features


def train_and_score(
    df_features: pd.DataFrame, features: np.ndarray, contamination: float
) -> pd.DataFrame:
    """Train model and score anomalies."""
    model = fit_isolation_forest(features, contamination=contamination)
    scores = score_anomalies(model, features)

    df_features["anomaly_score"] = scores.round(2)
    threshold = np.percentile(scores, 100 * (1 - contamination))
    df_features["anomaly_label"] = (scores >= threshold).astype(int)

    logging.info(
        "Model trained. %d anomalies detected.", df_features["anomaly_label"].sum()
    )
    return df_features


def save_explained_anomalies(
    explained: pd.DataFrame, output_dir: str, numeric_cols: List[str]
) -> None:
    """Save explained anomalies to CSV file."""
    # Round numeric columns
    for col in numeric_cols:
        if col in explained:
            explained[col] = explained[col].round(2)

    output_path = os.path.join(output_dir, "top_anomalies.csv")
    explained.to_csv(output_path, index=False, header=True)
    logging.info("Saved top anomalies to %s", output_path)


def save_features_with_scores(
    df_features: pd.DataFrame, output_dir: str, numeric_cols: List[str]
) -> None:
    """Save features with anomaly scores to CSV file."""
    # Round numeric columns
    for col in numeric_cols:
        if col in df_features:
            df_features[col] = df_features[col].round(2)

    if "anomaly_score" in df_features:
        df_features["anomaly_score"] = df_features["anomaly_score"].round(2)

    output_path = os.path.join(output_dir, "features_with_scores.csv")
    df_features.to_csv(output_path, index=False, header=True)
    logging.info("Saved features with scores to %s", output_path)


def _build_diagnostic_record(
    log: str, parsed: Dict, status: str, warning: str, fields_found: int, reason: str
) -> Dict:
    """Build diagnostic record for parsing."""
    return {
        "original_log": log,
        "parsed": parsed,
        "status": status,
        "warning": warning,
        "fields_found": fields_found,
        "reason": reason,
    }


def _diagnostic_parse_logic(
    log: str, parsing_diagnostics: List[Dict]
) -> Optional[Dict]:
    """Parse log with diagnostic tracking."""
    try:
        parsed = parse_log(log)
        if parsed is None:
            parsing_diagnostics.append(
                _build_diagnostic_record(
                    log, {}, "failed", "Failed to parse", 0, "No fields extracted"
                )
            )
            return None

        found_fields = (
            parsed.get("fields_found", []) if isinstance(parsed, dict) else []
        )
        n_fields = len(found_fields)
        required_keys = ["timestamp", "user", "type", "amount"]

        if not all(k in parsed and parsed.get(k) for k in required_keys):
            parsing_diagnostics.append(
                _build_diagnostic_record(
                    log,
                    parsed,
                    "partial",
                    "Missing required fields",
                    n_fields,
                    f"Fields found: {found_fields}",
                )
            )
        else:
            parsing_diagnostics.append(
                _build_diagnostic_record(
                    log,
                    parsed,
                    "full",
                    "",
                    n_fields,
                    "All required fields found",
                )
            )
        return parsed
    except Exception as exc:
        parsing_diagnostics.append(
            _build_diagnostic_record(
                log, {}, "failed", str(exc), 0, f"Exception: {str(exc)}"
            )
        )
        logging.warning("Failed to parse log: %s | Error: %s", log, exc)
        return None


def _parse_and_diagnose_logs(
    df_raw: pd.DataFrame, method_output_dir: str
) -> pd.DataFrame:
    """Parse logs with diagnostic reporting."""
    parsing_diagnostics = []

    def diagnostic_safe_parse(log):
        return _diagnostic_parse_logic(log, parsing_diagnostics)

    df_parsed = df_raw["raw_log"].apply(diagnostic_safe_parse)
    df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None])

    # Export diagnostic report
    diag_path = os.path.join(method_output_dir, "diagnostic_parsing_report.csv")
    with open(diag_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "original_log",
                "parsed",
                "status",
                "warning",
                "fields_found",
                "reason",
            ],
        )
        writer.writeheader()
        for row in parsing_diagnostics:
            row_out = row.copy()
            row_out["parsed"] = str(row_out["parsed"])
            writer.writerow(row_out)

    logging.info("Exported diagnostic parsing report to %s", diag_path)

    if df_parsed.empty:
        logging.error(NO_VALID_LOGS_ERROR)
        raise ValueError(NO_VALID_LOGS_ERROR)

    save_parsed_logs(df_parsed, method_output_dir)
    return df_parsed


def _validate_features(df_features: pd.DataFrame) -> None:
    """Validate feature engineering results."""
    if df_features.empty:
        logging.error("Feature engineering produced empty DataFrame.")
        raise ValueError(
            "Feature engineering produced empty DataFrame. "
            "Check input data format and parsing."
        )

    missing_cols = [
        col
        for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
        if col not in df_features.columns
    ]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        raise ValueError(f"Missing required columns: {missing_cols}")


def run_pipeline(
    input_path: str,
    output_dir: str,
    contamination: float = 0.02,
    top_n: int = 30,
    method: str = "isolation_forest",
) -> None:
    """Execute the full anomaly detection pipeline."""
    method_output_dir = os.path.join(output_dir, method)
    os.makedirs(method_output_dir, exist_ok=True)

    logging.info("Starting pipeline...")

    # Load and parse data
    df_raw = load_raw_data(input_path)
    df_parsed = _parse_and_diagnose_logs(df_raw, method_output_dir)

    # Feature engineering
    df_features = engineer_features(df_parsed)
    _validate_features(df_features)

    # Model preparation and training
    features, df_features = prepare_model_data(
        df_features, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
    )

    if features.shape[0] == 0:
        logging.error("Feature matrix is empty after preparation.")
        raise ValueError("Feature matrix is empty after preparation.")

    df_features = train_and_score(df_features, features, contamination)

    # Interpretation and visualization
    explained = explain_anomalies(df_features, top_n=top_n)
    save_explained_anomalies(explained, method_output_dir, NUMERIC_COLUMNS)

    create_visualisations(df_features, output_dir=method_output_dir, method=method)
    save_features_with_scores(df_features, method_output_dir, NUMERIC_COLUMNS)

    logging.info(
        "Pipeline completed successfully. Outputs saved to %s", method_output_dir
    )


def rule_based_anomaly_detection(
    dataframe: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Rule-based anomaly detection: high amount + new location."""
    dataframe = dataframe.copy()

    # Extract numeric amount using the same logic as feature engineering
    converted = dataframe["amount"].apply(_safe_convert_amount)
    dataframe["amount_value"] = converted.apply(lambda x: x[0])

    dataframe = dataframe.sort_values(["user", "timestamp"])
    dataframe["prev_location"] = dataframe.groupby("user")["location"].shift(1)
    dataframe["is_new_location"] = (
        dataframe["location"] != dataframe["prev_location"]
    ).astype(int)

    # Apply rule: high amount AND new location
    dataframe["anomaly_label"] = (
        (dataframe["amount_value"] > RULE_THRESHOLD)
        & (dataframe["is_new_location"] == 1)
    ).astype(int)

    dataframe["anomaly_score"] = (
        dataframe["amount_value"] * dataframe["is_new_location"]
    )
    dataframe["anomaly_status"] = dataframe["anomaly_label"].map(
        {0: "Normal", 1: "Anomaly"}
    )
    dataframe["explanation"] = np.where(
        dataframe["anomaly_label"] == 1,
        "High amount + new location",
        "Rule not triggered",
    )

    return dataframe


def sequence_modeling_anomaly_detection(
    dataframe: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Sequence modeling: flag rare location transitions."""
    dataframe = dataframe.copy()

    # Add amount_value for visualization compatibility
    converted = dataframe["amount"].apply(_safe_convert_amount)
    dataframe["amount_value"] = converted.apply(lambda x: x[0])

    dataframe = dataframe.sort_values(["user", "timestamp"])
    dataframe["prev_location"] = dataframe.groupby("user")["location"].shift(1)

    # Find rare transitions
    transitions = (
        dataframe.groupby(["prev_location", "location"])
        .size()
        .reset_index(name="count")
    )
    rare_transitions = transitions[transitions["count"] <= RARE_TRANSITION_THRESHOLD][
        ["prev_location", "location"]
    ]

    rare_transition_set = {
        (row["prev_location"], row["location"])
        for _, row in rare_transitions.iterrows()
    }

    dataframe["anomaly_label"] = dataframe.apply(
        lambda row: int((row["prev_location"], row["location"]) in rare_transition_set),
        axis=1,
    )

    dataframe["anomaly_score"] = dataframe["anomaly_label"]
    dataframe["anomaly_status"] = dataframe["anomaly_label"].map(
        {0: "Normal", 1: "Anomaly"}
    )
    dataframe["explanation"] = np.where(
        dataframe["anomaly_label"] == 1, "Rare location transition", "Common transition"
    )

    return dataframe


def embedding_autoencoder_anomaly_detection(
    dataframe: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Embedding + autoencoder: PCA reconstruction error on text fields."""
    dataframe = dataframe.copy()

    # Add amount_value for visualization compatibility
    converted = dataframe["amount"].apply(_safe_convert_amount)
    dataframe["amount_value"] = converted.apply(lambda x: x[0])

    text_fields = ["type", "location", "device"]
    for col in text_fields:
        dataframe[col] = dataframe[col].astype(str)

    # Create embeddings and apply PCA
    features = pd.get_dummies(dataframe[text_fields])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=min(10, x_scaled.shape[1]))
    x_pca = pca.fit_transform(x_scaled)
    x_reconstructed = pca.inverse_transform(x_pca)

    # Calculate reconstruction error
    reconstruction_error = np.mean((x_scaled - x_reconstructed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, PCA_PERCENTILE)

    dataframe["anomaly_score"] = reconstruction_error
    dataframe["anomaly_label"] = (reconstruction_error > threshold).astype(int)
    dataframe["anomaly_status"] = dataframe["anomaly_label"].map(
        {0: "Normal", 1: "Anomaly"}
    )
    dataframe["explanation"] = np.where(
        dataframe["anomaly_label"] == 1,
        "Unusual text pattern detected",
        "Normal pattern",
    )

    return dataframe


def main() -> None:
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Detect anomalous financial transactions from raw logs."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the raw CSV file with a 'raw_log' column.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store output artifacts.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Approximate proportion of anomalies in the dataset.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of top anomalies to explain and save.",
    )

    args = parser.parse_args()
    run_pipeline(
        args.input, args.output_dir, contamination=args.contamination, top_n=args.top_n
    )


if __name__ == "__main__":
    main()
