import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from dateutil import parser as date_parser
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

from parsing_utils import parse_log

# Setup logging for production
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Feature engineering functions
NO_VALID_LOGS_ERROR = "No valid logs could be parsed. Check input data format."
###############################################################################


def convert_amount(val) -> Tuple[float, Optional[str]]:
    """Convert a currency-prefixed string into a numeric value and code.

    Examples:
        "€304.0" -> (304.00, "EUR")
        "$1215.74" -> (1215.74, "USD")
        "£2428.72" -> (2428.72, "GBP")

    Parameters
    ----------
    val : str or numeric
        Raw amount string possibly prefaced by a currency symbol.

    Returns
    -------
    (float, str)
        Tuple of numeric amount (rounded to 2 decimal places) and ISO-like currency code.
    """
    if pd.isna(val):
        return (np.nan, None)

    # Handle numeric values
    if isinstance(val, (int, float)):
        return (round(float(val), 2), None)

    # Handle string values
    if not isinstance(val, str):
        val = str(val)

    val = val.strip()
    if not val:
        return (np.nan, None)

    symbol_to_curr = {"€": "EUR", "$": "USD", "£": "GBP"}
    currency = None
    number_str = val

    # Check for currency symbol at the beginning
    if len(val) > 0 and val[0] in symbol_to_curr:
        currency = symbol_to_curr[val[0]]
        number_str = val[1:]
    # Check for currency symbol at the end
    elif len(val) > 0 and val[-1] in symbol_to_curr:
        currency = symbol_to_curr[val[-1]]
        number_str = val[:-1]

    # Remove commas used as thousand separators
    number_str = number_str.replace(",", "")

    # Try to convert to float
    try:
        amount = round(float(number_str), 2)
        if amount < 0:
            amount = abs(amount)  # Handle negative amounts
    except (ValueError, TypeError):
        amount = np.nan

    return (amount, currency)


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse a date/time string into a :class:`datetime` object.


    The raw timestamps in the logs come in two primary formats:

    - ISO style "YYYY-MM-DD HH:MM:SS" (e.g. "2025-07-05 19:18:10")
    - European style "DD/MM/YYYY HH:MM:SS" (e.g. "24/07/2025 22:47:06")

    Using ``dayfirst=True`` unconditionally can lead to mis-parsed
    values for ISO strings (interpreting ``2025-06-12`` as
    ``2025-12-06``).  To mitigate this, we infer the correct
    ``dayfirst`` flag based on the delimiter present.  If the
    timestamp contains a forward slash ``/`` we assume day comes
    first; otherwise we assume the canonical ISO ordering.

    Parameters
    ----------
    dt_str : str
        Date/time string extracted from a log.

    Returns
    -------
    datetime or None
        Parsed datetime or ``None`` if parsing fails.
    """
    if pd.isna(dt_str):
        return None
    dt_str = dt_str.strip()
    if not dt_str:
        return None
    # Determine the appropriate day/month ordering
    dayfirst = "/" in dt_str
    try:
        return date_parser.parse(dt_str, dayfirst=dayfirst)
    except (ValueError, TypeError):
        return None


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dt"] = df["timestamp"].apply(parse_datetime)

    # Handle amount conversion more safely
    def safe_convert_amount(val):
        try:
            result = convert_amount(val)
        except Exception as e:
            logging.error(f"Exception in safe_convert_amount for value: {val} | {e}")
            return (np.nan, "UNKNOWN")
        # Force output to always be a (float, str) tuple
        if isinstance(result, tuple) and len(result) == 2:
            return result
        if (
            isinstance(result, float)
            or isinstance(result, int)
            or result is None
            or pd.isna(result)
        ):
            logging.error(
                f"convert_amount returned invalid non-tuple (float, int, nan or None) for value: {val} -> {result}"
            )
            return (np.nan, None)
        logging.error(
            f"convert_amount returned non-tuple for value: {val} -> {result} (type: {type(result)})"
        )
        return (np.nan, None)

    converted = df["amount"].apply(safe_convert_amount)
    df["amount_value"] = converted.apply(lambda x: x[0])
    df["currency"] = converted.apply(lambda x: x[1] if x[1] is not None else "UNKNOWN")
    df["hour"] = df["dt"].dt.hour
    df["weekday"] = df["dt"].dt.weekday
    df["day_of_month"] = df["dt"].dt.day
    df["month"] = df["dt"].dt.month
    df["type"] = df["type"].str.lower()
    # Ensure required columns exist and fill missing with mode (most frequent value), fallback to 'UNKNOWN'
    required_cols = ["currency", "type", "location", "device", "weekday"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        mode_val = df[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
        df[col] = df[col].fillna(fill_val)
    return df


def _add_sequential_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user", "dt"])
    df["prev_dt"] = df.groupby("user")["dt"].shift(1)
    df["time_diff_hours"] = (
        (df["dt"] - df["prev_dt"]).dt.total_seconds() / 3600
    ).round(2)
    df["time_diff_hours"] = df["time_diff_hours"].fillna(df["time_diff_hours"].median())
    df["prev_device"] = df.groupby("user")["device"].shift(1)
    df["prev_location"] = df.groupby("user")["location"].shift(1)
    df["is_new_device"] = (df["device"] != df["prev_device"]).astype(int)
    df["is_new_location"] = (df["location"] != df["prev_location"]).astype(int)
    return df


def _add_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["user_amount_median"] = (
        df.groupby("user")["amount_value"].transform(
            lambda x: x.expanding().median().shift(1)
        )
    ).round(2)
    df["user_amount_std"] = (
        df.groupby("user")["amount_value"].transform(
            lambda x: x.expanding().std(ddof=1).shift(1)
        )
    ).round(2)
    global_median = df["amount_value"].median()
    global_std = df["amount_value"].std()
    df["user_amount_median"] = df["user_amount_median"].fillna(global_median)
    df["user_amount_std"] = df["user_amount_std"].fillna(global_std)
    return df


def _add_z_score(df: pd.DataFrame) -> pd.DataFrame:
    df["amount_z_user"] = (
        (
            (df["amount_value"] - df["user_amount_median"])
            / df["user_amount_std"].replace(0, np.nan)
        )
        .fillna(0)
        .round(2)
    )
    return df


def _round_numeric(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    for col in numeric_cols:
        df[col] = df[col].round(2)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional features from the parsed logs.
    """
    df = _add_basic_features(df)
    # Always create normalized location column
    df["location_norm"] = df["location"].astype(str).str.strip().str.lower()
    df = _add_sequential_features(df)
    df = _add_user_stats(df)
    df = _add_z_score(df)
    numeric_cols = [
        "amount_value",
        "time_diff_hours",
        "user_amount_median",
        "user_amount_std",
        "amount_z_user",
    ]
    df = _round_numeric(df, numeric_cols)
    df.drop(columns=["prev_dt", "prev_device", "prev_location"], inplace=True)
    return df


###############################################################################
# Modeling and interpretation
###############################################################################


def prepare_features_for_model(
    df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]
) -> Tuple[np.ndarray, OneHotEncoder, StandardScaler]:
    """Encode categorical variables and scale numeric ones.

    This function uses one-hot encoding for categorical features and
    z-score standardisation for numeric features.  The encoders and
    scalers are returned alongside the encoded feature matrix so that
    they can be reused for inference or explanation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all engineered features.
    categorical_cols : list of str
        Column names to one-hot encode.
    numeric_cols : list of str
        Column names to scale.

    Returns
    -------
    (np.ndarray, OneHotEncoder, StandardScaler)
        Tuple of the encoded feature matrix, the fitted encoder and
        scaler.
    """
    # One-hot encode categorical columns
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_matrix = ohe.fit_transform(df[categorical_cols])
    # Standardise numeric columns
    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(df[numeric_cols])
    # Combine into a single feature matrix
    features = np.hstack([num_matrix, cat_matrix])
    return features, ohe, scaler


def fit_isolation_forest(X: np.ndarray, contamination: float = 0.02) -> IsolationForest:
    """Train an Isolation Forest on the given feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    contamination : float
        The proportion of outliers in the data set.  This parameter
        controls the threshold on the anomaly score; smaller values
        produce fewer flagged anomalies.  The default of 2% is a
        reasonable starting point for financial fraud detection.

    Returns
    -------
    IsolationForest
        Fitted model ready to score new observations.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso


def score_anomalies(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Compute anomaly scores (the lower, the more anomalous).

    IsolationForest returns the opposite of the anomaly score
    (i.e. larger negative numbers correspond to outliers).  We negate
    the scores so that larger positive values denote greater
    abnormality.

    Parameters
    ----------
    model : IsolationForest
        Trained isolation forest.
    X : np.ndarray
        Encoded feature matrix.

    Returns
    -------
    np.ndarray
        Array of anomaly scores.
    """
    raw_scores = model.decision_function(X)
    return -raw_scores


def explain_anomalies(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Generate simple explanations for the most anomalous events.

    The explanation heuristic flags which engineered features deviate
    strongly from the norm.  For each of the top N anomalies, it
    reports:

    - If the amount is more than 3 standard deviations above the user's
      historical mean (``amount_z_user > 3``).
    - If the transaction occurred on an unusual device for the user
      (``is_new_device == 1``).
    - If the transaction took place at a new location for the user
      (``is_new_location == 1``).
    - If the time since the previous transaction is anomalously large
      (above the 95th percentile).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed anomaly scores and engineered features.
    top_n : int
        Number of anomalies to explain.

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` containing the top anomalies along with an
        ``explanation`` column describing why each event might be
        suspicious.
    """
    # Compute percentile threshold for time_diff_hours
    time_threshold = df["time_diff_hours"].quantile(0.95)

    def build_explanation(row: pd.Series) -> str:
        reasons = []
        if row["amount_z_user"] > 3:
            reasons.append("amount far above user average")
        if row["is_new_device"] == 1:
            reasons.append("first time using device")
        if row["is_new_location"] == 1:
            reasons.append("unseen location")
        if row["time_diff_hours"] > time_threshold:
            reasons.append("unusual time gap since last txn")
        if reasons:
            return "; ".join(reasons)
        else:
            return (
                "Anomaly detected: This transaction does not match typical patterns for this user, "
                "but no single feature stands out. It may be due to a combination of subtle changes "
                "or factors not directly captured by the main rules. Please review this event in context."
            )

    df_sorted = df.sort_values("anomaly_score", ascending=False).head(top_n).copy()
    df_sorted["explanation"] = df_sorted.apply(build_explanation, axis=1)
    return df_sorted


###############################################################################
# Utility for plotting
###############################################################################


def create_visualisations(df: pd.DataFrame, output_dir: str) -> None:
    """Generate a set of diagnostic plots and save them to disk.

    Currently this function produces:

    1. Histogram of transaction amounts coloured by anomaly label.
    2. Scatter plot of amount vs. anomaly score.
    3. Bar chart of anomaly count by transaction type.

    Additional plots can be added to enrich the analysis.  All
    figures are saved as HTML files in ``output_dir``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``amount_value``, ``anomaly_score``,
        ``anomaly_label``, and ``type``.
    output_dir : str
        Directory to write image files into.

    Note:** Chrome browser and Kaleido is required for image export. Install Kaleido via pip:

    ```
    pip install -U kaleido
    ```

    The image save syntaxes have been commented out of the pipeline script because I don't have Chrome browser installed on my machine.

    """
    os.makedirs(output_dir, exist_ok=True)
    df["anomaly_status"] = df["anomaly_label"].map({0: "Normal", 1: "Anomaly"})

    # 1. Histogram of amounts coloured by anomaly label
    fig1 = px.histogram(
        df,
        x="amount_value",
        color="anomaly_status",
        title="Distribution of transaction amounts by anomaly label",
        color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
    )
    fig1.write_html(os.path.join(output_dir, "amount_histogram.html"))
    # fig1.write_image(os.path.join(output_dir, "amount_histogram.png"))

    # 2. Scatter plot of amount vs. anomaly score
    fig2 = px.scatter(
        df,
        x="amount_value",
        y="anomaly_score",
        color="anomaly_status",
        title="Amount vs. anomaly score",
        color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
    )
    fig2.write_html(os.path.join(output_dir, "amount_vs_score.html"))
    # fig2.write_image(os.path.join(output_dir, "amount_vs_score.png"))

    # 3. Bar chart of anomaly count by transaction type
    counts = df.groupby(["type", "anomaly_status"]).size().reset_index(name="count")
    fig3 = px.bar(
        counts,
        x="type",
        y="count",
        color="anomaly_status",
        title="Anomaly counts by transaction type",
        color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
    )
    fig3.write_html(os.path.join(output_dir, "anomaly_by_type.html"))
    # fig3.write_image(os.path.join(output_dir, "anomaly_by_type.png"))

    # 4. Time series of anomalies over time
    if "dt" in df and "anomaly_label" in df:
        ts = df.copy()
        ts["date"] = ts["dt"].dt.date
        ts_counts = (
            ts.groupby(["date", "anomaly_label"]).size().reset_index(name="count")
        )
        ts_counts["anomaly_status"] = ts_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig4 = px.line(
            ts_counts,
            x="date",
            y="count",
            color="anomaly_status",
            title="Anomaly frequency over time",
            markers=True,
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        fig4.write_html(os.path.join(output_dir, "anomaly_timeseries.html"))

    # 5. Device usage patterns
    if "device" in df and "anomaly_label" in df:
        device_counts = (
            df.groupby(["device", "anomaly_label"]).size().reset_index(name="count")
        )
        device_counts["anomaly_status"] = device_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig5 = px.bar(
            device_counts,
            x="device",
            y="count",
            color="anomaly_status",
            title="Device usage and anomaly frequency",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        fig5.write_html(os.path.join(output_dir, "device_usage.html"))

    # 6. Location-based anomaly heatmap (if enough unique locations)
    if "location" in df and "anomaly_label" in df:
        # Normalize location names to avoid duplicates due to whitespace/case
            # Use normalized location from parsing step if available
            location_col = "location_norm" if "location_norm" in df.columns else "location"
            loc_counts = (
                df.drop_duplicates(subset=[location_col, "anomaly_label", "user", "timestamp"])
                  .groupby([location_col, "anomaly_label"])
                  .size().reset_index(name="count")
            )
            loc_counts["anomaly_status"] = loc_counts["anomaly_label"].map({0: "Normal", 1: "Anomaly"})
            fig6 = px.bar(
                loc_counts,
                x=location_col,
                y="count",
                color="anomaly_status",
                title="Anomaly frequency by location",
                color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
            )
            fig6.write_html(os.path.join(output_dir, "location_anomaly.html"))

    # 7. User-level anomaly frequency
    if "user" in df and "anomaly_label" in df:
        user_counts = (
            df.groupby(["user", "anomaly_label"]).size().reset_index(name="count")
        )
        user_counts["anomaly_status"] = user_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig7 = px.bar(
            user_counts,
            x="user",
            y="count",
            color="anomaly_status",
            title="User-level anomaly frequency",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        fig7.write_html(os.path.join(output_dir, "user_anomaly.html"))

    # 8. Boxplot of transaction amounts by type
    if "amount_value" in df and "type" in df:
        fig8 = px.box(
            df,
            x="type",
            y="amount_value",
            color="anomaly_status" if "anomaly_status" in df else None,
            title="Transaction amount distribution by type",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        fig8.write_html(os.path.join(output_dir, "amount_boxplot.html"))


###############################################################################
# Main execution logic
###############################################################################


def load_raw_data(input_path: str) -> pd.DataFrame:
    """Load raw transaction logs from CSV or XLSX file."""
    try:
        _, ext = os.path.splitext(input_path.lower())
        if ext == ".csv":
            df = pd.read_csv(input_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        logging.info(f"Loaded raw data from {input_path} with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        raise RuntimeError(f"Failed to read input file: {e}")


def parse_logs(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse raw logs into structured DataFrame."""

    def safe_parse(log):
        try:
            return parse_log(log)
        except Exception as e:
            logging.warning(f"Failed to parse log: {log} | Error: {e}")
            return None

    df_parsed = df_raw["raw_log"].apply(safe_parse)
    df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None])
    if df_parsed.empty:
        logging.error(NO_VALID_LOGS_ERROR)
        raise ValueError(NO_VALID_LOGS_ERROR)
    logging.info(f"Parsed {len(df_parsed)} valid logs.")
    return df_parsed


def save_parsed_logs(
    df_parsed: pd.DataFrame, output_dir: str, _numeric_cols: List[str]
) -> None:
    """Save parsed logs to CSV file."""
    df_parsed.to_csv(
        os.path.join(output_dir, "parsed_logs.csv"), index=False, header=True
    )
    logging.info(f"Saved parsed logs to {output_dir}/parsed_logs.csv")


def prepare_model_data(
    df_features: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    # Diagnostic logging for NaN counts and missing columns
    nan_counts = df_features[categorical_cols + numeric_cols].isna().sum()
    missing_cols = [
        col for col in categorical_cols + numeric_cols if col not in df_features.columns
    ]
    logging.info(f"NaN counts before dropna: {nan_counts.to_dict()}")
    if missing_cols:
        logging.error(f"Missing columns before dropna: {missing_cols}")
    df_features.dropna(subset=categorical_cols + numeric_cols, inplace=True)
    X, _ohe, _scaler = prepare_features_for_model(
        df_features, categorical_cols, numeric_cols
    )
    return X, df_features


def train_and_score(
    df_features: pd.DataFrame, X: np.ndarray, contamination: float
) -> pd.DataFrame:
    iso = fit_isolation_forest(X, contamination=contamination)
    scores = score_anomalies(iso, X)
    df_features["anomaly_score"] = scores.round(2)
    threshold = np.percentile(scores, 100 * (1 - contamination))
    df_features["anomaly_label"] = (scores >= threshold).astype(int)
    logging.info(
        f"Model trained. {df_features['anomaly_label'].sum()} anomalies detected."
    )
    return df_features


def save_explained_anomalies(
    explained: pd.DataFrame, output_dir: str, numeric_cols: List[str]
) -> None:
    # Round numeric columns to 2 decimal places before saving
    for col in numeric_cols:
        if col in explained:
            explained[col] = explained[col].round(2)
    explained.to_csv(
        os.path.join(output_dir, "top_anomalies.csv"), index=False, header=True
    )
    logging.info(f"Saved top anomalies to {output_dir}/top_anomalies.csv")


def save_features_with_scores(
    df_features: pd.DataFrame, output_dir: str, numeric_cols: List[str]
) -> None:
    """Save features with anomaly scores to CSV file."""
    # Round numeric columns to 2 decimal places before saving
    for col in numeric_cols:
        if col in df_features:
            df_features[col] = df_features[col].round(2)
    # Also round anomaly_score if it exists
    if "anomaly_score" in df_features:
        df_features["anomaly_score"] = df_features["anomaly_score"].round(2)
    df_features.to_csv(
        os.path.join(output_dir, "features_with_scores.csv"), index=False, header=True
    )
    logging.info(f"Saved features with scores to {output_dir}/features_with_scores.csv")


def run_pipeline(
    input_path: str, output_dir: str, contamination: float = 0.02, top_n: int = 30
) -> None:
    """Execute the full anomaly detection pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = [
        "amount_value",
        "time_diff_hours",
        "user_amount_median",
        "user_amount_std",
        "amount_z_user",
    ]
    categorical_cols = ["currency", "type", "location", "device", "weekday"]

    logging.info("Starting pipeline...")
    # 1. Load raw data
    df_raw = load_raw_data(input_path)

    # 2. Parse logs into structured columns
    # Diagnostic: collect parse status for each log
    parsing_diagnostics = []
    def diagnostic_safe_parse(log):
        try:
            parsed = parse_log(log)
            if parsed is None:
                parsing_diagnostics.append({
                    'original_log': log,
                    'parsed': {},
                    'status': 'failed',
                    'warning': 'Failed to parse',
                    'fields_found': 0,
                    'reason': 'No fields extracted'
                })
                return None
            # Count fields found
            found_fields = parsed.get('fields_found', []) if isinstance(parsed, dict) else []
            n_fields = len(found_fields)
            # Heuristic: partial if missing any of timestamp, user, type, amount
            required_keys = ['timestamp', 'user', 'type', 'amount']
            if not all(k in parsed and parsed.get(k) for k in required_keys):
                parsing_diagnostics.append({
                    'original_log': log,
                    'parsed': parsed,
                    'status': 'partial',
                    'warning': 'Missing required fields',
                    'fields_found': n_fields,
                    'reason': f'Fields found: {found_fields}'
                })
            else:
                parsing_diagnostics.append({
                    'original_log': log,
                    'parsed': parsed,
                    'status': 'full',
                    'warning': '',
                    'fields_found': n_fields,
                    'reason': 'All required fields found'
                })
            return parsed
        except Exception as e:
            parsing_diagnostics.append({
                'original_log': log,
                'parsed': {},
                'status': 'failed',
                'warning': str(e),
                'fields_found': 0,
                'reason': f'Exception: {str(e)}'
            })
            logging.warning(f"Failed to parse log: {log} | Error: {e}")
            return None

    df_parsed = df_raw["raw_log"].apply(diagnostic_safe_parse)
    df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None])
    # Export diagnostic report
    import csv
    diag_path = os.path.join(output_dir, "diagnostic_parsing_report.csv")
    with open(diag_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["original_log", "parsed", "status", "warning", "fields_found", "reason"])
        writer.writeheader()
        for row in parsing_diagnostics:
            # Write parsed as string for CSV
            row_out = row.copy()
            row_out["parsed"] = str(row_out["parsed"])
            writer.writerow(row_out)
    logging.info(f"Exported diagnostic parsing report to {diag_path}")
    if df_parsed.empty:
        logging.error(NO_VALID_LOGS_ERROR)
        raise ValueError(NO_VALID_LOGS_ERROR)
    save_parsed_logs(df_parsed, output_dir, numeric_cols)

    # 3. Feature engineering
    df_features = engineer_features(df_parsed)
    if df_features.empty:
        logging.error("Feature engineering produced empty DataFrame. Pipeline halted.")
        raise ValueError(
            "Feature engineering produced empty DataFrame. Check input data format and parsing."
        )
    # Check required columns
    missing_cols = [
        col for col in categorical_cols + numeric_cols if col not in df_features.columns
    ]
    if missing_cols:
        logging.error(
            f"Missing required columns after feature engineering: {missing_cols}. Pipeline halted."
        )
        raise ValueError(
            f"Missing required columns after feature engineering: {missing_cols}."
        )

    # 4. Prepare data for modeling
    X, df_features = prepare_model_data(df_features, categorical_cols, numeric_cols)
    if X.shape[0] == 0:
        logging.error("Feature matrix is empty after preparation. Pipeline halted.")
        raise ValueError(
            "Feature matrix is empty after preparation. Check input data and feature engineering."
        )

    # 5. Model training and scoring
    df_features = train_and_score(df_features, X, contamination)

    # 6. Interpret top anomalies
    explained = explain_anomalies(df_features, top_n=top_n)
    save_explained_anomalies(explained, output_dir, numeric_cols)

    # 7. Create visualisations
    create_visualisations(df_features, output_dir=output_dir)

    # 8. Save the full feature set with scores
    save_features_with_scores(df_features, output_dir, numeric_cols)

    logging.info(f"Pipeline completed successfully. Outputs saved to {output_dir}")


def main():
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
        help="Directory to store output artefacts.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Approximate proportion of anomalies in the data set (for IsolationForest).",
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

def rule_based_anomaly_detection(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Rule-based: amount > threshold AND new location for user."""
    threshold = 3000
    df = df.copy()
    df["amount_value"] = df["amount"].apply(lambda x: float(str(x).replace(",", "").replace("€", "").replace("$", "").replace("£", "")) if pd.notna(x) else 0)
    df = df.sort_values(["user", "timestamp"])
    df["prev_location"] = df.groupby("user")["location"].shift(1)
    df["is_new_location"] = (df["location"] != df["prev_location"]).astype(int)
    df["anomaly_label"] = ((df["amount_value"] > threshold) & (df["is_new_location"] == 1)).astype(int)
    df["anomaly_score"] = df["amount_value"] * df["is_new_location"]
    df["anomaly_status"] = df["anomaly_label"].map({0: "Normal", 1: "Anomaly"})
    df["explanation"] = np.where(df["anomaly_label"] == 1, "High amount + new location", "Rule not triggered")
    return df.sort_values("anomaly_score", ascending=False).head(top_n)

def sequence_modeling_anomaly_detection(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Sequence modeling: flag rare transitions in user location sequence (Markov Chain)."""
    df = df.copy()
    df = df.sort_values(["user", "timestamp"])
    df["prev_location"] = df.groupby("user")['location'].shift(1)
    transitions = df.groupby(["prev_location", "location"]).size().reset_index(name="count")
    # Relax threshold: transitions seen ≤ 3 times
    rare_transitions = transitions[transitions["count"] <= 3][["prev_location", "location"]]
    rare_transition_set = {(row["prev_location"], row["location"]) for _, row in rare_transitions.iterrows()}
    df["anomaly_label"] = df.apply(lambda row: int((row["prev_location"], row["location"]) in rare_transition_set), axis=1)
    df["anomaly_score"] = df["anomaly_label"]
    df["anomaly_status"] = df["anomaly_label"].map({0: "Normal", 1: "Anomaly"})
    df["explanation"] = np.where(df["anomaly_label"] == 1, "Rare location transition", "Common transition")
    # Always return top N rare transitions
    return df[df["anomaly_label"] == 1].head(top_n)

def embedding_autoencoder_anomaly_detection(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Embedding + autoencoder: PCA reconstruction error on text fields."""
    df = df.copy()
    text_fields = ["type", "location", "device"]
    for col in text_fields:
        df[col] = df[col].astype(str)
    X = pd.get_dummies(df[text_fields])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 98)
    df["anomaly_score"] = reconstruction_error
    df["anomaly_label"] = (reconstruction_error > threshold).astype(int)
    df["anomaly_status"] = df["anomaly_label"].map({0: "Normal", 1: "Anomaly"})
    df["explanation"] = np.where(df["anomaly_label"] == 1, "Unusual text pattern detected", "Normal pattern")
    return df[df["anomaly_label"] == 1].sort_values("anomaly_score", ascending=False).head(top_n)
