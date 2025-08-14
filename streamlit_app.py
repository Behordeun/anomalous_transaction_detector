import logging
import os

import pandas as pd
import streamlit as st

try:
    from analysis import (
        embedding_autoencoder_anomaly_detection,
        engineer_features,
        explain_anomalies,
        fit_isolation_forest,
        prepare_features_for_model,
        rule_based_anomaly_detection,
        score_anomalies,
        sequence_modeling_anomaly_detection,
    )
    from parsing_utils import parse_log
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from analysis import (
        embedding_autoencoder_anomaly_detection,
        engineer_features,
        explain_anomalies,
        fit_isolation_forest,
        prepare_features_for_model,
        rule_based_anomaly_detection,
        score_anomalies,
        sequence_modeling_anomaly_detection,
    )
    from parsing_utils import parse_log

# Custom CSS for professional look
st.markdown(
    """
    <style>
    .main {background-color: #181818; color: #f5f5f5;}
    .stApp {background-color: #181818;}
    .metric-label {font-size: 1.1em; font-weight: 600;}
    .impact-card {background: #23272f; border-radius: 10px; padding: 1.5em; margin-bottom: 1em;}
    .impact-badge {padding: 0.3em 0.8em; border-radius: 8px; font-weight: 600;}
    .badge-high {background: #d62728; color: #fff;}
    .badge-medium {background: #ff9800; color: #fff;}
    .badge-low {background: #2ca02c; color: #fff;}
    </style>
""",
    unsafe_allow_html=True,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


st.set_page_config(page_title="Fraud Detection Explorer", layout="wide")
st.markdown(
    "<h1 style='font-size:2.5em;font-weight:700;margin-bottom:0.2em;'>Fraud Detection System - Visual Explorer</h1>",
    unsafe_allow_html=True,
)

DYNAMIC_REPORT_TITLE = "Insights Based on Current Filters"
MOST_ANOMALOUS_USERS_LABEL = "**Most anomalous users:**"
MOST_ANOMALOUS_DEVICES_LABEL = "**Most anomalous devices:**"
MOST_ANOMALOUS_LOCATIONS_LABEL = "**Most anomalous locations:**"


# Sidebar: Data selection

st.sidebar.header("Analysis Controls")
method = st.sidebar.selectbox(
    "Anomaly Detection Method",
    [
        "Rule-based",
        "Isolation Forest (Statistical)",
        "Sequence Modeling",
        "Embedding + Autoencoder",
    ],
    index=1,
    help="Select the anomaly detection method to use.",
)
contamination = st.sidebar.slider(
    "Anomaly contamination rate", 0.01, 0.10, 0.02, 0.01, key="contamination_slider"
)
top_n = st.sidebar.slider("Show top N anomalies", 10, 50, 20, 5, key="top_n_slider")

st.header("Data Input")
example_path = "data/synthetic_dirty_transaction_logs.csv"
input_file = st.file_uploader(
    "Upload transaction log file (CSV or Excel)", type=["csv", "xlsx", "xls"]
)
use_example = st.checkbox("Use example dataset", value=True)


def _get_file_name_and_size_from_obj(file_obj):
    """
    Extract the file name and size from a file-like object.
    Args:
        file_obj: File-like object with 'name' and 'size' attributes.
    Returns:
        Tuple of (name, size).
    """
    name = getattr(file_obj, "name", str(file_obj))
    size = getattr(file_obj, "size", None)
    return name, size


def _get_file_name_and_size_from_path(file_path):
    """
    Get file name and size from a file path.
    Args:
        file_path: Path to the file.
    Returns:
        Tuple of (name, size).
    """
    name = str(file_path)
    size = os.path.getsize(file_path) if os.path.exists(file_path) else None
    return name, size


def file_info(file_obj_or_path):
    """
    Get file name and size from either a file object or file path.
    Args:
        file_obj_or_path: File object or file path.
    Returns:
        Tuple of (name, size).
    """
    if hasattr(file_obj_or_path, "name"):
        name, size = _get_file_name_and_size_from_obj(file_obj_or_path)
    else:
        name, size = _get_file_name_and_size_from_path(file_obj_or_path)
    return name, size


def load_input_file(file_obj_or_path):
    """
    Load a CSV or Excel file into a pandas DataFrame.
    Args:
        file_obj_or_path: File object or file path.
    Returns:
        pd.DataFrame containing the loaded data.
    Raises:
        ValueError: If file type is unsupported.
    """

    if hasattr(file_obj_or_path, "name"):
        ext = os.path.splitext(file_obj_or_path.name.lower())[1]
    else:
        ext = os.path.splitext(str(file_obj_or_path).lower())[1]
    if ext == ".csv":
        return pd.read_csv(file_obj_or_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_obj_or_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def parse_and_report(df_raw):
    """
    Parse raw logs and display summary in Streamlit.
    Args:
        df_raw: DataFrame with a 'raw_log' column.
    Returns:
        DataFrame of parsed logs.
    """
    st.write(f"Loaded {len(df_raw)} raw logs.")
    df_parsed = df_raw["raw_log"].apply(parse_log)
    df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None])
    st.write(f"Parsed {len(df_parsed)} valid logs.")
    return df_parsed


def load_and_parse_data(input_file, example_path):
    """
    Load and parse data from input file or example file, with error handling.
    Args:
        input_file: Uploaded file object.
        example_path: Path to example file.
    Returns:
        DataFrame of parsed logs or None if error.
    """
    try:
        df_raw = (
            load_input_file(input_file) if input_file else load_input_file(example_path)
        )
        df_parsed = parse_and_report(df_raw)
        return df_parsed
    except Exception as e:
        logging.error(f"Error loading or parsing data: {e}")
        st.error(f"Failed to load or parse data: {e}")
        return None


def run_pipeline(df_parsed, contamination, top_n):
    """
    Run the full fraud detection pipeline on parsed logs.
    Args:
        df_parsed: DataFrame of parsed logs.
        contamination: Expected anomaly rate.
        top_n: Number of top anomalies to explain.
    Returns:
        Tuple of (features DataFrame, explained anomalies DataFrame, categorical columns, numeric columns).
    """
    try:
        categorical_cols = ["currency", "type", "location", "device", "weekday"]
        numeric_cols = [
            "amount_value",
            "hour",
            "day_of_month",
            "month",
            "time_diff_hours",
            "is_new_device",
            "is_new_location",
            "amount_z_user",
        ]
        df_features = engineer_features(df_parsed)
        for col in ["amount_value", "time_diff_hours", "amount_z_user"]:
            if col in df_features:
                df_features[col] = df_features[col].round(2)
        df_features.dropna(subset=categorical_cols + numeric_cols, inplace=True)
        X, _, _ = prepare_features_for_model(
            df_features, categorical_cols, numeric_cols
        )
        iso = fit_isolation_forest(X, contamination=contamination)
        scores = score_anomalies(iso, X)
        df_features["anomaly_score"] = scores.round(2)
        threshold = pd.Series(scores).quantile(1 - contamination)
        df_features["anomaly_label"] = (scores >= threshold).astype(int)
        explained = explain_anomalies(df_features, top_n=top_n)
        # Visualization handled in separate functions
        df_features["anomaly_status"] = df_features["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        return df_features, explained, categorical_cols, numeric_cols
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        st.error(f"Pipeline failed: {e}")
        return None, None, None, None


def extract_amount(val):
    """
    Extract numeric amount from a string or numeric value.
    Args:
        val: Amount as string or number.
    Returns:
        float value of amount.
    """
    import re

    if isinstance(val, str):
        match = re.search(r"([\d\.]+)", val)
        return float(match.group(1)) if match else 0.0
    elif isinstance(val, (int, float)):
        return float(val)
    return 0.0


def estimate_business_impact(top_anomalies, anomaly_count):
    """
    Estimate average and total financial impact of detected anomalies.
    Args:
        top_anomalies: DataFrame of top anomalies.
        anomaly_count: Number of anomalies detected.
    Returns:
        Tuple of (average amount, estimated total loss).
    """
    if "amount" in top_anomalies:
        numeric_amounts = top_anomalies["amount"].apply(extract_amount)
        avg_amount = numeric_amounts.mean()
    else:
        avg_amount = 0
    return avg_amount, anomaly_count * avg_amount


def get_risk_badge(percent_anomalies):
    """
    Get risk level and badge class based on anomaly percentage.
    Args:
        percent_anomalies: Percentage of anomalies detected.
    Returns:
        Tuple of (risk level string, badge class string).
    """
    if percent_anomalies > 2:
        return "High", "badge-high"
    elif percent_anomalies > 1:
        return "Medium", "badge-medium"
    else:
        return "Low", "badge-low"


def render_list(label, items):
    """
    Render a labeled list of items and their counts in Streamlit.
    Args:
        label: Label for the list.
        items: Dictionary of item counts.
    """
    st.markdown(f"<span class='metric-label'>{label}</span>", unsafe_allow_html=True)
    for key, count in items.items():
        st.markdown(f"- {key} ({count} times)")


def show_dynamic_report(df_features, explained, top_n):
    """
    Display a dynamic business impact and anomaly report in Streamlit.
    Args:
        df_features: DataFrame of features and anomaly labels.
        explained: DataFrame of explained anomalies.
        top_n: Number of top anomalies to show.
    """
    st.markdown(
        f"<h2 style='margin-top:1.5em;'>{DYNAMIC_REPORT_TITLE}</h2>",
        unsafe_allow_html=True,
    )
    anomaly_count = df_features["anomaly_label"].sum()
    total_count = len(df_features)
    percent_anomalies = 100 * anomaly_count / total_count if total_count > 0 else 0
    top_anomalies = explained.head(top_n)
    most_anomalous_users = top_anomalies["user"].value_counts().head(3)
    most_anomalous_devices = top_anomalies["device"].value_counts().head(3)
    most_anomalous_locations = top_anomalies["location"].value_counts().head(3)
    avg_anomaly_score = (
        top_anomalies["anomaly_score"].mean() if not top_anomalies.empty else 0
    )

    _, est_loss = estimate_business_impact(top_anomalies, anomaly_count)
    risk_level, badge_class = get_risk_badge(percent_anomalies)

    st.markdown(
        f"""
    <div class='impact-card'>
        <span class='metric-label'>Estimated Financial Impact:</span> <span style='font-size:1.3em;font-weight:700;'>${est_loss:,.2f}</span><br>
        <span class='metric-label'>Risk Level:</span> <span class='impact-badge {badge_class}'>{risk_level}</span><br>
        <span class='metric-label'>Anomalies Detected:</span> <span style='font-size:1.1em;font-weight:600;'>{anomaly_count} ({percent_anomalies:.2f}%)</span><br>
        <span class='metric-label'>Average anomaly score (top {top_n}):</span> <span style='font-size:1.1em;font-weight:600;'>{avg_anomaly_score:.2f}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    render_list(MOST_ANOMALOUS_USERS_LABEL, most_anomalous_users)
    render_list(MOST_ANOMALOUS_DEVICES_LABEL, most_anomalous_devices)
    render_list(MOST_ANOMALOUS_LOCATIONS_LABEL, most_anomalous_locations)
    if "dt" in df_features:
        recent = df_features[
            df_features["dt"] > (df_features["dt"].max() - pd.Timedelta(days=7))
        ]
        recent_anomalies = recent["anomaly_label"].sum()
        st.markdown(
            f"<span class='metric-label'>Anomalies in last 7 days:</span> <span style='font-weight:600;'>{recent_anomalies}</span>",
            unsafe_allow_html=True,
        )


def show_time_series_anomalies(df_features, key=None):
    import plotly.express as px

    if "dt" in df_features and "anomaly_label" in df_features:
        ts = df_features.copy()
        ts["date"] = ts["dt"].dt.date
        ts_counts = (
            ts.groupby(["date", "anomaly_label"]).size().reset_index(name="count")
        )
        ts_counts["anomaly_status"] = ts_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig = px.line(
            ts_counts,
            x="date",
            y="count",
            color="anomaly_status",
            title="Transaction frequency over time",
            markers=True,
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_device_usage_patterns(df_features, key=None):
    import plotly.express as px

    if "device" in df_features and "anomaly_label" in df_features:
        device_counts = (
            df_features.groupby(["device", "anomaly_label"])
            .size()
            .reset_index(name="count")
        )
        device_counts["anomaly_status"] = device_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig = px.bar(
            device_counts,
            x="device",
            y="count",
            color="anomaly_status",
            title="Device usage patterns",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_location_heatmap(df_features, key=None):
    import plotly.express as px

    location_col = (
        "location_norm" if "location_norm" in df_features.columns else "location"
    )
    if location_col in df_features and "anomaly_label" in df_features:
        loc_counts = (
            df_features.drop_duplicates(
                subset=[location_col, "anomaly_label", "user", "timestamp"]
            )
            .groupby([location_col, "anomaly_label"])
            .size()
            .reset_index(name="count")
        )
        loc_counts["anomaly_status"] = loc_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig = px.bar(
            loc_counts,
            x=location_col,
            y="count",
            color="anomaly_status",
            title="Transaction frequency by location",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_geospatial_anomaly_map(df_features, key=None):
    import plotly.express as px

    if "latitude" in df_features and "longitude" in df_features:
        geo_df = df_features[
            (df_features["anomaly_label"] == 1)
            & df_features["latitude"].notna()
            & df_features["longitude"].notna()
        ]
        if not geo_df.empty:
            fig = px.scatter_mapbox(
                geo_df,
                lat="latitude",
                lon="longitude",
                hover_name="location",
                hover_data=["user", "type", "amount", "device"],
                zoom=2,
                height=500,
                title="Geospatial Distribution of Anomalies",
                color_discrete_sequence=["#d62728"],
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True, key=key)


def show_user_anomaly_frequency(df_features, key=None):
    import plotly.express as px

    if "user" in df_features and "anomaly_label" in df_features:
        user_counts = (
            df_features.groupby(["user", "anomaly_label"])
            .size()
            .reset_index(name="count")
        )
        user_counts["anomaly_status"] = user_counts["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig = px.bar(
            user_counts,
            x="user",
            y="count",
            color="anomaly_status",
            title="User-level transaction frequency",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_amount_boxplot(df_features, key=None):
    import plotly.express as px

    if "amount_value" in df_features and "type" in df_features:
        df_features["anomaly_status"] = df_features["anomaly_label"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig = px.box(
            df_features,
            x="type",
            y="amount_value",
            color="anomaly_status",
            title="Transaction amount distribution by type",
            color_discrete_map={"Normal": "#2ca02c", "Anomaly": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_visualizations(df_features):
    """
    Display a set of diagnostic and business visualizations in Streamlit.
    Args:
        df_features: DataFrame of features and anomaly labels.
    """
    st.markdown(
        "<h2 style='margin-top:2em;'>Visualizations</h2>", unsafe_allow_html=True
    )
    show_time_series_anomalies(df_features, key="full_time_series")
    show_device_usage_patterns(df_features, key="full_device_usage")
    show_location_heatmap(df_features, key="full_location_heatmap")
    show_geospatial_anomaly_map(df_features, key="full_geo_map")
    show_user_anomaly_frequency(df_features, key="full_user_anomaly")
    show_amount_boxplot(df_features, key="full_amount_boxplot")


def show_top_anomalies(explained):
    """
    Display a table of the top anomalies with explanations in Streamlit.
    Args:
        explained: DataFrame of explained anomalies.
    """
    st.markdown(
        "<h2 style='margin-top:2em;'>Top Anomalies</h2>", unsafe_allow_html=True
    )
    display_cols = [
        "timestamp",
        "user",
        "type",
        "amount",
        "location",
        "device",
        "anomaly_score",
        "explanation",
    ]
    explained_display = explained[display_cols].copy()
    explained_display["anomaly_score"] = explained_display["anomaly_score"].round(2)
    st.dataframe(explained_display, use_container_width=True)


# Ensure main() is defined before calling
def display_file_info(input_file, use_example, example_path):
    """
    Display file name and size information in the Streamlit sidebar.
    Args:
        input_file: Uploaded file object.
        use_example: Boolean, whether example file is used.
        example_path: Path to example file.
    """
    if input_file:
        name, size = file_info(input_file)
        st.sidebar.markdown(f"**File uploaded:** {name}")
        if size:
            st.sidebar.markdown(f"**Size:** {size/1024:.1f} KB")
    elif use_example:
        name, size = file_info(example_path)


def _detect_anomalies(df_parsed, contamination, top_n, method):

    if method == "Rule-based":
        df_features = df_parsed.copy()
        explained = rule_based_anomaly_detection(df_features, top_n)
        return df_features, explained
    elif method == "Isolation Forest (Statistical)":
        df_features, explained, _, _ = run_pipeline(df_parsed, contamination, top_n)
        return df_features, explained
    elif method == "Sequence Modeling":
        df_features = df_parsed.copy()
        explained = sequence_modeling_anomaly_detection(df_features, top_n)
        return df_features, explained
    elif method == "Embedding + Autoencoder":
        df_features = df_parsed.copy()
        explained = embedding_autoencoder_anomaly_detection(df_features, top_n)
        return df_features, explained
    else:
        st.error("Unknown method selected.")
        return None, None


def _show_results(df_features, explained, top_n):
    if df_features is not None and explained is not None:
        # For non-Isolation Forest methods, merge anomaly info back to full dataset
        if len(df_features) != len(explained) and "anomaly_label" in explained.columns:
            # Create full dataset with default normal labels
            df_features["anomaly_label"] = 0
            df_features["anomaly_score"] = 0.0
            df_features["anomaly_status"] = "Normal"

            # Update with anomaly information where available
            if "timestamp" in df_features.columns and "timestamp" in explained.columns:
                anomaly_indices = df_features["timestamp"].isin(explained["timestamp"])
                df_features.loc[anomaly_indices, "anomaly_label"] = 1
                df_features.loc[anomaly_indices, "anomaly_status"] = "Anomaly"
                # Map scores if available
                if "anomaly_score" in explained.columns:
                    score_map = dict(
                        zip(explained["timestamp"], explained["anomaly_score"])
                    )
                    df_features.loc[anomaly_indices, "anomaly_score"] = (
                        df_features.loc[anomaly_indices, "timestamp"]
                        .map(score_map)
                        .fillna(0.0)
                    )

        show_dynamic_report(df_features, explained, top_n)
        show_visualizations(df_features)
        show_top_anomalies(explained)
        st.markdown("<hr>", unsafe_allow_html=True)
        if isinstance(explained, pd.DataFrame):
            csv = explained.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Top Anomalies as CSV",
                data=csv,
                file_name="top_anomalies.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "Top anomalies could not be exported as CSV because the result is not a DataFrame."
            )
    else:
        st.info(
            "Upload a CSV or Excel file with a 'raw_log' column or use the example dataset."
        )


def handle_processing(input_file, example_path, contamination, top_n):
    """
    Handle file processing, pipeline execution, and result display in Streamlit.
    Args:
        input_file: Uploaded file object.
        example_path: Path to example file.
        contamination: Expected anomaly rate.
        top_n: Number of top anomalies to show.
    """
    with st.spinner("Processing file and running analysis..."):
        df_parsed = load_and_parse_data(input_file, example_path)
    if df_parsed is not None and not df_parsed.empty:
        df_features, explained = _detect_anomalies(
            df_parsed, contamination, top_n, method
        )
        _show_results(df_features, explained, top_n)
    else:
        st.info(
            "Upload a CSV or Excel file with a 'raw_log' column or use the example dataset."
        )


def main():
    """
    Main entry point for the Streamlit fraud detection app.
    Handles sidebar controls, file upload, preloading, and pipeline execution.
    """
    display_file_info(input_file, use_example, example_path)
    process_clicked = False
    if input_file or use_example:
        process_clicked = st.button(
            "Process", help="Run end-to-end analysis of the uploaded file"
        )
    if process_clicked:
        handle_processing(input_file, example_path, contamination, top_n)
    else:
        st.info("Upload a CSV or Excel file and click 'Process' to run analysis.")


if __name__ == "__main__":
    main()
