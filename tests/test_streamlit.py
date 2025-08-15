import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import streamlit_app


class TestFileOperations:
    def test_load_input_file_csv(self):
        mock_file = MagicMock()
        mock_file.name = "test.csv"

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2, 3]})
            result = streamlit_app.load_input_file(mock_file)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_load_input_file_excel(self):
        mock_file = MagicMock()
        mock_file.name = "test.xlsx"

        with patch("pandas.read_excel") as mock_read_excel:
            mock_read_excel.return_value = pd.DataFrame({"col1": [1, 2, 3]})
            result = streamlit_app.load_input_file(mock_file)
            assert isinstance(result, pd.DataFrame)

    def test_load_input_file_unsupported(self):
        mock_file = MagicMock()
        mock_file.name = "test.txt"

        with pytest.raises(ValueError, match="Unsupported file type"):
            streamlit_app.load_input_file(mock_file)

    def test_file_info_with_object(self):
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.size = 1024

        name, size = streamlit_app.file_info(mock_file)
        assert name == "test.csv"
        assert size == 1024

    def test_file_info_with_path(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp.flush()

            try:
                name, size = streamlit_app.file_info(tmp.name)
                assert tmp.name in name
                assert size is not None and size > 0
            finally:
                os.unlink(tmp.name)


class TestUtilityFunctions:
    def test_extract_amount_string(self):
        assert streamlit_app.extract_amount("$123.45") == pytest.approx(123.45)
        assert streamlit_app.extract_amount("€250.75") == pytest.approx(250.75)
        assert streamlit_app.extract_amount("100.50") == pytest.approx(100.50)

    def test_extract_amount_numeric(self):
        assert streamlit_app.extract_amount(100.50) == pytest.approx(100.50)
        assert streamlit_app.extract_amount(42) == pytest.approx(42.0)

    def test_extract_amount_invalid(self):
        assert streamlit_app.extract_amount("invalid") == pytest.approx(0.0)
        assert streamlit_app.extract_amount("") == pytest.approx(0.0)

    def test_get_risk_badge_levels(self):
        level, badge = streamlit_app.get_risk_badge(6.0)
        assert level == "High Risk"
        assert badge == "badge-high"

        level, badge = streamlit_app.get_risk_badge(3.0)
        assert level == "Medium Risk"
        assert badge == "badge-medium"

        level, badge = streamlit_app.get_risk_badge(1.0)
        assert level == "Low Risk"
        assert badge == "badge-low"

    def test_estimate_business_impact(self):
        df = pd.DataFrame({"amount": ["$100", "$200", "$300"]})
        avg, total = streamlit_app.estimate_business_impact(df, 3)
        assert avg == pytest.approx(200.0)
        assert total == pytest.approx(600.0)

    def test_estimate_business_impact_no_amount_column(self):
        df = pd.DataFrame({"user": ["user1", "user2"]})
        avg, total = streamlit_app.estimate_business_impact(df, 2)
        assert avg == 0
        assert total == 0


class TestDataParsing:
    @patch("streamlit.write")
    def test_parse_and_report_valid_logs(self, mock_write):
        df_raw = pd.DataFrame(
            {
                "raw_log": [
                    "2023-01-01 10:00:00:::123:::withdrawal:::$500:::NYC:::mobile",
                    "2023-01-02 11:00:00:::456:::deposit:::€300:::London:::desktop",
                ]
            }
        )

        result = streamlit_app.parse_and_report(df_raw)
        assert len(result) == 2
        assert "user" in result.columns

    @patch("streamlit.write")
    def test_parse_and_report_invalid_logs(self, mock_write):
        df_raw = pd.DataFrame({"raw_log": ["invalid log", "another invalid"]})

        result = streamlit_app.parse_and_report(df_raw)
        assert len(result) == 0

    @patch("streamlit_app.load_input_file")
    @patch("streamlit_app.parse_and_report")
    def test_load_and_parse_data_success(self, mock_parse, mock_load):
        mock_load.return_value = pd.DataFrame({"raw_log": ["test"]})
        mock_parse.return_value = pd.DataFrame({"user": ["user1"]})

        result = streamlit_app.load_and_parse_data("test.csv", "example.csv")
        assert result is not None
        assert len(result) == 1

    @patch("streamlit_app.load_input_file")
    @patch("streamlit.error")
    def test_load_and_parse_data_failure(self, mock_error, mock_load):
        mock_load.side_effect = Exception("File not found")

        result = streamlit_app.load_and_parse_data("invalid.csv", "example.csv")
        assert result is None
        mock_error.assert_called_once()


class TestMethodPipeline:
    @patch("streamlit_app.rule_based_anomaly_detection")
    def test_run_method_pipeline_rule_based(self, mock_rule_based):
        mock_df = pd.DataFrame({"user": ["user1"], "amount": ["$100"]})
        mock_rule_based.return_value = pd.DataFrame(
            {"anomaly_label": [0], "anomaly_score": [0.1], "anomaly_status": ["Normal"]}
        )

        result = streamlit_app.run_method_pipeline(mock_df, "Rule-based", 0.02, 20)
        assert result is not None
        mock_rule_based.assert_called_once()

    @patch("streamlit_app.sequence_modeling_anomaly_detection")
    def test_run_method_pipeline_sequence(self, mock_sequence):
        mock_df = pd.DataFrame({"user": ["user1"], "amount": ["$100"]})
        mock_sequence.return_value = pd.DataFrame(
            {"anomaly_label": [0], "anomaly_score": [0.1], "anomaly_status": ["Normal"]}
        )

        result = streamlit_app.run_method_pipeline(
            mock_df, "Sequence Modeling", 0.02, 20
        )
        assert result is not None
        mock_sequence.assert_called_once()

    @patch("streamlit_app.engineer_features")
    @patch("streamlit_app.prepare_features_for_model")
    @patch("streamlit_app.fit_isolation_forest")
    @patch("streamlit_app.score_anomalies")
    @patch("pandas.Series.quantile")
    def test_run_method_pipeline_isolation_forest(
        self, mock_quantile, mock_score, mock_fit, mock_prepare, mock_engineer
    ):
        mock_df = pd.DataFrame({"user": ["user1"], "amount": ["$100"]})

        # Mock the complete pipeline
        engineered_df = pd.DataFrame(
            {
                "currency": ["USD"],
                "type": ["purchase"],
                "location": ["NYC"],
                "device": ["mobile"],
                "weekday": [1],
                "amount_value": [100.0],
                "hour": [10],
                "day_of_month": [1],
                "month": [1],
                "time_diff_hours": [1.0],
                "is_new_device": [0],
                "is_new_location": [0],
                "amount_z_user": [0.5],
            }
        )
        mock_engineer.return_value = engineered_df

        mock_prepare.return_value = (np.array([[1, 2, 3]]), None, None)
        mock_fit.return_value = MagicMock()
        mock_score.return_value = np.array([0.5])
        mock_quantile.return_value = 0.4

        result = streamlit_app.run_method_pipeline(
            mock_df, "Isolation Forest (Statistical)", 0.02, 20
        )
        assert result is not None
        assert "anomaly_score" in result.columns
        assert "anomaly_label" in result.columns

    @patch("streamlit.error")
    def test_run_method_pipeline_exception_handling(self, mock_error):
        mock_df = pd.DataFrame({"user": ["user1"], "amount": ["$100"]})

        with patch(
            "streamlit_app.rule_based_anomaly_detection",
            side_effect=Exception("Test error"),
        ):
            result = streamlit_app.run_method_pipeline(mock_df, "Rule-based", 0.02, 20)

        assert result is None
        mock_error.assert_called_once()


class TestAnomalyDetection:
    def test_detect_anomalies_rule_based(self):
        mock_df = pd.DataFrame(
            {
                "user": ["user1", "user2"],
                "amount": ["$100", "$200"],
                "location": ["NYC", "LA"],
                "timestamp": ["2023-01-01 10:00:00", "2023-01-02 11:00:00"],
            }
        )

        with patch("streamlit_app.rule_based_anomaly_detection") as mock_rule:
            mock_rule.return_value = pd.DataFrame(
                {"anomaly_label": [0, 1], "anomaly_score": [0.1, 0.8]}
            )

            df_features, explained = streamlit_app._detect_anomalies(
                mock_df, 0.02, 20, "Rule-based"
            )

            assert df_features is not None
            assert explained is not None
            mock_rule.assert_called_once()

    def test_detect_anomalies_unknown_method(self):
        mock_df = pd.DataFrame({"user": ["user1"]})

        with patch("streamlit.error") as mock_error:
            result = streamlit_app._detect_anomalies(
                mock_df, 0.02, 20, "Unknown Method"
            )

            assert result == (None, None)
            mock_error.assert_called_once()

    def test_merge_anomaly_info(self):
        df_features = pd.DataFrame(
            {"timestamp": ["2023-01-01", "2023-01-02"], "user": ["user1", "user2"]}
        )

        explained = pd.DataFrame(
            {"timestamp": ["2023-01-01"], "anomaly_score": [0.8], "anomaly_label": [1]}
        )

        result = streamlit_app._merge_anomaly_info(df_features, explained)

        assert "anomaly_label" in result.columns
        assert "anomaly_score" in result.columns
        assert result.loc[0, "anomaly_label"] == 1
        assert result.loc[1, "anomaly_label"] == 0


class TestVisualizationHelpers:
    @patch("streamlit.markdown")
    def test_render_list(self, mock_markdown):
        items = {"item1": 5, "item2": 3}
        streamlit_app.render_list("Test Label", items)

        # Should call markdown multiple times
        assert mock_markdown.call_count >= 3

    @patch("streamlit.plotly_chart")
    def test_show_time_series_anomalies(self, mock_chart):
        df = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "anomaly_label": [0, 1],
            }
        )

        streamlit_app.show_time_series_anomalies(df)
        mock_chart.assert_called_once()

    @patch("streamlit.plotly_chart")
    def test_show_device_usage_patterns(self, mock_chart):
        df = pd.DataFrame({"device": ["mobile", "web"], "anomaly_label": [0, 1]})

        streamlit_app.show_device_usage_patterns(df)
        mock_chart.assert_called_once()

    @patch("streamlit.plotly_chart")
    def test_show_location_heatmap(self, mock_chart):
        df = pd.DataFrame(
            {
                "location": ["NYC", "LA"],
                "anomaly_label": [0, 1],
                "user": ["user1", "user2"],
                "timestamp": ["2023-01-01", "2023-01-02"],
            }
        )

        streamlit_app.show_location_heatmap(df)
        mock_chart.assert_called_once()

    @patch("streamlit.plotly_chart")
    def test_show_amount_boxplot(self, mock_chart):
        df = pd.DataFrame(
            {
                "amount_value": [100, 200],
                "type": ["purchase", "withdrawal"],
                "anomaly_label": [0, 1],
            }
        )

        streamlit_app.show_amount_boxplot(df)
        mock_chart.assert_called_once()


class TestBusinessIntelligence:
    @patch("streamlit.markdown")
    def test_show_dynamic_report(self, mock_markdown):
        df_features = pd.DataFrame(
            {"anomaly_label": [0, 1, 0], "user": ["user1", "user2", "user3"]}
        )

        explained = pd.DataFrame(
            {
                "user": ["user2"],
                "device": ["mobile"],
                "location": ["NYC"],
                "anomaly_score": [0.8],
            }
        )

        streamlit_app.show_dynamic_report(df_features, explained, 10)

        # Should call markdown multiple times for the report
        assert mock_markdown.call_count > 0

    @patch("streamlit.dataframe")
    def test_show_top_anomalies(self, mock_dataframe):
        explained = pd.DataFrame(
            {
                "timestamp": ["2023-01-01"],
                "user": ["user1"],
                "type": ["purchase"],
                "amount": ["$100"],
                "location": ["NYC"],
                "device": ["mobile"],
                "anomaly_score": [0.8],
                "explanation": ["Test explanation"],
            }
        )

        streamlit_app.show_top_anomalies(explained)
        mock_dataframe.assert_called_once()


class TestIntegration:
    @patch("streamlit_app.show_dynamic_report")
    @patch("streamlit_app.show_visualizations")
    @patch("streamlit_app.show_top_anomalies")
    @patch("streamlit.download_button")
    def test_show_results_complete_flow(
        self, mock_download, mock_top, mock_viz, mock_report
    ):
        df_features = pd.DataFrame(
            {"anomaly_label": [0, 1], "timestamp": ["2023-01-01", "2023-01-02"]}
        )

        explained = pd.DataFrame(
            {"timestamp": ["2023-01-02"], "anomaly_score": [0.8], "anomaly_label": [1]}
        )

        streamlit_app._show_results(df_features, explained, 10)

        mock_report.assert_called_once()
        mock_viz.assert_called_once()
        mock_top.assert_called_once()
        mock_download.assert_called_once()

    @patch("streamlit.info")
    def test_show_results_none_inputs(self, mock_info):
        streamlit_app._show_results(None, None, 10)
        mock_info.assert_called_once()

    @patch("streamlit_app.load_and_parse_data")
    @patch("streamlit_app._detect_anomalies")
    @patch("streamlit_app._show_results")
    @patch("streamlit.spinner")
    def test_handle_processing_success(
        self, mock_spinner, mock_show, mock_detect, mock_load
    ):
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        mock_load.return_value = pd.DataFrame({"user": ["user1"]})
        mock_detect.return_value = (
            pd.DataFrame({"anomaly_label": [1]}),
            pd.DataFrame({"anomaly_score": [0.8]}),
        )

        streamlit_app.handle_processing("test.csv", "example.csv", 0.02, 20)

        mock_load.assert_called_once()
        mock_detect.assert_called_once()
        mock_show.assert_called_once()

    @patch("streamlit_app.load_and_parse_data")
    @patch("streamlit.info")
    def test_handle_processing_no_data(self, mock_info, mock_load):
        mock_load.return_value = None

        streamlit_app.handle_processing("test.csv", "example.csv", 0.02, 20)

        mock_info.assert_called_once()


class TestEdgeCases:
    def test_extract_amount_edge_cases(self):
        assert streamlit_app.extract_amount(None) == pytest.approx(0.0)
        assert streamlit_app.extract_amount("") == pytest.approx(0.0)
        assert streamlit_app.extract_amount("no_numbers") == pytest.approx(0.0)

    def test_file_info_edge_cases(self):
        # Test with object without size attribute
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        del mock_file.size  # Remove size attribute

        name, size = streamlit_app.file_info(mock_file)
        assert name == "test.csv"
        assert size is None

    def test_estimate_business_impact_edge_cases(self):
        # Empty dataframe
        df = pd.DataFrame({"amount": []})
        avg, total = streamlit_app.estimate_business_impact(df, 0)
        assert avg == 0 or pd.isna(avg)  # Could be NaN for empty series
        assert total == 0 or pd.isna(total)

    def test_get_risk_badge_boundary_values(self):
        # Test exact boundary values
        level, _ = streamlit_app.get_risk_badge(5.0)
        assert level == "Medium Risk"  # 5.0 is not > 5

        level = streamlit_app.get_risk_badge(5.1)[0]
        assert level == "High Risk"

        level, _ = streamlit_app.get_risk_badge(2.0)
        assert level == "Low Risk"  # 2.0 is not > 2
