import pytest
import pandas as pd
import numpy as np
from analysis import (
    convert_amount, engineer_features, rule_based_anomaly_detection,
    sequence_modeling_anomaly_detection, embedding_autoencoder_anomaly_detection,
    parse_datetime
)
import os
import tempfile
from unittest.mock import patch


class TestConvertAmount:
    def test_convert_amount_usd(self):
        amount, currency = convert_amount("$100.50")
        assert amount == 100.50
        assert currency == "USD"

    def test_convert_amount_eur(self):
        amount, currency = convert_amount("€250.75")
        assert amount == 250.75
        assert currency == "EUR"

    def test_convert_amount_numeric(self):
        amount, currency = convert_amount(150.25)
        assert amount == 150.25
        assert currency is None

    def test_convert_amount_invalid(self):
        amount, currency = convert_amount("invalid")
        assert pd.isna(amount)
        assert currency is None

    def test_convert_amount_edge_cases(self):
        # Test None input
        amount, currency = convert_amount(None)
        assert pd.isna(amount)
        assert currency is None
        
        # Test empty string
        amount, currency = convert_amount("")
        assert pd.isna(amount)
        assert currency is None


class TestParseDatetime:
    def test_parse_datetime_edge_cases(self):
        # Test None input
        result = parse_datetime(None)
        assert result is None
        
        # Test empty string
        result = parse_datetime("")
        assert result is None
        
        # Test malformed date
        result = parse_datetime("2023-13-45 25:70:80")
        assert result is None


class TestRuleBasedDetection:
    def test_rule_based_detection(self):
        data = {
            "user": ["user1", "user1", "user2"],
            "amount": ["$5000", "$100", "$4000"],
            "location": ["NYC", "LA", "NYC"],
            "timestamp": ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-03 12:00:00"],
            "type": ["withdrawal", "deposit", "transfer"],
            "device": ["mobile", "web", "mobile"]
        }
        df = pd.DataFrame(data)
        result = rule_based_anomaly_detection(df)
        
        assert "anomaly_label" in result.columns
        assert "anomaly_score" in result.columns
        assert result["anomaly_label"].sum() >= 0


class TestFeatureEngineering:
    def test_engineer_features(self):
        data = {
            "user": ["user1", "user1"],
            "amount": ["$100", "$200"],
            "location": ["NYC", "LA"],
            "timestamp": ["2023-01-01 10:00:00", "2023-01-02 11:00:00"],
            "type": ["withdrawal", "deposit"],
            "device": ["mobile", "web"]
        }
        df = pd.DataFrame(data)
        result = engineer_features(df)
        
        assert "amount_value" in result.columns
        assert "currency" in result.columns
        assert "hour" in result.columns
        assert "weekday" in result.columns


class TestAnomalyDetection:
    def test_fit_isolation_forest(self):
        from analysis import fit_isolation_forest
        features = np.random.rand(100, 5)
        model = fit_isolation_forest(features, contamination=0.1)
        assert model is not None
        assert hasattr(model, 'decision_function')

    def test_score_anomalies(self):
        from analysis import fit_isolation_forest, score_anomalies
        features = np.random.rand(50, 3)
        model = fit_isolation_forest(features)
        scores = score_anomalies(model, features)
        assert len(scores) == 50
        assert isinstance(scores, np.ndarray)

    def test_explain_anomalies(self):
        from analysis import explain_anomalies
        data = {
            "anomaly_score": [0.8, 0.6, 0.4, 0.2],
            "amount_z_user": [5, 2, 1, 0],
            "is_new_device": [1, 0, 1, 0],
            "is_new_location": [1, 1, 0, 0],
            "time_diff_hours": [100, 50, 10, 5]
        }
        df = pd.DataFrame(data)
        result = explain_anomalies(df, top_n=2)
        assert len(result) == 2
        assert "explanation" in result.columns

    def test_sequence_modeling_detection(self):
        data = {
            "user": ["user1", "user1", "user2"],
            "amount": ["$100", "$200", "$300"],
            "location": ["NYC", "LA", "NYC"],
            "timestamp": ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-03 12:00:00"],
            "type": ["withdrawal", "deposit", "transfer"],
            "device": ["mobile", "web", "mobile"]
        }
        df = pd.DataFrame(data)
        result = sequence_modeling_anomaly_detection(df)
        
        assert "anomaly_label" in result.columns
        assert "anomaly_score" in result.columns
        assert "explanation" in result.columns

    def test_embedding_autoencoder_detection(self):
        # Create larger dataset to avoid PCA dimension issues
        data = {
            "user": [f"user{i}" for i in range(20)],
            "amount": [f"${100 + i*10}" for i in range(20)],
            "location": (["NYC", "LA", "Chicago", "Miami"] * 5)[:20],
            "timestamp": [f"2023-01-{i+1:02d} 10:00:00" for i in range(20)],
            "type": (["withdrawal", "deposit", "transfer", "purchase"] * 5)[:20],
            "device": (["mobile", "web", "tablet"] * 7)[:20]
        }
        df = pd.DataFrame(data)
        result = embedding_autoencoder_anomaly_detection(df)
        
        assert "anomaly_label" in result.columns
        assert "anomaly_score" in result.columns
        assert "explanation" in result.columns


class TestDataProcessing:
    def test_load_raw_data_csv(self):
        from analysis import load_raw_data
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("raw_log\ntest log")
            f.flush()
            
            try:
                result = load_raw_data(f.name)
                assert isinstance(result, pd.DataFrame)
                assert "raw_log" in result.columns
            finally:
                os.unlink(f.name)

    def test_save_parsed_logs(self):
        from analysis import save_parsed_logs
        
        df = pd.DataFrame({"user": ["user1"], "amount": ["$100"]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_parsed_logs(df, temp_dir)
            output_file = os.path.join(temp_dir, "parsed_logs.csv")
            assert os.path.exists(output_file)

    def test_save_explained_anomalies(self):
        from analysis import save_explained_anomalies, NUMERIC_COLUMNS
        
        df = pd.DataFrame({
            "user": ["user1"], 
            "amount_value": [100.123],
            "anomaly_score": [0.789]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_explained_anomalies(df, temp_dir, NUMERIC_COLUMNS)
            output_file = os.path.join(temp_dir, "top_anomalies.csv")
            assert os.path.exists(output_file)

    def test_create_visualisations(self):
        from analysis import create_visualisations
        
        df = pd.DataFrame({
            "anomaly_label": [0, 1, 0],
            "anomaly_score": [0.1, 0.8, 0.3],
            "amount_value": [100, 200, 150],
            "user": ["user1", "user2", "user3"],
            "type": ["purchase", "withdrawal", "deposit"]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_visualisations(df, temp_dir, "test_method")
            # Check if at least one visualization file was created
            files = os.listdir(temp_dir)
            html_files = [f for f in files if f.endswith('.html')]
            assert len(html_files) > 0

    def test_prepare_features_for_model(self):
        from analysis import prepare_features_for_model
        
        data = {
            "cat1": ["A", "B", "A"],
            "cat2": ["X", "Y", "X"],
            "num1": [1.0, 2.0, 3.0],
            "num2": [10.0, 20.0, 30.0]
        }
        df = pd.DataFrame(data)
        
        features, encoder, scaler = prepare_features_for_model(
            df, ["cat1", "cat2"], ["num1", "num2"]
        )
        
        assert features.shape[0] == 3
        assert features.shape[1] > 4  # Should be expanded due to one-hot encoding
        assert encoder is not None
        assert scaler is not None

    def test_save_features_with_scores(self):
        from analysis import save_features_with_scores, NUMERIC_COLUMNS
        
        df = pd.DataFrame({
            "user": ["user1"], 
            "amount_value": [100.123],
            "anomaly_score": [0.789]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_features_with_scores(df, temp_dir, NUMERIC_COLUMNS)
            output_file = os.path.join(temp_dir, "features_with_scores.csv")
            assert os.path.exists(output_file)


class TestMainFunction:
    @patch('analysis.load_raw_data')
    @patch('analysis._parse_and_diagnose_logs')
    @patch('analysis.rule_based_anomaly_detection')
    @patch('analysis.save_explained_anomalies')
    @patch('analysis.save_features_with_scores')
    @patch('analysis.create_visualisations')
    def test_main_rule_based(self, mock_viz, mock_save_features, mock_save_anomalies, 
                           mock_rule_based, mock_parse, mock_load):
        from analysis import main
        import sys
        
        # Mock command line arguments
        test_args = ['analysis.py', '--input', 'test.csv', '--method', 'rule_based']
        
        mock_load.return_value = pd.DataFrame({"raw_log": ["test"]})
        mock_parse.return_value = pd.DataFrame({"user": ["user1"]})
        mock_rule_based.return_value = pd.DataFrame({
            "anomaly_label": [1], 
            "anomaly_score": [0.8]
        })
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_load.assert_called_once()
        mock_parse.assert_called_once()
        mock_rule_based.assert_called_once()

    @patch('analysis.load_raw_data')
    @patch('analysis._parse_and_diagnose_logs')
    @patch('analysis.isolation_forest_anomaly_detection')
    @patch('analysis.save_explained_anomalies')
    @patch('analysis.save_features_with_scores')
    @patch('analysis.create_visualisations')
    def test_main_isolation_forest(self, mock_viz, mock_save_features, mock_save_anomalies, 
                                 mock_isolation, mock_parse, mock_load):
        from analysis import main
        import sys
        
        test_args = ['analysis.py', '--input', 'test.csv', '--method', 'isolation_forest']
        
        mock_load.return_value = pd.DataFrame({"raw_log": ["test"]})
        mock_parse.return_value = pd.DataFrame({"user": ["user1"]})
        mock_isolation.return_value = pd.DataFrame({
            "anomaly_label": [1], 
            "anomaly_score": [0.8]
        })
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_isolation.assert_called_once()

    @patch('analysis.load_raw_data')
    @patch('analysis._parse_and_diagnose_logs')
    @patch('analysis.sequence_modeling_anomaly_detection')
    @patch('analysis.save_explained_anomalies')
    @patch('analysis.save_features_with_scores')
    @patch('analysis.create_visualisations')
    def test_main_sequence_modeling(self, mock_viz, mock_save_features, mock_save_anomalies, 
                                  mock_sequence, mock_parse, mock_load):
        from analysis import main
        import sys
        
        test_args = ['analysis.py', '--input', 'test.csv', '--method', 'sequence_modeling']
        
        mock_load.return_value = pd.DataFrame({"raw_log": ["test"]})
        mock_parse.return_value = pd.DataFrame({"user": ["user1"]})
        mock_sequence.return_value = pd.DataFrame({
            "anomaly_label": [1], 
            "anomaly_score": [0.8]
        })
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_sequence.assert_called_once()

    @patch('analysis.load_raw_data')
    @patch('analysis._parse_and_diagnose_logs')
    @patch('analysis.embedding_autoencoder_anomaly_detection')
    @patch('analysis.save_explained_anomalies')
    @patch('analysis.save_features_with_scores')
    @patch('analysis.create_visualisations')
    def test_main_embedding_autoencoder(self, mock_viz, mock_save_features, mock_save_anomalies, 
                                      mock_embedding, mock_parse, mock_load):
        from analysis import main
        import sys
        
        test_args = ['analysis.py', '--input', 'test.csv', '--method', 'embedding_autoencoder']
        
        mock_load.return_value = pd.DataFrame({"raw_log": ["test"]})
        mock_parse.return_value = pd.DataFrame({"user": ["user1"]})
        mock_embedding.return_value = pd.DataFrame({
            "anomaly_label": [1], 
            "anomaly_score": [0.8]
        })
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        mock_embedding.assert_called_once()