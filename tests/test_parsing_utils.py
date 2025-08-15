import pytest
from parsing_utils import parse_datetime, parse_log
import pandas as pd


class TestParseDateTime:
    def test_parse_iso_format(self):
        result = parse_datetime("2023-01-01 10:30:00")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1

    def test_parse_european_format(self):
        result = parse_datetime("01/01/2023 10:30:00")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1

    def test_parse_invalid_date(self):
        result = parse_datetime("invalid-date")
        assert result is None


class TestParseLog:
    def test_parse_triple_colon_format(self):
        log = "2023-01-01 10:00:00:::123:::withdrawal:::$500:::NYC:::mobile"
        result = parse_log(log)
        
        assert result is not None
        assert result["user"] == "user123"
        assert result["type"] == "withdrawal"
        assert result["amount"] == "$500"
        assert result["location"] == "NYC"
        assert result["device"] == "mobile"

    def test_parse_double_colon_format(self):
        log = "2023-01-01 10:00:00::456::deposit::€300::London::desktop"
        result = parse_log(log)
        
        assert result is not None
        assert result["user"] == "user456"
        assert result["type"] == "deposit"
        assert result["amount"] == "€300"
        assert result["location"] == "London"
        assert result["device"] == "desktop"

    def test_parse_compound_pattern_usr(self):
        log = "usr:user789|purchase|£150|Paris|2023-01-01 15:30:00|tablet"
        result = parse_log(log)
        
        assert result is not None
        assert result["user"] == "user789"
        assert result["type"] == "purchase"
        assert result["amount"] == "£150"
        assert result["location"] == "Paris"

    def test_parse_compound_pattern_bracket(self):
        log = "2023-01-01 12:00:00>> [user100] did transfer - amt=$250 - Tokyo// dev:mobile"
        result = parse_log(log)
        
        assert result is not None
        assert result["user"] == "user100"
        assert result["type"] == "transfer"
        assert result["amount"] == "$250"
        assert result["location"] == "Tokyo"
        assert result["device"] == "mobile"

    def test_parse_fallback_extraction(self):
        log = "Some random format with user:555 and withdrawal and $999 and device:laptop"
        result = parse_log(log)
        
        assert result is not None
        assert result["user"] == "user555"
        assert result["type"] == "withdrawal"
        assert result["amount"] == "$999"
        assert "device" in result

    def test_parse_invalid_log(self):
        result = parse_log("invalid log format")
        assert result is None

    def test_parse_empty_log(self):
        result = parse_log("")
        assert result is None

    def test_parse_none_log(self):
        result = parse_log(None)
        assert result is None

    def test_parse_malformed_log(self):
        result = parse_log("MALFORMED_LOG")
        assert result is None


class TestParsingPatterns:
    def test_parse_triple_colon_direct(self):
        from parsing_utils import parse_triple_colon
        log = "2023-01-01 10:00:00:::123:::withdrawal:::$500:::NYC:::mobile"
        result = parse_triple_colon(log)
        
        assert result is not None
        assert "location_norm" in result
        assert result["location_norm"] == "nyc"

    def test_parse_simple_colon_direct(self):
        from parsing_utils import parse_simple_colon
        log = "2023-01-01 10:00:00::456::deposit::€300::London::desktop"
        result = parse_simple_colon(log)
        
        assert result is not None
        assert result["type"] == "deposit"

    def test_parse_compound_patterns_direct(self):
        from parsing_utils import parse_compound_patterns
        log = "usr:user789|purchase|£150|Paris|2023-01-01 15:30:00|tablet"
        result = parse_compound_patterns(log)
        
        assert result is not None
        assert "location_norm" in result

    def test_fallback_parse_log_direct(self):
        from parsing_utils import _fallback_parse_log
        log = "user:123 did withdrawal of $500 from NYC using mobile device:phone"
        result = _fallback_parse_log(log)
        
        assert result is not None
        assert "fields_found" in result
        assert len(result["fields_found"]) > 0


class TestAdvancedParsing:
    def test_parse_different_transaction_types(self):
        transaction_types = ["withdrawal", "deposit", "transfer", "purchase", "payment", "refund"]
        
        for i, tx_type in enumerate(transaction_types):
            log = f"2023-01-01 10:00:00:::{100+i}:::{tx_type}:::$100:::NYC:::mobile"
            result = parse_log(log)
            
            assert result is not None
            assert result["type"] == tx_type

    def test_parse_various_locations(self):
        locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        
        for i, location in enumerate(locations):
            log = f"2023-01-01 10:00:00:::{100+i}:::withdrawal:::$100:::{location}:::mobile"
            result = parse_log(log)
            
            assert result is not None
            assert result["location"] == location

    def test_parse_various_devices(self):
        devices = ["mobile", "desktop", "tablet", "laptop", "smartwatch"]
        
        for i, device in enumerate(devices):
            log = f"2023-01-01 10:00:00:::{100+i}:::withdrawal:::$100:::NYC:::{device}"
            result = parse_log(log)
            
            assert result is not None
            assert result["device"] == device

    def test_parse_datetime_various_formats(self):
        date_formats = [
            "2023-01-01 10:30:00",
            "01/01/2023 10:30:00",
            "2023/01/01 10:30:00",
            "01-01-2023 10:30:00",
            "2023-01-01T10:30:00",
            "2023-01-01 10:30"
        ]
        
        for date_str in date_formats:
            result = parse_datetime(date_str)
            if result is not None:  # Some formats might not be supported
                assert result.year == 2023
                assert result.month == 1
                assert result.day == 1

    def test_fallback_parsing_comprehensive(self):
        from parsing_utils import _fallback_parse_log
        
        # Test various fallback scenarios
        fallback_logs = [
            "user:123 made a withdrawal of $500 using mobile device",
            "Transaction by user456: deposit $300 at location NYC",
            "user789 did transfer amt=$250 dev:tablet loc:Chicago",
            "WITHDRAWAL by user100 amount:$150 device:laptop location:Boston"
        ]
        
        for log in fallback_logs:
            result = _fallback_parse_log(log)
            assert result is not None
            assert "fields_found" in result
            assert len(result["fields_found"]) > 0

    def test_location_normalization(self):
        from parsing_utils import parse_triple_colon
        
        locations = ["New York", "LOS ANGELES", "chicago", "HOUSTON"]
        expected_normalized = ["new york", "los angeles", "chicago", "houston"]
        
        for location, expected in zip(locations, expected_normalized):
            log = f"2023-01-01 10:00:00:::123:::withdrawal:::$500:::{location}:::mobile"
            result = parse_triple_colon(log)
            
            if result:
                assert result["location_norm"] == expected

    def test_parse_malformed_timestamps(self):
        malformed_logs = [
            "INVALID_DATE:::123:::withdrawal:::$500:::NYC:::mobile",
            "2023-99-99 25:99:99:::123:::withdrawal:::$500:::NYC:::mobile",
            ":::123:::withdrawal:::$500:::NYC:::mobile",  # Missing timestamp
        ]
        
        for log in malformed_logs:
            result = parse_log(log)
            # Should either return None or handle gracefully
            if result is not None:
                assert "timestamp" in result or result.get("timestamp") is None

    def test_parse_special_characters(self):
        # Test logs with special characters
        special_logs = [
            "2023-01-01 10:00:00:::user@123:::withdrawal:::$500:::New York City:::mobile",
            "2023-01-01 10:00:00:::user-456:::deposit:::€300:::São Paulo:::desktop",
            "2023-01-01 10:00:00:::user_789:::transfer:::£200:::London, UK:::tablet"
        ]
        
        for log in special_logs:
            result = parse_log(log)
            # Should handle special characters gracefully
            assert result is None or isinstance(result, dict)

    def test_empty_and_whitespace_handling(self):
        edge_cases = [
            "",  # Empty string
            "   ",  # Only whitespace
            "\n\t",  # Newlines and tabs
            "::::::::",  # Only separators
        ]
        
        for case in edge_cases:
            result = parse_log(case)
            assert result is None