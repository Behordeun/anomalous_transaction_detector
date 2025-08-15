import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from errorlogger import Logger, LogLevel, system_logger


class TestLogger:
    def test_logger_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False, debug_mode=True)
            assert logger.log_dir == Path(temp_dir)
            assert not logger.preserve_logs
            assert logger.debug_mode

    def test_log_level_enum(self):
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"

    def test_get_caller_info(self):
        current_func, parent_func = Logger._get_caller_info()
        assert isinstance(current_func, str)
        assert isinstance(parent_func, str)

    def test_format_message_info(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            message = logger._format_message(LogLevel.INFO, "Test message")
            assert "INFO" in message
            assert "Test message" in message
            assert "TIMESTAMP:" in message

    def test_format_message_with_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            error = ValueError("Test error")
            message = logger._format_message(LogLevel.ERROR, "Error occurred", error=error)
            assert "ERROR" in message
            assert "ValueError" in message
            assert "Test error" in message

    def test_format_message_with_additional_info(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            additional_info = {"key": "value", "number": 42}
            message = logger._format_message(LogLevel.INFO, "Test", additional_info=additional_info)
            assert "key: value" in message
            assert "number: 42" in message

    def test_info_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            logger.info("Test info message")
            
            info_file = Path(temp_dir) / "info.log"
            assert info_file.exists()
            content = info_file.read_text()
            assert "Test info message" in content

    def test_warning_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            logger.warning("Test warning message")
            
            warning_file = Path(temp_dir) / "warning.log"
            assert warning_file.exists()
            content = warning_file.read_text()
            assert "Test warning message" in content

    def test_error_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            error = RuntimeError("Test error")
            logger.error(error)
            
            error_file = Path(temp_dir) / "error.log"
            assert error_file.exists()
            content = error_file.read_text()
            assert "RuntimeError" in content

    def test_clear_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            logger.info("Test message")
            
            info_file = Path(temp_dir) / "info.log"
            assert info_file.exists()
            
            logger.clear_logs(force=True)
            content = info_file.read_text()
            assert content == ""

    def test_enable_disable_debug(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, debug_mode=False)
            assert not logger.debug_mode
            
            logger.enable_debug()
            assert logger.debug_mode
            
            logger.disable_debug()
            assert not logger.debug_mode

    def test_duplicate_prevention(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=True)
            
            # Log the same message twice
            logger.info("Duplicate message")
            logger.info("Duplicate message")
            
            info_file = Path(temp_dir) / "info.log"
            content = info_file.read_text()
            # Should only appear once due to deduplication
            assert content.count("Duplicate message") == 1

    def test_system_logger_exists(self):
        assert system_logger is not None
        assert isinstance(system_logger, Logger)

    @patch('errorlogger.Path.mkdir')
    def test_ensure_log_directory_creation_failure(self, mock_mkdir):
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        # Should fallback to current directory without raising exception
        logger = Logger(log_dir="invalid_path", preserve_logs=False, debug_mode=False)
        assert logger.log_dir.name == "logs"

    def test_write_log_failure_handling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            
            # Make directory read-only to cause write failure
            os.chmod(temp_dir, 0o444)
            
            try:
                # Should not raise exception, just handle gracefully
                logger.info("Test message")
            except Exception:
                pytest.fail("Logger should handle write failures gracefully")
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_dir, 0o755)

    def test_rotate_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            
            # Create a large log file
            info_file = Path(temp_dir) / "info.log"
            with open(info_file, 'w') as f:
                f.write("x" * (11 * 1024 * 1024))  # 11MB file
            
            logger.rotate_logs(max_size_mb=10)
            
            # Original file should be smaller or renamed
            assert not info_file.exists() or info_file.stat().st_size < 10 * 1024 * 1024

    def test_log_with_traceback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            
            try:
                raise ValueError("Test exception")
            except Exception as e:
                logger.error(e, exc_info=True)
            
            error_file = Path(temp_dir) / "error.log"
            content = error_file.read_text()
            assert "Traceback" in content
            assert "ValueError" in content

    def test_log_with_additional_info_and_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            error = RuntimeError("Test error")
            additional_info = {"context": "test", "value": 42}
            
            logger.error(error, additional_info=additional_info)
            
            error_file = Path(temp_dir) / "error.log"
            content = error_file.read_text()
            assert "RuntimeError" in content
            assert "context: test" in content
            assert "value: 42" in content

    def test_preserve_logs_functionality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=True)
            
            # Log some messages
            logger.info("First message")
            logger.warning("Warning message")
            
            # Clear logs with preserve_logs=True should not clear
            logger.clear_logs()
            
            info_file = Path(temp_dir) / "info.log"
            warning_file = Path(temp_dir) / "warning.log"
            
            # Files should still contain content
            assert info_file.read_text().strip() != ""
            assert warning_file.read_text().strip() != ""

    def test_session_separator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            
            # First session
            logger.info("Session 1 message")
            
            # Start new session
            logger._add_session_separator()
            
            # Second session
            logger.info("Session 2 message")
            
            info_file = Path(temp_dir) / "info.log"
            content = info_file.read_text()
            
            assert "=" * 50 in content
            assert "Session 1 message" in content
            assert "Session 2 message" in content

    def test_debug_mode_console_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False, debug_mode=True)
            
            # In debug mode, should not suppress console output
            with patch('builtins.print') as mock_print:
                logger.info("Debug message")
                mock_print.assert_called()

    def test_error_handling_in_write_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            
            # Mock file operations to raise exception
            with patch('pathlib.Path.open', side_effect=PermissionError("Access denied")):
                # Should not raise exception, just handle gracefully
                logger.info("Test message")
                
                # Logger should still be functional
                assert logger.log_dir == Path(temp_dir)

    def test_caller_info_extraction(self):
        current_func, parent_func = Logger._get_caller_info()
        
        # Should return function names as strings
        assert isinstance(current_func, str)
        assert isinstance(parent_func, str)
        assert current_func == "test_caller_info_extraction"

    def test_log_level_formatting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, preserve_logs=False)
            
            # Test all log levels
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error(Exception("Error message"))
            
            # Check that each file contains the correct level
            info_content = (Path(temp_dir) / "info.log").read_text()
            warning_content = (Path(temp_dir) / "warning.log").read_text()
            error_content = (Path(temp_dir) / "error.log").read_text()
            
            assert "INFO" in info_content
            assert "WARNING" in warning_content
            assert "ERROR" in error_content