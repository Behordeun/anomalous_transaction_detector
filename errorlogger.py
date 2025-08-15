import inspect
import sys
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Logger:
    def __init__(
        self,
        log_dir: str = "logs",
        preserve_logs: bool = True,
        debug_mode: bool = False,
    ) -> None:
        """
        Initialize the Logger with specified configuration.

        Args:
            log_dir: Directory to store log files (default: "logs").
            preserve_logs: Whether to preserve existing logs (default: True).
            debug_mode: Whether to enable debug output (default: False).
        """
        self.log_dir = Path(log_dir)
        self.preserve_logs = preserve_logs
        self.debug_mode = debug_mode  # Control debug output
        self.log_files = {
            LogLevel.INFO: self.log_dir / "info.log",
            LogLevel.WARNING: self.log_dir / "warning.log",
            LogLevel.ERROR: self.log_dir / "error.log",
        }
        self._ensure_log_directory()
        self._log_cache = self._load_existing_log_hashes() if preserve_logs else set()

        # Only print debug info if debug mode is enabled
        if self.debug_mode:
            print(f"Logger initialized with directory: {self.log_dir.absolute()}")
            print("Log files will be created at:")
            for level, path in self.log_files.items():
                print(f"  {level.value}: {path.absolute()}")

    def _ensure_log_directory(self) -> None:
        """Create the log directory if it doesn't exist."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            if self.debug_mode:
                print(f"Log directory created/verified: {self.log_dir.absolute()}")

            # Test write permissions silently
            test_file = self.log_dir / "test_write.tmp"
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()  # Remove test file

                if self.debug_mode:
                    print("Write permissions verified")
            except Exception as perm_error:
                # Always show permission errors as they're critical
                print(
                    f"Permission error in log directory: {perm_error}", file=sys.stderr
                )

        except Exception as e:
            # Always show critical directory creation errors
            print(
                f"Failed to create log directory {self.log_dir}: {e}", file=sys.stderr
            )

            # Fallback to current directory
            self.log_dir = Path.cwd() / "logs"
            if self.debug_mode:
                print(f"Falling back to: {self.log_dir.absolute()}")

            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as fallback_error:
                print(
                    f"Fallback directory creation failed: {fallback_error}",
                    file=sys.stderr,
                )

    def _load_existing_log_hashes(self) -> set:
        """Load hashes of existing log entries to prevent duplicates."""
        cache = set()
        for log_file in self.log_files.values():
            self._process_log_file_for_hashes(log_file, cache)
        return cache

    def _process_log_file_for_hashes(self, log_file: Path, cache: set) -> None:
        """Helper to process a single log file and add entry hashes to cache."""
        if not log_file.exists():
            return
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Split by the separator and hash each complete log entry
                entries = content.split("=" * 80 + "\n")
                for entry in entries:
                    if entry.strip():
                        cache.add(hash("=" * 80 + "\n" + entry.strip() + "\n"))
            if self.debug_mode:
                print(f"Loaded {len(cache)} existing log entries from {log_file}")
        except Exception as e:
            if self.debug_mode:
                print(f"Warning: Could not read existing log file {log_file}: {e}")

    def _add_session_separator(self) -> None:
        """Add a session separator to indicate app restart."""
        if not self.preserve_logs:
            return

        separator = f"\n{'='*80}\nNEW SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n\n"

        for level, log_file in self.log_files.items():
            if log_file.exists() and log_file.stat().st_size > 0:
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(separator)
                        f.flush()  # Ensure immediate write

                    if self.debug_mode:
                        print(f"Session separator added to {log_file}")
                except Exception as e:
                    print(
                        f"Failed to add session separator to {log_file}: {e}",
                        file=sys.stderr,
                    )

    @staticmethod
    def _get_caller_info(_tb=None) -> tuple[str, str]:
        """
        Get information about the calling function and its parent.
        Skips logger frames to get actual caller.
        """
        stack = inspect.stack()
        # Skip logger frames
        caller_frame = next(
            (frame for frame in stack if frame.filename != __file__),
            stack[2] if len(stack) > 2 else None,
        )
        current_function = caller_frame.function if caller_frame else "Unknown"
        parent_function = stack[3].function if len(stack) > 3 else "Unknown"
        return current_function, parent_function

    def _format_message(
        self,
        level: LogLevel,
        message: str,
        error: Optional[Union[Exception, str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> str:
        """Format the complete log message with all metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_function, parent_function = self._get_caller_info()

        log_msg = [
            "=" * 80,
            f"TIMESTAMP: {timestamp}",
            f"LEVEL: {level.value}",
            f"FUNCTION: {current_function}",
            f"PARENT FUNCTION: {parent_function}",
            "-" * 80,
            f"MESSAGE: {message}",
        ]

        # Handle Exception logging
        if isinstance(error, BaseException):
            log_msg.extend(
                [
                    f"ERROR TYPE: {type(error).__name__}",
                    f"ERROR MESSAGE: {str(error)}",
                    "-" * 80,
                ]
            )
            if exc_info:
                try:
                    trace_lines = traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                    log_msg.extend(["FULL TRACEBACK:", "".join(trace_lines)])
                except Exception as format_err:
                    log_msg.append(f"Failed to format traceback: {str(format_err)}")

        elif isinstance(error, str):
            log_msg.extend(
                [
                    f"ERROR MESSAGE: {error}",
                    "-" * 80,
                ]
            )

        # Add default + custom context
        default_context = {
            "ml_engineer": "Muhammad Abiodun SULAIMAN",
            "environment": "production",
        }
        if additional_info:
            default_context.update(additional_info)

        log_msg.extend(
            [
                "-" * 80,
                "CONTEXT:",
                "\n".join(f"{k}: {v}" for k, v in default_context.items()),
                "=" * 80 + "\n",
            ]
        )

        return "\n".join(log_msg)

    def _write_log(self, level: LogLevel, message: str) -> None:
        """Write a message to the log file with duplicate prevention."""
        log_hash = hash(message)
        if log_hash in self._log_cache:
            if self.debug_mode:
                print(f"Duplicate log message detected for {level.value}, skipping")
            return

        try:
            # Ensure directory exists before writing
            self._ensure_log_directory()

            log_file = self.log_files[level]
            if self.debug_mode:
                print(f"Attempting to write to: {log_file.absolute()}")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message)
                f.flush()  # Force immediate write to disk

            self._log_cache.add(log_hash)

            if self.debug_mode:
                print(f"Successfully wrote {len(message)} characters to {log_file}")

        except Exception as e:
            # Always show write errors as they're important
            error_msg = f"Failed to write log to {self.log_files[level]}: {e}"
            print(error_msg, file=sys.stderr)

            # Try to write to stderr as fallback (only show first 200 chars to avoid spam)
            print(f"FALLBACK LOG [{level.value}]: {message[:200]}...", file=sys.stderr)

    def info(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an informational message."""
        log_message = self._format_message(
            LogLevel.INFO, message, additional_info=additional_info
        )
        self._write_log(LogLevel.INFO, log_message)

    def warning(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a warning message."""
        log_message = self._format_message(
            LogLevel.WARNING, message, additional_info=additional_info
        )
        self._write_log(LogLevel.WARNING, log_message)

    def error(
        self,
        error: Union[Exception, str],
        additional_info: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """
        Log an error with optional traceback.

        Args:
            error: Exception or string to log
            additional_info: Optional context dictionary
            exc_info: Whether to include full traceback if `error` is an Exception
        """
        log_message = self._format_message(
            LogLevel.ERROR, "An error occurred", error, additional_info, exc_info
        )
        self._write_log(LogLevel.ERROR, log_message)

    def clear_logs(self, level: Optional[LogLevel] = None, force: bool = False) -> None:
        """
        Clear log files for the specified level or all levels.

        Args:
            level: Specific log level to clear, or None for all levels
            force: Force clearing even if preserve_logs is True
        """
        if self.preserve_logs and not force:
            if self.debug_mode:
                print(
                    "Log clearing is disabled. Use force=True to override.",
                    file=sys.stderr,
                )
            return

        try:
            targets = [self.log_files[level]] if level else self.log_files.values()
            for log_file in targets:
                try:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("")
                    if self.debug_mode:
                        print(f"Cleared log file: {log_file}")
                    self._log_cache.clear()
                except FileNotFoundError:
                    if self.debug_mode:
                        print(f"Log file not found (already cleared): {log_file}")
        except Exception as e:
            print(f"Failed to clear logs: {str(e)}", file=sys.stderr)

    def rotate_logs(self, max_size_mb: int = 10) -> None:
        """Rotate log files if they exceed the specified size."""
        max_size_bytes = max_size_mb * 1024 * 1024

        for level, log_file in self.log_files.items():
            if log_file.exists() and log_file.stat().st_size > max_size_bytes:
                # Create backup with timestamp
                backup_name = (
                    f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
                backup_path = log_file.parent / backup_name

                try:
                    log_file.rename(backup_path)
                    if self.debug_mode:
                        print(f"Rotated log file: {log_file} -> {backup_path}")
                    # Clear cache since we rotated the file
                    self._log_cache.clear()
                except Exception as e:
                    print(f"Failed to rotate log {log_file}: {e}", file=sys.stderr)

    def enable_debug(self) -> None:
        """Enable debug output for troubleshooting."""
        self.debug_mode = True

    def disable_debug(self) -> None:
        """Disable debug output for clean console."""
        self.debug_mode = False

    def test_logging(self) -> None:
        """Test method to verify logging is working"""
        original_debug = self.debug_mode
        self.debug_mode = True  # Temporarily enable debug for testing

        print("Testing logging functionality...")

        self.info("This is a test info message", {"test": True})
        self.warning("This is a test warning message", {"test": True})
        self.error("This is a test error message", {"test": True})

        # Check if files were created
        for level, log_file in self.log_files.items():
            if log_file.exists():
                size = log_file.stat().st_size
                print(f"✅ {level.value} log file exists: {log_file} ({size} bytes)")
            else:
                print(f"❌ {level.value} log file NOT found: {log_file}")

        self.debug_mode = original_debug  # Restore original debug setting


# Initialize the logger with log preservation enabled and debug disabled by default
system_logger = Logger(preserve_logs=True, debug_mode=False)

# Add session separator to indicate new app session (silently)
system_logger._add_session_separator()

# Test the logger if run directly
if __name__ == "__main__":
    # Enable debug mode for direct testing
    system_logger.enable_debug()

    print("Testing the logger...")
    system_logger.test_logging()

    # Show current working directory
    print(f"Current working directory: {Path.cwd()}")

    # Try to create a simple test
    try:
        system_logger.info(
            "Logger test completed successfully",
            {"timestamp": datetime.now().isoformat()},
        )
        print("Logger test completed. Check the logs directory for output files.")
    except Exception as e:
        print(f"Logger test failed: {e}")

    # Disable debug mode after testing
    system_logger.disable_debug()
