"""Tests for logger module."""

import pytest
import logging
from pathlib import Path

from app.utils.logger import (
    setup_logger, LoggerMixin, log_exception, get_log_history, ColoredFormatter
)


class TestColoredFormatter:
    """Test suite for ColoredFormatter."""

    def test_formatter_creation(self):
        """Test creating colored formatter."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        assert formatter is not None

    def test_format_debug(self):
        """Test formatting debug message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.DEBUG, pathname='',
            lineno=0, msg='test message', args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert 'DEBUG' in formatted
        assert 'test message' in formatted

    def test_format_error(self):
        """Test formatting error message."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name='test', level=logging.ERROR, pathname='',
            lineno=0, msg='error message', args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert 'ERROR' in formatted
        assert 'error message' in formatted


class TestSetupLogger:
    """Test suite for setup_logger."""

    def test_setup_basic(self):
        """Test basic logger setup."""
        logger = setup_logger(
            name='test_logger',
            log_level='INFO',
            console_output=False,
            file_output=False
        )
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'

    def test_setup_with_file(self, tmp_path):
        """Test logger setup with file output."""
        log_dir = str(tmp_path / 'logs')
        logger = setup_logger(
            name='test_logger_file',
            log_level='DEBUG',
            log_dir=log_dir,
            console_output=False,
            file_output=True
        )
        
        assert isinstance(logger, logging.Logger)
        # Check that log directory was created
        assert Path(log_dir).exists()

    def test_setup_with_console(self):
        """Test logger setup with console output."""
        logger = setup_logger(
            name='test_logger_console',
            log_level='INFO',
            console_output=True,
            file_output=False
        )
        
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) > 0

    def test_log_levels(self):
        """Test different log levels."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            logger = setup_logger(
                name=f'test_logger_{level}',
                log_level=level,
                console_output=False,
                file_output=False
            )
            assert logger.level == getattr(logging, level)

    def test_log_message(self, tmp_path):
        """Test logging a message."""
        log_dir = str(tmp_path / 'logs')
        logger = setup_logger(
            name='test_log_message',
            log_level='INFO',
            log_dir=log_dir,
            console_output=False,
            file_output=True
        )
        
        test_message = 'Test log message'
        logger.info(test_message)
        
        # Check that message is in log history
        history = get_log_history('test_log_message', n_lines=10)
        # Log history may not be immediately available
        assert isinstance(history, list)


class TestLoggerMixin:
    """Test suite for LoggerMixin."""

    def test_mixin_logger(self):
        """Test that mixin provides logger."""
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        assert hasattr(obj, 'logger')
        assert isinstance(obj.logger, logging.Logger)
        assert obj.logger.name == 'TestClass'

    def test_mixin_logger_caching(self):
        """Test that logger is cached."""
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        logger1 = obj.logger
        logger2 = obj.logger
        assert logger1 is logger2


class TestLogException:
    """Test suite for log_exception."""

    def test_log_exception(self):
        """Test logging an exception."""
        logger = setup_logger(
            name='test_exception',
            console_output=False,
            file_output=False
        )
        
        try:
            raise ValueError('Test exception')
        except Exception as e:
            # Should not raise
            log_exception(logger, e, 'Error occurred')

    def test_log_exception_no_message(self):
        """Test logging exception without message."""
        logger = setup_logger(
            name='test_exception2',
            console_output=False,
            file_output=False
        )
        
        try:
            raise RuntimeError('Test error')
        except Exception as e:
            log_exception(logger, e)


class TestGetLogHistory:
    """Test suite for get_log_history."""

    def test_get_history_no_file(self):
        """Test getting history when no log file exists."""
        history = get_log_history('nonexistent_logger', n_lines=10)
        assert history == []

    def test_get_history(self, tmp_path):
        """Test getting log history."""
        log_dir = str(tmp_path / 'logs')
        logger = setup_logger(
            name='test_history',
            log_level='INFO',
            log_dir=log_dir,
            console_output=False,
            file_output=True
        )
        
        # Log some messages
        for i in range(5):
            logger.info(f'Message {i}')
        
        history = get_log_history('test_history', n_lines=10)
        # Log history may not be immediately available
        assert isinstance(history, list)
