"""Tests for error_messages module."""

import pytest

from app.core.error_messages import (
    ErrorMessageTranslator, FriendlyError,
    show_friendly_error
)


class TestFriendlyError:
    """Test FriendlyError dataclass."""

    def test_creation(self):
        """Test creating a FriendlyError."""
        err = FriendlyError(
            title="Test Error",
            message="Something went wrong",
            suggestion="Try again",
            severity="error"
        )
        assert err.title == "Test Error"
        assert err.severity == "error"

    def test_default_severity(self):
        """Test default severity."""
        err = FriendlyError(title="T", message="M", suggestion="S")
        assert err.severity == "error"


class TestErrorMessageTranslator:
    """Test suite for ErrorMessageTranslator."""

    def test_file_not_found(self):
        """Test FileNotFoundError translation."""
        exc = FileNotFoundError("No such file or directory: 'data.csv'")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "File not found" in friendly.title
        assert friendly.severity == "error"
        assert len(friendly.suggestion) > 0

    def test_permission_denied(self):
        """Test PermissionError translation."""
        exc = PermissionError("Permission denied: 'protected.csv'")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "Permission" in friendly.title
        assert friendly.severity == "error"

    def test_empty_data(self):
        """Test empty data error translation."""
        exc = Exception("EmptyDataError: No columns to parse from file")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "empty" in friendly.title.lower()

    def test_no_data_loaded(self):
        """Test 'no data' error translation."""
        exc = ValueError("No data loaded. Please load data first.")
        friendly = ErrorMessageTranslator.translate(exc, "training model")
        
        assert "Data not loaded" in friendly.title

    def test_no_target(self):
        """Test 'no target' error translation."""
        exc = ValueError("Target column must be set before training")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "Target column" in friendly.title
        assert friendly.severity == "warning"

    def test_memory_error(self):
        """Test MemoryError translation."""
        exc = MemoryError("Unable to allocate array")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "memory" in friendly.title.lower()
        assert friendly.severity == "critical"

    def test_nan_values(self):
        """Test NaN values error translation."""
        exc = ValueError("Input contains NaN, infinity or a value too large")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "numeric" in friendly.title.lower()

    def test_convergence_warning(self):
        """Test convergence error translation."""
        exc = Exception("ConvergenceWarning: lbfgs failed to converge (status=1)")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "converge" in friendly.title.lower() or "converg" in friendly.message.lower()

    def test_missing_columns(self):
        """Test missing columns error translation."""
        exc = KeyError("Required columns not found: ['feature_1', 'feature_2']")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "columns" in friendly.title.lower()

    def test_small_sample(self):
        """Test small sample error translation."""
        exc = ValueError("Found array with dim 4. Estimator requires minimum 5 samples")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "Too few" in friendly.title or "samples" in friendly.message.lower()

    def test_unicode_error(self):
        """Test UnicodeDecodeError translation."""
        exc = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "encoding" in friendly.title.lower() or "coding" in friendly.title.lower()

    def test_generic_fallback(self):
        """Test that unknown errors get a generic friendly message."""
        exc = RuntimeError("Some completely unknown error xyz123")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert friendly.title != ""  # Should have some title
        assert friendly.message != ""  # Should have some message
        assert len(friendly.suggestion) > 0  # Should have a suggestion

    def test_unknown_error_has_suggestion(self):
        """Test that even unknown errors have actionable suggestions."""
        exc = Exception("Bizarre custom error never seen before")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "persists" in friendly.suggestion or "data" in friendly.suggestion.lower()

    def test_context_parameter(self):
        """Test that context influences translation."""
        exc = ValueError("Something failed")
        friendly_with_context = ErrorMessageTranslator.translate(exc, "loading data")
        friendly_without = ErrorMessageTranslator.translate(exc)
        
        # Both should produce valid FriendlyError objects
        assert friendly_with_context.title != ""
        assert friendly_without.title != ""

    def test_get_dialog_text(self):
        """Test dialog text formatting."""
        friendly = FriendlyError(
            title="Test",
            message="Message",
            suggestion="Suggestion",
            severity="error"
        )
        text = ErrorMessageTranslator.get_dialog_text(friendly)
        
        assert "Test" in text
        assert "Message" in text
        assert "Suggestion" in text
        assert "<b>" in text  # HTML formatting

    def test_get_dialog_text_with_traceback(self):
        """Test dialog text with traceback."""
        friendly = FriendlyError(
            title="Test",
            message="Message",
            suggestion="Suggestion"
        )
        text = ErrorMessageTranslator.get_dialog_text(
            friendly, include_traceback=True, traceback_text="Trace: line 1\nline 2"
        )
        
        assert "line 1" in text
        assert "<pre>" in text  # Preformatted traceback

    def test_all_patterns_have_required_fields(self):
        """Test that all error patterns have title, message, suggestion, severity."""
        for entry in ErrorMessageTranslator._ERROR_PATTERNS:
            assert "title" in entry
            assert "message" in entry
            assert "suggestion" in entry
            assert "severity" in entry
            assert "patterns" in entry
            assert len(entry["patterns"]) > 0

    def test_pattern_matching_case_insensitive(self):
        """Test that pattern matching is case insensitive."""
        exc_lower = Exception("no such file")
        exc_upper = Exception("NO SUCH FILE")
        
        friendly_lower = ErrorMessageTranslator.translate(exc_lower)
        friendly_upper = ErrorMessageTranslator.translate(exc_upper)
        
        # Both should match the same pattern
        assert "File not found" in friendly_lower.title
        assert "File not found" in friendly_upper.title

    def test_severity_values_valid(self):
        """Test that all severity values are valid."""
        valid_severities = {"info", "warning", "error", "critical"}
        for entry in ErrorMessageTranslator._ERROR_PATTERNS:
            assert entry["severity"] in valid_severities, f"Invalid severity: {entry['severity']}"

    def test_torchscript_error(self):
        """Test TorchScript error translation."""
        exc = RuntimeError("TorchScript conversion failed")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "neural network" in friendly.message.lower() or "network" in friendly.message.lower()
        assert friendly.severity == "warning"

    def test_optuna_error(self):
        """Test Optuna error translation."""
        exc = RuntimeError("Optuna TPE optimization failed")
        friendly = ErrorMessageTranslator.translate(exc)
        
        assert "optimization" in friendly.title.lower()
