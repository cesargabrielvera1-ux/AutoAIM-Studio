"""
User-friendly error message translator for AutoAIM Studio.

Converts technical Python/scikit-learn/pandas error messages into plain,
understandable English messages for non-technical users.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FriendlyError:
    """A user-friendly error with actionable guidance."""
    title: str
    message: str
    suggestion: str
    severity: str = "error"  # "info", "warning", "error", "critical"


class ErrorMessageTranslator:
    """Translates technical errors into user-friendly messages."""

    # Pattern-based translations for common errors
    _ERROR_PATTERNS = [
        # --- Data loading errors ---
        {
            "patterns": ["No such file", "FileNotFoundError", "does not exist"],
            "title": "File not found",
            "message": "The file you are trying to load does not exist at that location.",
            "suggestion": "Verify the file path is correct. Use the 'Browse' button to select it graphically.",
            "severity": "error",
        },
        {
            "patterns": ["Permission denied", "PermissionError"],
            "title": "Permission denied",
            "message": "You do not have permission to access this file or directory.",
            "suggestion": "Try copying the file to your Documents or Desktop folder, or run AutoAIM as administrator.",
            "severity": "error",
        },
        {
            "patterns": ["EmptyDataError", "No columns to parse", "empty"],
            "title": "Empty file",
            "message": "The CSV/Excel file is empty or does not contain valid data.",
            "suggestion": "Open the file in Excel or a text editor and verify it has at least one header row and data.",
            "severity": "error",
        },
        {
            "patterns": ["UnicodeDecodeError", "codec can't decode"],
            "title": "Encoding problem",
            "message": "The file has special characters that could not be read correctly.",
            "suggestion": "Save the CSV file with UTF-8 encoding. In Excel: Save As -> CSV UTF-8.",
            "severity": "error",
        },
        {
            "patterns": ["ParserError", "Expected", "tokenizing"],
            "title": "Invalid file format",
            "message": "The file does not have a valid CSV/Excel format.",
            "suggestion": "Verify the file uses commas (',') or semicolons (';') as separator. If Excel, ensure it has no protected sheets.",
            "severity": "error",
        },
        # --- Data structure errors ---
        {
            "patterns": ["target column", "target must be set", "No target"],
            "title": "Target column not selected",
            "message": "You have not selected which column you want to predict.",
            "suggestion": "In the 'Data' tab, select the column you want to predict from the 'Target Column' dropdown menu.",
            "severity": "warning",
        },
        {
            "patterns": ["No data loaded", "data is None", "No data"],
            "title": "Data not loaded",
            "message": "No data is loaded in the application.",
            "suggestion": "Go to the 'Data' tab and load a CSV or Excel file with your data before training models.",
            "severity": "warning",
        },
        {
            "patterns": ["no model", "No model", "model is None"],
            "title": "Model not selected",
            "message": "You have not selected any model to train.",
            "suggestion": "In the 'Training' tab, select at least one model from the list. You can select multiple by holding Ctrl.",
            "severity": "warning",
        },
        {
            "patterns": ["n_samples", "Found array with", "min_samples"],
            "title": "Too few data samples",
            "message": "The dataset is too small to train the selected model.",
            "suggestion": "You need more samples (rows) in your dataset. Some models require at least 5-10 samples. Try a larger dataset.",
            "severity": "error",
        },
        # --- Feature/Column errors ---
        {
            "patterns": ["not in index", "feature_names", "columns"],
            "title": "Columns do not match",
            "message": "The prediction data columns do not match the training data columns.",
            "suggestion": "Ensure the new data has exactly the same columns as used for training. Check column names and order.",
            "severity": "error",
        },
        {
            "patterns": ["Required columns not found", "Invalid input"],
            "title": "Required columns missing",
            "message": "Columns needed for prediction are missing.",
            "suggestion": "Check the manifest.json file in the model bundle to see which columns are expected, and ensure your input data includes them.",
            "severity": "error",
        },
        # --- Memory errors ---
        {
            "patterns": ["MemoryError", "Unable to allocate", "out of memory", "Killed"],
            "title": "Insufficient memory",
            "message": "The system ran out of RAM during the operation.",
            "suggestion": "Try with less data, fewer simultaneous models, or close other applications. For large datasets (>100,000 rows), consider using only lightweight models like Ridge or SVR.",
            "severity": "critical",
        },
        # --- Numerical errors ---
        {
            "patterns": ["NaN", "inf", "infinity", "Infinite"],
            "title": "Problematic numeric values",
            "message": "Your data contains infinite or empty (NaN) values that prevent training.",
            "suggestion": "In the 'Data' tab, apply preprocessing: select 'mean' or 'median' in 'Imputation Strategy' and then click 'Apply Preprocessing'.",
            "severity": "error",
        },
        {
            "patterns": ["ConvergenceWarning", "did not converge", "convergence"],
            "title": "Model did not converge",
            "message": "The algorithm could not find a stable solution within the allowed number of iterations.",
            "suggestion": "Try increasing the number of iterations (max_iter), scale your data (apply preprocessing), or use a different model like Random Forest.",
            "severity": "warning",
        },
        # --- Value errors ---
        {
            "patterns": ["ValueError", "could not convert"],
            "title": "Incorrect data type",
            "message": "Some values in your data are not numeric where they should be.",
            "suggestion": "Check your CSV: look for cells with text in columns that should be numbers, or symbols like '-' or 'N/A' instead of empty values.",
            "severity": "error",
        },
        # --- Model training errors ---
        {
            "patterns": ["TorchScript", "torch.jit"],
            "title": "Error saving neural network model",
            "message": "Could not convert the neural network to optimized format.",
            "suggestion": "The model will be saved in standard format instead. This does not affect model quality, only file size.",
            "severity": "warning",
        },
        {
            "patterns": ["Optuna", "TPESampler", "optimization"],
            "title": "Optimization error",
            "message": "The hyperparameter optimization process encountered a problem.",
            "suggestion": "Verify that data is properly loaded and that the selected model is compatible with your problem type (regression vs classification).",
            "severity": "error",
        },
        # --- Generic fallback ---
    ]

    @classmethod
    def translate(cls, error: Exception, context: str = "") -> FriendlyError:
        """Translate an exception into a user-friendly message.
        
        Args:
            error: The exception to translate
            context: Optional context about where the error occurred
            
        Returns:
            FriendlyError with user-friendly title, message and suggestion
        """
        error_str = str(error)
        error_type = type(error).__name__
        full_text = f"{error_type}: {error_str}"

        # Try to match against known patterns
        for entry in cls._ERROR_PATTERNS:
            for pattern in entry["patterns"]:
                if pattern.lower() in full_text.lower() or pattern.lower() in context.lower():
                    return FriendlyError(
                        title=entry["title"],
                        message=entry["message"],
                        suggestion=entry["suggestion"],
                        severity=entry["severity"],
                    )

        # Fallback: return a generic but friendly message
        return FriendlyError(
            title=f"Error: {error_type}",
            message=f"A technical error occurred: {error_str[:200]}",
            suggestion="If the problem persists, check that your data is properly loaded and that you selected a valid model and target column.",
            severity="error",
        )

    @classmethod
    def get_dialog_text(cls, friendly: FriendlyError, include_traceback: bool = False,
                        traceback_text: str = "") -> str:
        """Format a FriendlyError as dialog text.
        
        Args:
            friendly: The friendly error
            include_traceback: Whether to include technical traceback
            traceback_text: The traceback text to include
            
        Returns:
            Formatted text for QMessageBox
        """
        text = f"<b>{friendly.title}</b><br><br>"
        text += f"{friendly.message}<br><br>"
        text += f"<b>Suggestion:</b><br>{friendly.suggestion}"
        
        if include_traceback and traceback_text:
            text += f"<br><br><small><b>Technical details (for support):</b><br>"
            text += f"<pre>{traceback_text[:1000]}</pre></small>"
        
        return text


def show_friendly_error(parent, error: Exception, context: str = "",
                       include_traceback: bool = False):
    """Show a user-friendly error dialog.
    
    This is a convenience function that translates an exception and shows
    it in a QMessageBox with an understandable message.
    
    Args:
        parent: Parent widget for the dialog
        error: The exception to display
        context: Context about where the error occurred
        include_traceback: Whether to show technical details
    """
    from PyQt6.QtWidgets import QMessageBox
    from PyQt6.QtCore import Qt
    import traceback
    
    friendly = ErrorMessageTranslator.translate(error, context)
    tb_text = traceback.format_exc() if include_traceback else ""
    dialog_text = ErrorMessageTranslator.get_dialog_text(friendly, include_traceback, tb_text)
    
    # Map severity to QMessageBox icon
    icon_map = {
        "info": QMessageBox.Icon.Information,
        "warning": QMessageBox.Icon.Warning,
        "error": QMessageBox.Icon.Critical,
        "critical": QMessageBox.Icon.Critical,
    }
    icon = icon_map.get(friendly.severity, QMessageBox.Icon.Critical)
    
    # Map severity to title
    title_map = {
        "info": "Information",
        "warning": "Warning",
        "error": "Error",
        "critical": "Critical Error",
    }
    title = title_map.get(friendly.severity, "Error")
    
    msg_box = QMessageBox(parent)
    msg_box.setIcon(icon)
    msg_box.setWindowTitle(title)
    msg_box.setTextFormat(Qt.TextFormat.RichText)
    msg_box.setText(dialog_text)
    
    # Add a "Show Details" button for technical info
    if include_traceback:
        msg_box.setDetailedText(tb_text)
    
    msg_box.exec()


# Qt import is done lazily inside show_friendly_error() to avoid
# PyQt6 dependency when only translating errors (e.g., in tests)
