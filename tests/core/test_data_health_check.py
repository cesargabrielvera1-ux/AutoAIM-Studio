"""Tests for data_health_check module."""

import pytest
import numpy as np
import pandas as pd

from app.core.data_health_check import (
    DataHealthChecker, HealthReport, HealthIssue,
    IssueSeverity, IssueCategory
)


class TestHealthIssue:
    """Test HealthIssue dataclass."""

    def test_creation(self):
        """Test creating a HealthIssue."""
        issue = HealthIssue(
            category=IssueCategory.MISSING_VALUES,
            severity=IssueSeverity.HIGH,
            title="Test Issue",
            description="A test issue",
            recommendation="Fix it"
        )
        assert issue.title == "Test Issue"
        assert issue.severity == IssueSeverity.HIGH
        assert issue.affected_columns == []


class TestHealthReport:
    """Test HealthReport dataclass."""

    def test_creation(self):
        """Test creating a HealthReport."""
        report = HealthReport(n_samples=100, n_features=5, overall_score=85.0)
        assert report.overall_score == 85.0
        assert report.critical_count == 0
        assert report.has_blocking_issues is False

    def test_has_blocking_issues_with_critical(self):
        """Test blocking issues detection."""
        report = HealthReport(
            n_samples=10, n_features=2, overall_score=30.0,
            issues=[HealthIssue(
                category=IssueCategory.SAMPLE_SIZE,
                severity=IssueSeverity.CRITICAL,
                title="Too small",
                description="Small",
                recommendation="Get more"
            )]
        )
        assert report.has_blocking_issues is True

    def test_get_issues_by_category(self):
        """Test filtering issues by category."""
        report = HealthReport(
            n_samples=100, n_features=5, overall_score=80.0,
            issues=[
                HealthIssue(IssueCategory.MISSING_VALUES, IssueSeverity.HIGH, "M1", "D", "R"),
                HealthIssue(IssueCategory.OUTLIERS, IssueSeverity.LOW, "O1", "D", "R"),
                HealthIssue(IssueCategory.MISSING_VALUES, IssueSeverity.MEDIUM, "M2", "D", "R"),
            ]
        )
        missing = report.get_issues_by_category(IssueCategory.MISSING_VALUES)
        assert len(missing) == 2

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        report = HealthReport(
            n_samples=100, n_features=5, overall_score=80.0,
            issues=[
                HealthIssue(IssueCategory.MISSING_VALUES, IssueSeverity.HIGH, "H1", "D", "R"),
                HealthIssue(IssueCategory.OUTLIERS, IssueSeverity.LOW, "L1", "D", "R"),
            ]
        )
        high = report.get_issues_by_severity(IssueSeverity.HIGH)
        assert len(high) == 1


class TestDataHealthChecker:
    """Test suite for DataHealthChecker."""

    def test_checker_initialization(self):
        """Test checker initialization."""
        checker = DataHealthChecker()
        assert checker is not None

    def test_perfect_data(self):
        """Test with clean, perfect data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_a': np.random.randn(200),
            'feature_b': np.random.randn(200),
            'feature_c': np.random.randn(200),
            'target': np.random.randn(200),
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        assert isinstance(report, HealthReport)
        assert report.overall_score >= 90  # Perfect data should score high
        assert not report.has_blocking_issues

    def test_missing_values_detection(self):
        """Test missing values detection."""
        df = pd.DataFrame({
            'col_a': [1.0, 2.0, np.nan, 4.0, 5.0],
            'col_b': [np.nan, np.nan, np.nan, np.nan, np.nan],  # 100% missing
            'target': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        missing_issues = report.get_issues_by_category(IssueCategory.MISSING_VALUES)
        assert len(missing_issues) > 0
        # Should detect col_b as critical (100% missing)
        critical_missing = [i for i in missing_issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical_missing) > 0

    def test_missing_values_medium(self):
        """Test medium-level missing values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'col_a': np.random.randn(100),
            'col_b': np.random.randn(100),
            'target': np.random.randn(100),
        })
        # Make 30% of col_b missing
        missing_idx = np.random.choice(100, 30, replace=False)
        df.loc[missing_idx, 'col_b'] = np.nan
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        missing_issues = report.get_issues_by_category(IssueCategory.MISSING_VALUES)
        assert len(missing_issues) > 0
        high_issues = [i for i in missing_issues if i.severity == IssueSeverity.HIGH]
        assert len(high_issues) > 0

    def test_duplicates_detection(self):
        """Test duplicate row detection."""
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],  # 50% duplicates
            'target': [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        dup_issues = report.get_issues_by_category(IssueCategory.DUPLICATES)
        assert len(dup_issues) > 0
        assert dup_issues[0].severity in (IssueSeverity.HIGH, IssueSeverity.MEDIUM)

    def test_constant_columns(self):
        """Test constant column detection."""
        df = pd.DataFrame({
            'varying': [1.0, 2.0, 3.0, 4.0, 5.0],
            'constant': [7.0, 7.0, 7.0, 7.0, 7.0],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        const_issues = report.get_issues_by_category(IssueCategory.CONSTANTS)
        assert len(const_issues) > 0
        assert 'constant' in const_issues[0].affected_columns

    def test_sample_size_critical(self):
        """Test critical sample size."""
        df = pd.DataFrame({
            'f1': [1.0, 2.0],  # Only 2 samples
            'f2': [3.0, 4.0],
            'target': [1.0, 2.0],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        size_issues = report.get_issues_by_category(IssueCategory.SAMPLE_SIZE)
        assert len(size_issues) > 0
        assert size_issues[0].severity == IssueSeverity.CRITICAL
        assert report.overall_score < 70  # Critical + correlation = low score

    def test_sample_size_ratio(self):
        """Test samples-to-features ratio."""
        # 50 samples, 20 features = ratio 2.5:1 (should warn)
        np.random.seed(42)
        data = {f'feature_{i}': np.random.randn(50) for i in range(20)}
        data['target'] = np.random.randn(50)
        df = pd.DataFrame(data)
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        size_issues = report.get_issues_by_category(IssueCategory.SAMPLE_SIZE)
        assert len(size_issues) > 0

    def test_outliers_detection(self):
        """Test outlier detection."""
        df = pd.DataFrame({
            'feature': list(np.random.randn(95)) + [100.0, -100.0, 150.0, -80.0, 200.0],
            'target': np.random.randn(100),
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        outlier_issues = report.get_issues_by_category(IssueCategory.OUTLIERS)
        assert len(outlier_issues) > 0

    def test_correlation_detection(self):
        """Test correlation detection."""
        base = np.random.randn(100)
        df = pd.DataFrame({
            'feature_a': base,
            'feature_b': base * 0.98 + np.random.randn(100) * 0.02,  # ~98% correlated
            'feature_c': np.random.randn(100),
            'target': np.random.randn(100),
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        corr_issues = report.get_issues_by_category(IssueCategory.CORRELATION)
        assert len(corr_issues) > 0

    def test_scaling_issue(self):
        """Test scaling detection."""
        df = pd.DataFrame({
            'tiny': np.random.randn(100) * 0.001,  # Range ~0.01
            'huge': np.random.randn(100) * 100000,  # Range ~1,000,000
            'target': np.random.randn(100),
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        scale_issues = report.get_issues_by_category(IssueCategory.SCALING)
        assert len(scale_issues) > 0

    def test_target_quality_missing(self):
        """Test target column with missing values."""
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [1.0, np.nan, 3.0, 4.0, np.nan],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        target_issues = report.get_issues_by_category(IssueCategory.TARGET_QUALITY)
        assert len(target_issues) > 0
        assert target_issues[0].severity == IssueSeverity.CRITICAL

    def test_target_constant(self):
        """Test constant target column."""
        df = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [7.0, 7.0, 7.0, 7.0, 7.0],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        target_issues = report.get_issues_by_category(IssueCategory.TARGET_QUALITY)
        assert len(target_issues) > 0
        assert any(i.severity == IssueSeverity.CRITICAL for i in target_issues)

    def test_class_imbalance(self):
        """Test class imbalance detection."""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': [0] * 90 + [1] * 10,  # 9:1 ratio
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        imbalance_issues = report.get_issues_by_category(IssueCategory.IMBALANCE)
        assert len(imbalance_issues) > 0
        assert imbalance_issues[0].severity == IssueSeverity.HIGH

    def test_no_target(self):
        """Test check without target column."""
        df = pd.DataFrame({
            'feature_a': np.random.randn(50),
            'feature_b': np.random.randn(50),
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column=None)
        
        assert isinstance(report, HealthReport)
        assert report.n_samples == 50
        # Should not have target-related issues
        target_issues = report.get_issues_by_category(IssueCategory.TARGET_QUALITY)
        assert len(target_issues) == 0

    def test_overall_score_calculation(self):
        """Test overall score is within [0, 100]."""
        # Very bad data
        df = pd.DataFrame({
            'f1': [np.nan] * 5,  # All missing
            'f2': [1.0] * 5,  # Constant
            'target': [1.0] * 5,  # Constant target
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        assert 0 <= report.overall_score <= 100
        assert report.overall_score < 70  # Low score due to critical issues

    def test_quick_summary(self):
        """Test quick summary generation."""
        checker = DataHealthChecker()
        
        good_report = HealthReport(n_samples=1000, n_features=10, overall_score=95.0, issues=[])
        assert "Healthy" in checker.get_quick_summary(good_report)
        
        bad_report = HealthReport(n_samples=5, n_features=20, overall_score=20.0, issues=[
            HealthIssue(IssueCategory.SAMPLE_SIZE, IssueSeverity.CRITICAL, "S", "D", "R")
        ])
        assert "poor" in checker.get_quick_summary(bad_report)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        checker = DataHealthChecker()
        report = checker.check(df)
        assert report.n_samples == 0
        assert report.overall_score <= 75  # Only 1 critical issue = -25 points

    def test_numeric_looking_strings(self):
        """Test detection of numeric values stored as strings."""
        df = pd.DataFrame({
            'numeric_str': ['1.5', '2.3', '3.7', '4.1', '5.0'],
            'real_numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        checker = DataHealthChecker()
        report = checker.check(df, target_column='target')
        
        type_issues = report.get_issues_by_category(IssueCategory.DATA_TYPE)
        assert len(type_issues) > 0
        assert 'numeric_str' in type_issues[0].affected_columns
