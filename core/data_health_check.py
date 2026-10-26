"""
Data Health Check module for AutoAIM Studio.

Analyzes datasets for common quality issues and provides actionable
recommendations before model training begins.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats


class IssueSeverity(Enum):
    """Severity levels for data quality issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of data quality issues."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    IMBALANCE = "imbalance"
    CORRELATION = "correlation"
    SCALING = "scaling"
    DUPLICATES = "duplicates"
    CONSTANTS = "constants"
    DATA_TYPE = "data_type"
    SAMPLE_SIZE = "sample_size"
    TARGET_QUALITY = "target_quality"


@dataclass
class HealthIssue:
    """A single data quality issue."""
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    recommendation: str
    affected_columns: List[str] = field(default_factory=list)
    metric_value: Optional[float] = None
    metric_name: Optional[str] = None


@dataclass
class HealthReport:
    """Complete health report for a dataset."""
    n_samples: int
    n_features: int
    overall_score: float  # 0-100
    issues: List[HealthIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)
    
    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.MEDIUM)
    
    @property
    def has_blocking_issues(self) -> bool:
        """Returns True if there are issues that prevent training."""
        return any(i.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH) 
                   for i in self.issues)
    
    def get_issues_by_category(self, category: IssueCategory) -> List[HealthIssue]:
        return [i for i in self.issues if i.category == category]
    
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[HealthIssue]:
        return [i for i in self.issues if i.severity == severity]


class DataHealthChecker:
    """Analyzes dataset quality and produces a health report."""

    # Thresholds for issue detection
    MISSING_HIGH_THRESHOLD = 0.50  # 50% missing
    MISSING_MEDIUM_THRESHOLD = 0.20  # 20% missing
    MISSING_LOW_THRESHOLD = 0.05  # 5% missing
    
    OUTLIER_ZSCORE = 3.0  # Z-score for outlier detection
    OUTLIER_IQR_FACTOR = 1.5  # IQR factor
    OUTLIER_HIGH_THRESHOLD = 0.10  # 10% outliers is high
    OUTLIER_MEDIUM_THRESHOLD = 0.05  # 5% outliers is medium
    
    DUPLICATE_HIGH_THRESHOLD = 0.20  # 20% duplicates
    DUPLICATE_MEDIUM_THRESHOLD = 0.10  # 10% duplicates
    
    CONSTANT_THRESHOLD = 0  # Any constant column is an issue
    
    CORRELATION_HIGH_THRESHOLD = 0.95  # Features correlated > 95%
    CORRELATION_MEDIUM_THRESHOLD = 0.90  # Features correlated > 90%
    
    MIN_SAMPLES_PER_FEATURE = 10  # Minimum samples per feature
    MIN_SAMPLES_ABSOLUTE = 20  # Absolute minimum samples
    MIN_FEATURES = 1  # Minimum number of features
    
    CLASS_IMBALANCE_RATIO = 0.2  # Minority class < 20% of majority
    
    SCALE_RATIO_THRESHOLD = 1000  # Features with scale ratio > 1000

    def check(self, df: pd.DataFrame, target_column: Optional[str] = None) -> HealthReport:
        """Run all health checks on a dataset.
        
        Args:
            df: The DataFrame to analyze
            target_column: Name of the target column (if set)
            
        Returns:
            HealthReport with all detected issues and recommendations
        """
        issues = []
        summary = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "target_column": target_column,
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        # Run all check categories
        issues.extend(self._check_missing_values(df))
        issues.extend(self._check_duplicates(df))
        issues.extend(self._check_constant_columns(df))
        issues.extend(self._check_sample_size(df))
        issues.extend(self._check_scaling(df))
        issues.extend(self._check_outliers(df, target_column))
        issues.extend(self._check_correlations(df, target_column))
        issues.extend(self._check_data_types(df))
        
        if target_column and target_column in df.columns:
            issues.extend(self._check_target_quality(df, target_column))
            issues.extend(self._check_class_balance(df, target_column))
        
        # Calculate overall score (0-100)
        score = self._calculate_overall_score(issues, len(df), len(df.columns))
        summary["overall_score"] = score
        
        return HealthReport(
            n_samples=len(df),
            n_features=len(df.columns),
            overall_score=score,
            issues=issues,
            summary=summary,
        )
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[HealthIssue]:
        """Check for missing values in each column."""
        issues = []
        missing_counts = df.isnull().sum()
        missing_pct = missing_counts / len(df)
        
        n_cells = df.shape[0] * df.shape[1]
        total_missing_pct = df.isnull().sum().sum() / n_cells if n_cells > 0 else 0.0
        
        # Columns with significant missing values
        high_missing = missing_pct[missing_pct >= self.MISSING_HIGH_THRESHOLD]
        medium_missing = missing_pct[
            (missing_pct >= self.MISSING_MEDIUM_THRESHOLD) & 
            (missing_pct < self.MISSING_HIGH_THRESHOLD)
        ]
        low_missing = missing_pct[
            (missing_pct >= self.MISSING_LOW_THRESHOLD) & 
            (missing_pct < self.MISSING_MEDIUM_THRESHOLD)
        ]
        
        if not high_missing.empty:
            issues.append(HealthIssue(
                category=IssueCategory.MISSING_VALUES,
                severity=IssueSeverity.CRITICAL,
                title="Columns with many missing values",
                description=f"{len(high_missing)} column(s) have more than 50% missing values: "
                           f"{', '.join(high_missing.index.tolist())}.",
                recommendation="Consider removing these columns or using advanced imputation (KNN). "
                              "If they are essential, you need to obtain the missing data.",
                affected_columns=high_missing.index.tolist(),
                metric_value=float(high_missing.mean()),
                metric_name="missing_ratio",
            ))
        
        if not medium_missing.empty:
            issues.append(HealthIssue(
                category=IssueCategory.MISSING_VALUES,
                severity=IssueSeverity.HIGH,
                title="Columns with moderate missing values",
                description=f"{len(medium_missing)} column(s) have between 20% and 50% missing values: "
                           f"{', '.join(medium_missing.index.tolist())}.",
                recommendation="Use imputation in the 'Data' tab: select 'knn' for better accuracy "
                              "or 'median'/'mean' for a quick solution, then click 'Apply Preprocessing'.",
                affected_columns=medium_missing.index.tolist(),
                metric_value=float(medium_missing.mean()),
                metric_name="missing_ratio",
            ))
        
        if not low_missing.empty:
            issues.append(HealthIssue(
                category=IssueCategory.MISSING_VALUES,
                severity=IssueSeverity.LOW,
                title="Minor missing values",
                description=f"{len(low_missing)} column(s) have between 5% and 20% missing values: "
                           f"{', '.join(low_missing.index.tolist())}.",
                recommendation="Impute missing values: go to 'Data' -> select an imputation strategy "
                              "(mean/median recommended) -> 'Apply Preprocessing'.",
                affected_columns=low_missing.index.tolist(),
                metric_value=float(low_missing.mean()),
                metric_name="missing_ratio",
            ))
        
        if total_missing_pct > 0 and high_missing.empty and medium_missing.empty and low_missing.empty:
            issues.append(HealthIssue(
                category=IssueCategory.MISSING_VALUES,
                severity=IssueSeverity.INFO,
                title="Missing values below 5%",
                description="Missing values exist but are below 5%. Automatic imputation will handle them.",
                recommendation="Apply preprocessing in the 'Data' tab to automatically impute.",
                metric_value=total_missing_pct,
                metric_name="total_missing_ratio",
            ))
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[HealthIssue]:
        """Check for duplicate rows."""
        issues = []
        n_duplicates = df.duplicated().sum()
        dup_ratio = n_duplicates / len(df) if len(df) > 0 else 0
        
        if dup_ratio >= self.DUPLICATE_HIGH_THRESHOLD:
            issues.append(HealthIssue(
                category=IssueCategory.DUPLICATES,
                severity=IssueSeverity.HIGH,
                title=f"Many duplicate rows ({n_duplicates})",
                description=f"{dup_ratio*100:.1f}% of rows are duplicated. This can cause overfitting.",
                recommendation="Remove duplicate rows before training. In pandas: df.drop_duplicates().",
                metric_value=dup_ratio,
                metric_name="duplicate_ratio",
            ))
        elif dup_ratio >= self.DUPLICATE_MEDIUM_THRESHOLD:
            issues.append(HealthIssue(
                category=IssueCategory.DUPLICATES,
                severity=IssueSeverity.MEDIUM,
                title=f"Duplicate rows detected ({n_duplicates})",
                description=f"{dup_ratio*100:.1f}% of rows are duplicated.",
                recommendation="Consider removing duplicates to avoid training bias.",
                metric_value=dup_ratio,
                metric_name="duplicate_ratio",
            ))
        
        return issues
    
    def _check_constant_columns(self, df: pd.DataFrame) -> List[HealthIssue]:
        """Check for columns with constant values."""
        issues = []
        constant_cols = [col for col in df.columns 
                        if df[col].nunique(dropna=False) <= 1]
        
        if constant_cols:
            issues.append(HealthIssue(
                category=IssueCategory.CONSTANTS,
                severity=IssueSeverity.MEDIUM,
                title=f"Constant columns ({len(constant_cols)})",
                description=f"The following columns have the same value in all rows: "
                           f"{', '.join(constant_cols)}. They provide no information to the model.",
                recommendation="Remove these columns from the dataset to reduce noise and speed up training.",
                affected_columns=constant_cols,
            ))
        
        return issues
    
    def _check_sample_size(self, df: pd.DataFrame) -> List[HealthIssue]:
        """Check if dataset has sufficient samples."""
        issues = []
        n_samples = len(df)
        n_features = len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
        
        if n_samples < self.MIN_SAMPLES_ABSOLUTE:
            issues.append(HealthIssue(
                category=IssueCategory.SAMPLE_SIZE,
                severity=IssueSeverity.CRITICAL,
                title=f"Very few samples ({n_samples})",
                description=f"The dataset only has {n_samples} rows. At least {self.MIN_SAMPLES_ABSOLUTE} are recommended.",
                recommendation="Get more data. With fewer than 20 samples, no model will give reliable results. "
                              "Try at least 50-100 samples for acceptable results.",
                metric_value=n_samples,
                metric_name="n_samples",
            ))
        elif n_samples < n_features * self.MIN_SAMPLES_PER_FEATURE:
            ratio = n_samples / n_features if n_features > 0 else 0
            issues.append(HealthIssue(
                category=IssueCategory.SAMPLE_SIZE,
                severity=IssueSeverity.HIGH,
                title="Low sample-to-feature ratio",
                description=f"You have {n_samples} samples and ~{n_features} numeric features "
                           f"(ratio {ratio:.1f}:1). At least 10:1 is recommended.",
                recommendation="Reduce the number of features (remove irrelevant columns) or "
                              "get more data. With few samples, use simple models like Ridge or SVR.",
                metric_value=ratio,
                metric_name="samples_per_feature",
            ))
        elif n_samples < 100:
            issues.append(HealthIssue(
                category=IssueCategory.SAMPLE_SIZE,
                severity=IssueSeverity.LOW,
                title=f"Small dataset ({n_samples} samples)",
                description="The dataset is small. Results may have high variance.",
                recommendation="Use cross-validation with more folds (10 instead of 5) to better estimate performance. "
                              "Consider robust models like Random Forest or Gradient Boosting.",
                metric_value=n_samples,
                metric_name="n_samples",
            ))
        
        return issues
    
    def _check_scaling(self, df: pd.DataFrame) -> List[HealthIssue]:
        """Check if features are on very different scales."""
        issues = []
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return issues
        
        # Compute ranges (max - min) for each column
        ranges = numeric_df.max() - numeric_df.min()
        ranges = ranges[ranges > 0]  # Exclude constant columns
        
        if len(ranges) < 2:
            return issues
        
        max_range = ranges.max()
        min_range = ranges.min()
        
        if max_range / min_range > self.SCALE_RATIO_THRESHOLD:
            issues.append(HealthIssue(
                category=IssueCategory.SCALING,
                severity=IssueSeverity.MEDIUM,
                title="Features on very different scales",
                description=f"Some features have very different value ranges "
                           f"(scale ratio: {max_range/min_range:.0f}:1). "
                           f"This affects SVM, KNN and neural networks.",
                recommendation="Apply scaling in the 'Data' tab: select 'standard' or 'robust' "
                              "in 'Scaling Strategy' and click 'Apply Preprocessing'. "
                              "Random Forest and Gradient Boosting do not need scaling.",
                metric_value=max_range / min_range if min_range > 0 else float('inf'),
                metric_name="scale_ratio",
            ))
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame, target_col: Optional[str]) -> List[HealthIssue]:
        """Check for outliers in numeric columns."""
        issues = []
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return issues
        
        # Exclude target column
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])
        
        # Count outliers per column using IQR method
        outlier_counts = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 4:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - self.OUTLIER_IQR_FACTOR * iqr
            upper = q3 + self.OUTLIER_IQR_FACTOR * iqr
            n_outliers = ((series < lower) | (series > upper)).sum()
            outlier_counts[col] = n_outliers / len(series)
        
        if not outlier_counts:
            return issues
        
        high_outlier_cols = [c for c, r in outlier_counts.items() 
                            if r >= self.OUTLIER_HIGH_THRESHOLD]
        med_outlier_cols = [c for c, r in outlier_counts.items() 
                           if self.OUTLIER_MEDIUM_THRESHOLD <= r < self.OUTLIER_HIGH_THRESHOLD]
        
        if high_outlier_cols:
            issues.append(HealthIssue(
                category=IssueCategory.OUTLIERS,
                severity=IssueSeverity.HIGH,
                title=f"Many outliers in {len(high_outlier_cols)} columns",
                description=f"Columns {', '.join(high_outlier_cols[:5])} have more than 10% outliers. "
                           f"Outliers can distort models.",
                recommendation="Consider using 'robust' scaling (outlier-resistant) or removing rows with extreme outliers. "
                              "For the target column, verify that values are correct and not measurement errors.",
                affected_columns=high_outlier_cols,
                metric_value=max(outlier_counts[c] for c in high_outlier_cols),
                metric_name="max_outlier_ratio",
            ))
        elif med_outlier_cols:
            issues.append(HealthIssue(
                category=IssueCategory.OUTLIERS,
                severity=IssueSeverity.LOW,
                title=f"Outliers detected in {len(med_outlier_cols)} columns",
                description=f"Some columns have between 5% and 10% outliers.",
                recommendation="This is normal in many datasets. If models underperform, "
                              "consider using 'robust' scaling or investigating if outliers are data errors.",
                affected_columns=med_outlier_cols,
                metric_value=max(outlier_counts[c] for c in med_outlier_cols),
                metric_name="max_outlier_ratio",
            ))
        
        return issues
    
    def _check_correlations(self, df: pd.DataFrame, target_col: Optional[str]) -> List[HealthIssue]:
        """Check for highly correlated features (multicollinearity)."""
        issues = []
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 3:
            return issues
        
        # Exclude target
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])
        
        if numeric_df.shape[1] < 2:
            return issues
        
        # Correlation matrix
        corr = numeric_df.corr().abs()
        
        # Find highly correlated pairs (upper triangle)
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                c = corr.iloc[i, j]
                if c >= self.CORRELATION_HIGH_THRESHOLD:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], c))
                elif c >= self.CORRELATION_MEDIUM_THRESHOLD:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], c))
        
        if not high_corr_pairs:
            return issues
        
        # Separate by severity
        critical_pairs = [(a, b, c) for a, b, c in high_corr_pairs if c >= self.CORRELATION_HIGH_THRESHOLD]
        medium_pairs = [(a, b, c) for a, b, c in high_corr_pairs 
                       if self.CORRELATION_MEDIUM_THRESHOLD <= c < self.CORRELATION_HIGH_THRESHOLD]
        
        if critical_pairs:
            pair_strs = [f"'{a}' - '{b}' ({c:.2f})" for a, b, c in critical_pairs[:5]]
            affected = list(set(a for a, _, _ in critical_pairs) | set(b for _, b, _ in critical_pairs))
            issues.append(HealthIssue(
                category=IssueCategory.CORRELATION,
                severity=IssueSeverity.HIGH,
                title=f"Strong multicollinearity ({len(critical_pairs)} pairs)",
                description=f"Some features are highly correlated (>95%): "
                           f"{', '.join(pair_strs)}. This causes instability in linear models.",
                recommendation="Remove one from each correlated pair (keep the most relevant) "
                              "or use Ridge Regression which handles multicollinearity well.",
                affected_columns=affected,
                metric_value=critical_pairs[0][2] if critical_pairs else 0,
                metric_name="max_correlation",
            ))
        elif medium_pairs:
            pair_strs = [f"'{a}' - '{b}' ({c:.2f})" for a, b, c in medium_pairs[:3]]
            affected = list(set(a for a, _, _ in medium_pairs) | set(b for _, b, _ in medium_pairs))
            issues.append(HealthIssue(
                category=IssueCategory.CORRELATION,
                severity=IssueSeverity.LOW,
                title=f"Moderate correlations detected ({len(medium_pairs)} pairs)",
                description=f"Some features have correlation >90%: {', '.join(pair_strs)}.",
                recommendation="Generally not a serious issue. For linear models, Ridge is more robust "
                              "than simple models against multicollinearity.",
                affected_columns=affected,
                metric_value=medium_pairs[0][2] if medium_pairs else 0,
                metric_name="max_correlation",
            ))
        
        return issues
    
    def _check_data_types(self, df: pd.DataFrame) -> List[HealthIssue]:
        """Check for problematic data types."""
        issues = []
        
        # Columns that are object/string but look numeric
        mixed_types = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to see if most values are numeric
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    numeric_count = pd.to_numeric(non_null, errors='coerce').notna().sum()
                    if numeric_count / len(non_null) > 0.8:
                        mixed_types.append(col)
        
        if mixed_types:
            issues.append(HealthIssue(
                category=IssueCategory.DATA_TYPE,
                severity=IssueSeverity.MEDIUM,
                title=f"Numeric columns in text format ({len(mixed_types)})",
                description=f"Columns {', '.join(mixed_types[:5])} appear numeric but are stored as text "
                           f"(likely due to decimal commas or symbols).",
                recommendation="Check the CSV: use periods (.) as decimal separator, not commas. "
                              "Remove symbols like '$', '%', or spaces from numeric cells.",
                affected_columns=mixed_types,
            ))
        
        return issues
    
    def _check_target_quality(self, df: pd.DataFrame, target_col: str) -> List[HealthIssue]:
        """Check the quality of the target column."""
        issues = []
        target = df[target_col]
        
        # Check for missing values in target
        target_missing = target.isnull().sum()
        if target_missing > 0:
            issues.append(HealthIssue(
                category=IssueCategory.TARGET_QUALITY,
                severity=IssueSeverity.CRITICAL,
                title=f"Missing values in target column ({target_missing})",
                description=f"The target column '{target_col}' has {target_missing} missing values. "
                           f"Cannot train without a target value for each sample.",
                recommendation="Remove rows with missing values in the target column, "
                              f"or impute values if you have a way to estimate them.",
                affected_columns=[target_col],
                metric_value=target_missing / len(df),
                metric_name="target_missing_ratio",
            ))
        
        # Check if target is constant
        if target.nunique(dropna=False) <= 1:
            issues.append(HealthIssue(
                category=IssueCategory.TARGET_QUALITY,
                severity=IssueSeverity.CRITICAL,
                title="Target column is constant",
                description=f"The target column '{target_col}' has the same value in all rows. "
                           f"There is nothing to predict.",
                recommendation="Verify you selected the correct column. "
                              "The target column must have varied values.",
                affected_columns=[target_col],
            ))
        
        return issues
    
    def _check_class_balance(self, df: pd.DataFrame, target_col: str) -> List[HealthIssue]:
        """Check class balance for classification problems."""
        issues = []
        target = df[target_col]
        
        # Only check if target looks like classification
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
            return issues  # Probably regression
        
        value_counts = target.value_counts()
        if len(value_counts) < 2:
            return issues
        
        majority = value_counts.iloc[0]
        minority = value_counts.iloc[-1]
        ratio = minority / majority
        
        if ratio < self.CLASS_IMBALANCE_RATIO:
            issues.append(HealthIssue(
                category=IssueCategory.IMBALANCE,
                severity=IssueSeverity.HIGH,
                title="Severe class imbalance",
                description=f"The most frequent class has {majority} samples, but the least frequent "
                           f"only has {minority} (ratio {ratio:.2f}). The model may ignore the minority class.",
                recommendation="Consider techniques like SMOTE (oversampling), undersampling the majority class, "
                              "or using 'class_weight=balanced' in supporting models (SVM, Random Forest).",
                metric_value=ratio,
                metric_name="class_balance_ratio",
            ))
        elif ratio < 0.5:
            issues.append(HealthIssue(
                category=IssueCategory.IMBALANCE,
                severity=IssueSeverity.LOW,
                title="Moderate class imbalance",
                description=f"There is moderate class imbalance (ratio {ratio:.2f}).",
                recommendation="Generally manageable. Monitor the F1-score metric in addition to accuracy "
                              "to ensure the model works well on all classes.",
                metric_value=ratio,
                metric_name="class_balance_ratio",
            ))
        
        return issues
    
    def _calculate_overall_score(self, issues: List[HealthIssue], 
                                  n_samples: int, n_features: int) -> float:
        """Calculate an overall health score from 0 to 100."""
        if n_samples == 0:
            return 0.0
        # Start at 100 and deduct points for issues
        score = 100.0
        
        deductions = {
            IssueSeverity.CRITICAL: 25,
            IssueSeverity.HIGH: 15,
            IssueSeverity.MEDIUM: 8,
            IssueSeverity.LOW: 3,
            IssueSeverity.INFO: 0,
        }
        
        for issue in issues:
            score -= deductions.get(issue.severity, 5)
        
        # Bonus for good sample size
        if n_samples >= 1000:
            score += 5
        elif n_samples >= 100:
            score += 2
        
        return max(0.0, min(100.0, score))
    
    def get_quick_summary(self, report: HealthReport) -> str:
        """Get a one-line summary of the health report."""
        if report.overall_score >= 90:
            return f"Healthy data (score: {report.overall_score:.0f}/100)"
        elif report.overall_score >= 70:
            return f"Acceptable data with {report.medium_count + report.high_count} warnings (score: {report.overall_score:.0f}/100)"
        elif report.overall_score >= 50:
            return f"Problems detected: {report.high_count} severe, {report.medium_count} moderate (score: {report.overall_score:.0f}/100)"
        else:
            return f"Data in poor state: {report.critical_count} critical, {report.high_count} severe (score: {report.overall_score:.0f}/100)"
