# Contributing to AutoAIM Studio

Thank you for your interest in contributing to AutoAIM Studio! This document provides guidelines for contributing to the project.

---

## 🐛 Reporting Bugs

If you find a bug, please create a GitHub Issue with the following information:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Step-by-step instructions to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Operating System (Windows/Linux/Mac)
   - Python version (`python --version`)
   - AutoAIM Studio version
6. **Screenshots**: If applicable, add screenshots

Use the bug report template when creating issues.

---

## 💡 Suggesting Features

We welcome feature suggestions! To propose a new feature:

1. Check existing issues to avoid duplicates
2. Create a GitHub Issue with the "Feature Request" label
3. Describe the feature and its use case
4. Explain why it would be useful

---

## 🔧 Contributing Code

### Setting Up Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/autoaim-studio.git
cd autoaim-studio

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests to verify setup
pytest
```

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Changes**
   - Follow the code style guidelines below
   - Write tests for new functionality
   - Update documentation if needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=app tests/
   
   # Run specific test file
   pytest tests/test_feature_engineering.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Code style changes (formatting)

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

---

## 📝 Code Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with these additions:

- **Line Length**: Maximum 100 characters
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints for function arguments and return values

### Formatting with Black

Format your code before committing:

```bash
# Format all Python files
black app/ tests/

# Check formatting without making changes
black --check app/ tests/
```

### Linting

```bash
# Run flake8
flake8 app/ tests/

# Run pylint
pylint app/
```

### Example Code Style

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'random_forest',
    n_estimators: int = 100
) -> Tuple[Any, Dict[str, float]]:
    """Train a machine learning model.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target values of shape (n_samples,)
        model_type: Type of model to train ('random_forest', 'xgboost', etc.)
        n_estimators: Number of estimators for ensemble methods
        
    Returns:
        Tuple of (trained_model, metrics_dict)
        
    Raises:
        ValueError: If model_type is not supported
    """
    # Implementation here
    pass
```

---

## 🧪 Testing Guidelines

### Test Structure

Tests are located in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_data_loading.py
├── test_feature_engineering.py
├── test_model_training.py
└── test_inference.py
```

### Writing Tests

Use pytest and follow these guidelines:

```python
import pytest
import numpy as np
from app.core.trainer import ModelTrainer

def test_trainer_initialization():
    """Test that ModelTrainer initializes correctly."""
    trainer = ModelTrainer()
    assert trainer is not None
    assert len(trainer.results) == 0

def test_model_training(sample_data):
    """Test training a simple model."""
    X, y = sample_data
    trainer = ModelTrainer()
    result = trainer.train('random_forest', X, y)
    
    assert result.model is not None
    assert 'rmse' in result.metrics
    assert result.metrics['r2'] > 0
```

### Fixtures

Add shared fixtures to `conftest.py`:

```python
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    return X, y

@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [1.5, 2.3, 3.1, 3.9, 4.7]
    })
```

---

## 📚 Documentation

- Update `USER_MANUAL.md` if adding new features
- Add docstrings to all public functions and classes
- Update `CHANGELOG.md` with your changes

---

## 🏷️ Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. Create a GitHub Release

---

## 📋 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment or discrimination of any kind
- Trolling, insulting/derogatory comments
- Personal or political attacks
- Publishing others' private information

---

## ❓ Questions?

- Check existing [GitHub Issues](https://github.com/yourusername/autoaim-studio/issues)
- Create a new issue with the "Question" label
- Join our discussions (coming soon)

---

Thank you for contributing to AutoAIM Studio! 🎉
