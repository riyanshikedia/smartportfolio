# Contributing to SmartPortfolio AI

Thank you for your interest in contributing to SmartPortfolio AI! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain a professional environment

---

## 💡 How to Contribute

### **Reporting Bugs**

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/SmartPortfolio/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version)
   - Error messages and logs

### **Suggesting Features**

1. Open an issue with the `enhancement` label
2. Describe the feature and its benefits
3. Provide examples or mockups if possible

### **Contributing Code**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

---

## 🛠️ Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/yourusername/SmartPortfolio.git
cd SmartPortfolio

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# 5. Create .env file
cp .env.example .env
# Edit .env with your database credentials
```

---

## 📝 Coding Standards

### **Python Style**

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [isort](https://pycqa.github.io/isort/) for import sorting

```bash
# Format code
black utils/ tests/

# Sort imports
isort utils/ tests/

# Lint
flake8 utils/ tests/
```

### **Docstrings**

Use Google-style docstrings:

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.045):
    """
    Calculate the Sharpe ratio for a return series.
    
    Args:
        returns (pd.Series): Daily returns
        risk_free_rate (float): Annual risk-free rate (default: 4.5%)
    
    Returns:
        float: Annualized Sharpe ratio
    
    Raises:
        ValueError: If returns is empty
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
    if returns.empty:
        raise ValueError("Returns cannot be empty")
    
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_returns / volatility
```

### **Naming Conventions**

- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private variables**: `_leading_underscore`

### **Type Hints**

Use type hints for function signatures:

```python
from typing import List, Dict, Optional
import pandas as pd

def screen_stocks(
    df: pd.DataFrame,
    min_score: float = 60.0,
    top_n: int = 50
) -> pd.DataFrame:
    """Screen stocks based on scoring criteria."""
    ...
```

---

## 🧪 Testing

### **Writing Tests**

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

```python
# tests/test_portfolio.py

def test_optimize_portfolio_weights_sum_to_one():
    """Test that optimized weights sum to 1.0"""
    weights = optimize_portfolio(returns, method='sharpe')
    assert abs(weights.sum() - 1.0) < 1e-6

def test_optimize_portfolio_respects_max_position():
    """Test that no position exceeds max_position constraint"""
    max_position = 0.10
    weights = optimize_portfolio(returns, max_position=max_position)
    assert all(weights <= max_position + 1e-6)
```

### **Running Tests**

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_portfolio.py -v

# Run with coverage
pytest tests/ --cov=utils --cov-report=html

# Run fast tests only
pytest tests/ -m "not slow"
```

### **Test Coverage**

- Aim for >80% code coverage
- Cover edge cases and error conditions
- Test both happy paths and failure modes

---

## 🔄 Pull Request Process

### **Before Submitting**

1. **Update documentation** - Update docstrings and README if needed
2. **Add tests** - Ensure tests pass and coverage is adequate
3. **Format code** - Run `black` and `isort`
4. **Lint code** - Fix all `flake8` warnings
5. **Update CHANGELOG** - Add entry for your changes

### **PR Checklist**

```markdown
- [ ] Tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] Linting passes (flake8)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts with main
```

### **PR Title Format**

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

**Examples:**
```
feat: Add support for sector-level constraints in optimization
fix: Correct Sharpe ratio calculation in backtest
docs: Add examples to portfolio optimization guide
refactor: Extract common data loading logic
test: Add unit tests for screening module
chore: Update dependencies to latest versions
```

### **PR Description Template**

```markdown
## Description
Brief description of the changes

## Motivation
Why are these changes needed?

## Changes Made
- Item 1
- Item 2
- Item 3

## Testing
How were these changes tested?

## Screenshots
If applicable, add screenshots

## Related Issues
Closes #123
```

### **Review Process**

1. **Automated checks** - GitHub Actions will run tests
2. **Code review** - Maintainers will review your code
3. **Feedback** - Address any requested changes
4. **Approval** - Once approved, your PR will be merged

---

## 🏗️ Project Structure

When adding new features, follow the project structure:

```
utils/
├── database_connector.py  # Database operations
├── data_helpers.py        # Data processing
├── ml_helpers.py          # ML utilities
└── visualization.py       # Plotting functions

notebooks/
├── 01_*.ipynb            # Data collection
├── 02_*.ipynb            # Screening
├── 03_*.ipynb            # ML predictions
├── 04_*.ipynb            # Optimization
├── 05_*.ipynb            # Simulation
└── 06_*.ipynb            # Dashboard

tests/
├── test_data_collection.py
├── test_screening.py
├── test_models.py
└── test_portfolio.py
```

---

## 📚 Additional Resources

- **Project Analysis**: [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md)
- **Structure Guide**: [STRUCTURE.md](STRUCTURE.md)
- **ML Approach**: [notebooks/README_ML_APPROACH.md](notebooks/README_ML_APPROACH.md)

---

## ❓ Questions?

- Open a [Discussion](https://github.com/yourusername/SmartPortfolio/discussions)
- Join our community chat
- Email: support@smartportfolio.ai

---

**Thank you for contributing to SmartPortfolio AI!** 🙏

