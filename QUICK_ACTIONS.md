# Quick Action Checklist for Protein Design Hub

## 🚨 Critical (Do First)

- [ ] **Add LICENSE file**
  ```bash
  # Create MIT License
  cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Protein Design Hub Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
  ```

- [ ] **Review and commit untracked files**
  ```bash
  # Review what's untracked
  git status
  
  # Add the files you want to keep
  git add src/protein_design_hub/app.py
  git add src/protein_design_hub/biophysics/
  git add src/protein_design_hub/evolution/
  git add src/protein_design_hub/msa/
  git add src/protein_design_hub/design/esmif/
  git add src/protein_design_hub/evaluation/metrics/disorder.py
  git add src/protein_design_hub/evaluation/metrics/sequence_recovery.py
  git add src/protein_design_hub/evaluation/metrics/shape_complementarity.py
  git add src/protein_design_hub/web/pages/4_evolution.py
  git add src/protein_design_hub/web/pages/6_settings.py
  git add src/protein_design_hub/web/pages/7_msa.py
  git add src/protein_design_hub/web/pages/8_mpnn.py
  git add src/protein_design_hub/web/pages/9_jobs.py
  
  # Commit modified files
  git add -u
  
  # Make a commit
  git commit -m "Add new features: evolution, MSA, biophysics modules and updated web pages"
  ```

- [ ] **Create GitHub Actions CI pipeline**
  ```bash
  mkdir -p .github/workflows
  ```
  Create `.github/workflows/ci.yml` (see below)

- [ ] **Add CONTRIBUTING.md**
  Create contribution guidelines

- [ ] **Install dev dependencies**
  ```bash
  conda activate protein_design_hub
  pip install -e ".[dev]"
  ```

## 📋 GitHub Actions CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: |
        ruff check .
    
    - name: Format check with black
      run: |
        black --check .
    
    - name: Type check with mypy
      run: |
        mypy src/protein_design_hub --ignore-missing-imports
      continue-on-error: true
    
    - name: Test with pytest
      run: |
        pytest -v --cov=protein_design_hub --cov-report=xml --cov-report=term
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      if: matrix.python-version == '3.10'
      with:
        file: ./coverage.xml
        fail_ci_if_error: false

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit security check
      run: |
        pip install bandit
        bandit -r src/protein_design_hub -f json -o bandit-report.json
      continue-on-error: true
```

## 📝 Documentation Files to Create

- [ ] **CONTRIBUTING.md**
  - Code style guide
  - Development setup
  - PR process
  - Testing requirements

- [ ] **CHANGELOG.md**
  ```markdown
  # Changelog
  
  All notable changes to this project will be documented in this file.
  
  The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
  and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
  
  ## [Unreleased]
  
  ### Added
  - Evolution module
  - MSA analysis tools
  - Biophysics metrics
  - ESM-IF designer
  - New evaluation metrics (disorder, sequence recovery, shape complementarity)
  
  ### Changed
  - Reorganized web pages
  - Updated web UI
  
  ## [0.1.0] - 2026-01-28
  
  ### Added
  - Initial release
  - ColabFold, Chai-1, Boltz-2 predictors
  - Comprehensive evaluation metrics
  - Streamlit web UI
  - CLI interface
  ```

- [ ] **CODE_OF_CONDUCT.md**
  Use Contributor Covenant

- [ ] **SECURITY.md**
  ```markdown
  # Security Policy
  
  ## Reporting a Vulnerability
  
  If you discover a security vulnerability, please email [your-email@example.com]
  instead of using the issue tracker.
  
  We will respond within 48 hours with next steps.
  ```

## 🧪 Testing Improvements

- [ ] **Create test fixtures directory**
  ```bash
  mkdir -p tests/fixtures
  mkdir -p tests/unit
  mkdir -p tests/integration
  ```

- [ ] **Add sample data**
  Download small PDB files for testing

- [ ] **Write predictor tests**
  - Mock predictor responses
  - Test error handling
  - Test validation

- [ ] **Write CLI tests**
  - Test all commands
  - Test error cases
  - Test help output

- [ ] **Add integration tests**
  - End-to-end workflows
  - Multi-predictor comparisons

## 🐳 Docker Support

- [ ] **Create Dockerfile**
  ```dockerfile
  FROM python:3.10-slim
  
  WORKDIR /app
  
  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      build-essential \
      && rm -rf /var/lib/apt/lists/*
  
  # Copy requirements
  COPY pyproject.toml .
  COPY README.md .
  
  # Install Python dependencies
  RUN pip install --no-cache-dir -e .
  
  # Copy source
  COPY src/ src/
  COPY config/ config/
  
  # Expose Streamlit port
  EXPOSE 8501
  
  CMD ["pdhub", "web", "--host", "0.0.0.0"]
  ```

- [ ] **Create docker-compose.yml**
  ```yaml
  version: '3.8'
  
  services:
    pdhub:
      build: .
      ports:
        - "8501:8501"
      volumes:
        - ./outputs:/app/outputs
        - ./config:/app/config
      environment:
        - PDHUB_OUTPUT_BASE_DIR=/app/outputs
  ```

## 📖 Examples to Create

- [ ] **Create examples directory**
  ```bash
  mkdir -p examples/{notebooks,scripts}
  ```

- [ ] **Example notebooks**
  - `01_basic_prediction.ipynb`
  - `02_structure_evaluation.ipynb`
  - `03_comparing_predictors.ipynb`
  - `04_batch_processing.ipynb`

- [ ] **Example scripts**
  - `predict_single.py`
  - `evaluate_structures.py`
  - `batch_predict.py`

## 🔧 Pre-commit Hooks

- [ ] **Create .pre-commit-config.yaml**
  ```yaml
  repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.5.0
      hooks:
        - id: trailing-whitespace
        - id: end-of-file-fixer
        - id: check-yaml
        - id: check-added-large-files
    
    - repo: https://github.com/psf/black
      rev: 23.12.1
      hooks:
        - id: black
    
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.1.9
      hooks:
        - id: ruff
          args: [--fix, --exit-non-zero-on-fix]
    
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.8.0
      hooks:
        - id: mypy
          args: [--ignore-missing-imports]
  ```

- [ ] **Install pre-commit**
  ```bash
  pip install pre-commit
  pre-commit install
  ```

## 📊 Monitoring Setup

- [ ] **Add logging configuration**
  Create `src/protein_design_hub/core/logging_config.py`

- [ ] **Add metrics collection**
  - Prediction time
  - Memory usage
  - Success/failure rates

## 🎯 Quick Wins (< 1 hour each)

1. [ ] Add LICENSE file (5 min)
2. [ ] Create CHANGELOG.md (10 min)
3. [ ] Add .pre-commit-config.yaml (15 min)
4. [ ] Create CONTRIBUTING.md template (20 min)
5. [ ] Add GitHub issue templates (30 min)
6. [ ] Create basic Dockerfile (30 min)
7. [ ] Add CODE_OF_CONDUCT.md (10 min)
8. [ ] Create SECURITY.md (10 min)
9. [ ] Add badges to README (15 min)
10. [ ] Set up GitHub Actions CI (45 min)

## 📈 Success Metrics

Track these metrics monthly:

- [ ] Test coverage percentage
- [ ] Documentation pages count
- [ ] Open/closed issues ratio
- [ ] PR merge time
- [ ] Number of contributors
- [ ] Download/install statistics

## 🔄 Maintenance Tasks

Weekly:
- [ ] Review open issues
- [ ] Merge approved PRs
- [ ] Update dependencies
- [ ] Check CI/CD status

Monthly:
- [ ] Update CHANGELOG
- [ ] Review roadmap
- [ ] Update documentation
- [ ] Security audit

Quarterly:
- [ ] Major version planning
- [ ] User survey
- [ ] Performance audit
- [ ] Dependencies cleanup

---

## Next Steps

1. Start with the **Critical** section
2. Run the commands in order
3. Test everything locally before pushing
4. Create PR for review
5. Iterate based on feedback

Good luck! 🚀
