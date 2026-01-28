# Contributing to Protein Design Hub

Thank you for your interest in contributing to Protein Design Hub! We welcome contributions from the community.

## Getting Started

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/pdhub.git
    cd pdhub
    ```
3.  **Set up the environment**:
    ```bash
    ./scripts/setup_environment.sh
    conda activate protein_design_hub
    ```
4.  **Create a branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```

## Development Workflow

-   **Code Style**: We use `black` for formatting and `ruff` for linting.
    ```bash
    black .
    ruff check .
    ```
-   **Type Checking**: We use `mypy` for static type checking.
    ```bash
    mypy src/protein_design_hub
    ```
-   **Testing**: Run tests using `pytest`.
    ```bash
    pytest
    ```

## Pull Request Process

1.  Ensure all tests ensure pass.
2.  Update documentation if you're changing functionality.
3.  Add your changes to the `CHANGELOG.md`.
4.  Submit a Pull Request targeting the `main` branch.

## Reporting Issues

Please use the GitHub Issue Tracker to report bugs or request features. Include:
-   Steps to reproduce the issue
-   Expected behavior
-   Actual behavior
-   Environment details (OS, Python version)

Thank you for helping improve Protein Design Hub!
