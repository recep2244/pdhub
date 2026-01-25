"""Custom exceptions for Protein Design Hub."""


class ProteinDesignHubError(Exception):
    """Base exception for Protein Design Hub."""

    pass


class PredictorNotFoundError(ProteinDesignHubError):
    """Raised when a predictor is not found in the registry."""

    def __init__(self, predictor_name: str):
        self.predictor_name = predictor_name
        super().__init__(f"Predictor not found: {predictor_name}")


class PredictorNotInstalledError(ProteinDesignHubError):
    """Raised when a predictor is not installed."""

    def __init__(self, predictor_name: str, install_hint: str = ""):
        self.predictor_name = predictor_name
        self.install_hint = install_hint
        message = f"Predictor not installed: {predictor_name}"
        if install_hint:
            message += f"\nTo install: {install_hint}"
        super().__init__(message)


class InputValidationError(ProteinDesignHubError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str = ""):
        self.field = field
        super().__init__(message)


class PredictionError(ProteinDesignHubError):
    """Raised when prediction fails."""

    def __init__(self, predictor_name: str, message: str, original_error: Exception = None):
        self.predictor_name = predictor_name
        self.original_error = original_error
        super().__init__(f"Prediction failed ({predictor_name}): {message}")


class EvaluationError(ProteinDesignHubError):
    """Raised when evaluation fails."""

    def __init__(self, metric: str, message: str, original_error: Exception = None):
        self.metric = metric
        self.original_error = original_error
        super().__init__(f"Evaluation failed ({metric}): {message}")


class InstallationError(ProteinDesignHubError):
    """Raised when installation fails."""

    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Installation failed ({tool_name}): {message}")


class ConfigurationError(ProteinDesignHubError):
    """Raised when configuration is invalid."""

    pass


class GPUNotAvailableError(ProteinDesignHubError):
    """Raised when GPU is required but not available."""

    def __init__(self, message: str = "GPU not available"):
        super().__init__(message)


class DependencyError(ProteinDesignHubError):
    """Raised when a required dependency is missing."""

    def __init__(self, dependency: str, install_hint: str = ""):
        self.dependency = dependency
        self.install_hint = install_hint
        message = f"Missing dependency: {dependency}"
        if install_hint:
            message += f"\nTo install: {install_hint}"
        super().__init__(message)
