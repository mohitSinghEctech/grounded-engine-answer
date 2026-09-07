class AppError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int):
        super().__init__(message)

        self.error_code = error_code
        self.message = message
        self.status_code = status_code


class UpstreamTimeout(AppError):
    def __init__(self, message: str = "The upstream LLM service timed out."):
        super().__init__(
            error_code="UPSTREAM_TIMEOUT", message=message, status_code=504
        )


class UpstreamUnavailable(AppError):
    def __init__(self, message: str = "The upstream LLM service is unavailable."):
        super().__init__(
            error_code="UPSTREAM_UNAVAILABLE", message=message, status_code=503
        )


class InvalidUpstreamResponse(AppError):
    def __init__(
        self, message: str = "The upstream LLM service returned an invalid response."
    ):
        super().__init__(
            error_code="INVALID_UPSTREAM_RESPONSE", message=message, status_code=502
        )


class UpstreamRateLimited(AppError):
    def __init__(
        self,
        message: str = "The upstream LLM service rate limited the request.",
        retry_after_seconds: float | None = None,
    ):
        super().__init__(
            error_code="UPSTREAM_RATE_LIMITED",
            message=message,
            status_code=429,
        )

        self.retry_after_seconds = retry_after_seconds
