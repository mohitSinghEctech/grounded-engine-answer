class AppError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int):
        super().__init__(message)

        self.error_code = error_code
        self.message = message
        self.status_code = status_code


class GatewayTimeout(AppError):
    def __init__(self, message: str = "The LLM gateway timed out."):
        super().__init__(error_code="GATEWAY_TIMEOUT", message=message, status_code=504)


class GatewayUnavailable(AppError):
    def __init__(self, message: str = "The LLM gateway is unavailable."):
        super().__init__(
            error_code="GATEWAY_UNAVAILABLE", message=message, status_code=503
        )


class InvalidGatewayResponse(AppError):
    def __init__(self, message: str = "The LLM gateway returned an invalid response."):
        super().__init__(
            error_code="INVALID_GATEWAY_RESPONSE", message=message, status_code=502
        )


class GatewayRateLimited(AppError):
    def __init__(
        self,
        message: str = "The LLM gateway rate limited the request.",
        retry_after_seconds: float | None = None,
    ):
        super().__init__(
            error_code="GATEWAY_RATE_LIMITED",
            message=message,
            status_code=429,
        )

        self.retry_after_seconds = retry_after_seconds


class RetrieverUnavailable(AppError):
    def __init__(self, message: str = "The document index is unavailable."):
        super().__init__(
            error_code="RETRIEVER_UNAVAILABLE", message=message, status_code=503
        )


class EmbeddingFailed(AppError):
    def __init__(self, message: str = "Could not embed the question."):
        super().__init__(
            error_code="EMBEDDING_FAILED", message=message, status_code=502
        )
