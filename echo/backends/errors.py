from __future__ import annotations


class BackendError(RuntimeError):
    def __init__(self, message: str, *, backend: str, model: str, detail: str = "") -> None:
        super().__init__(message)
        self.backend = backend
        self.model = model
        self.detail = detail


class BackendTimeoutError(BackendError):
    pass


class BackendUnreachableError(BackendError):
    pass


class BackendMalformedResponseError(BackendError):
    pass


class BackendModelMissingError(BackendError):
    pass
