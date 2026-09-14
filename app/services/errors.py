class PravahaServiceError(RuntimeError):
    status_code = 503


class DataStateServiceError(PravahaServiceError):
    pass


class MLIntelligenceServiceError(PravahaServiceError):
    pass

