"""Model registries (lazy, so each model module can be imported independently)."""
from __future__ import annotations

import importlib

from .base import MultiReturnModel, ReturnModel

_UNIVARIATE = {
    "GBM": "synthval.models.univariate:GBM",
    "Merton": "synthval.models.univariate:Merton",
    "GARCHt": "synthval.models.univariate:GARCHt",
    "Heston": "synthval.models.univariate:Heston",
    "Bootstrap": "synthval.models.univariate:StationaryBootstrap",
}
_MULTIVARIATE = {
    "MVGaussian": "synthval.models.multivariate:MVGaussian",
    "MVStudentT": "synthval.models.multivariate:MVStudentT",
    "CCCGARCHt": "synthval.models.multivariate:CCCGARCHt",
    "MVBootstrap": "synthval.models.multivariate:MVStationaryBootstrap",
    "FactorGARCHt": "synthval.models.multivariate:FactorGARCHt",
}
UNIVARIATE_MODELS = list(_UNIVARIATE)
MULTIVARIATE_MODELS = list(_MULTIVARIATE)


def _load(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def get_model(name: str, **kwargs) -> ReturnModel:
    """Instantiate a univariate model by registry name."""
    return _load(_UNIVARIATE[name])(**kwargs)


def get_mv_model(name: str, **kwargs) -> MultiReturnModel:
    """Instantiate a multivariate model by registry name."""
    return _load(_MULTIVARIATE[name])(**kwargs)


__all__ = ["ReturnModel", "MultiReturnModel", "get_model", "get_mv_model",
           "UNIVARIATE_MODELS", "MULTIVARIATE_MODELS"]
