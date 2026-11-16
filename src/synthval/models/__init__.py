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
#: Panel models: ONE univariate model trained on the pooled (per-series standardised) columns of a
#: panel; ``pipeline.fit_panel`` returns per-ticker views of it (``QuantGAN.series_models``).
_PANEL = {
    "QuantGAN": "synthval.models.neural:QuantGAN",
}
UNIVARIATE_MODELS = list(_UNIVARIATE)
MULTIVARIATE_MODELS = list(_MULTIVARIATE)
PANEL_MODELS = list(_PANEL)


def _load(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def get_model(name: str, **kwargs) -> ReturnModel:
    """Instantiate a univariate model by registry name."""
    return _load(_UNIVARIATE[name])(**kwargs)


def get_mv_model(name: str, **kwargs) -> MultiReturnModel:
    """Instantiate a multivariate model by registry name."""
    return _load(_MULTIVARIATE[name])(**kwargs)


def get_panel_model(name: str, **kwargs):
    """Instantiate a pooled panel model by registry name (``PANEL_MODELS``, e.g. QuantGAN)."""
    return _load(_PANEL[name])(**kwargs)


__all__ = ["ReturnModel", "MultiReturnModel", "get_model", "get_mv_model", "get_panel_model",
           "UNIVARIATE_MODELS", "MULTIVARIATE_MODELS", "PANEL_MODELS"]
