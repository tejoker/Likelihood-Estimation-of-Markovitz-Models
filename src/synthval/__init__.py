"""synthval: validating synthetic financial data."""
import os as _os
from pathlib import Path as _Path

# Matplotlib needs a writable cache dir; ~/.config is read-only in sandboxed runs.
_os.environ.setdefault("MPLCONFIGDIR", str(_Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))

__version__ = "0.1.0"
