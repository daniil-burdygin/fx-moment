import numpy as np
import pandas as pd
import pytest


def make_panel(n_days: int = 1500, seed: int = 7, start: str = "2016-01-04") -> pd.DataFrame:
    """Синтетическая панель курсов: случайное блуждание с небольшим дрейфом, только рабочие дни."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    cols = {}
    usd = 60 * np.exp(np.cumsum(rng.normal(0, 0.007, n_days)))
    cols["USD"] = usd
    for name, base, vol in (("TJS", 8.0, 0.004), ("KZT", 0.16, 0.004), ("UZS", 0.0065, 0.003)):
        local = np.exp(np.cumsum(rng.normal(0, vol, n_days)))
        cols[name] = base * (usd / usd[0]) * local
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "pub_date"
    return panel


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
