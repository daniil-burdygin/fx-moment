"""Снимок прогнозов TimesFM 3 по дневному ряду ЦБ — признаки обучаемого индикатора на замере.

Прогноз на дату публикации T строится только из курсов с pub_date ≤ T: контекст — последние
`CONTEXT_CAP` дней публикации до T включительно, горизонт `HORIZON` шагов. Снимок лежит рядом
с сырыми данными как производный (`data/derived/`) с метаданными: модель, ревизия весов,
устройство, снимок ЦБ, итог самопроверки. Модель импортируется лениво — `timesfm[torch]` живёт
в группе зависимостей `forecast`, основной путь её не грузит.

Веса 3.0 распространяются под некоммерческой лицензией Google: замер допустим, продукт — нет
(`docs/decisions/timesfm-experiment.md`)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from fxmoment.config import CORRIDORS
from fxmoment.data.store import RAW_CSV, load_meta, repo_root

MODEL_ID = "google/timesfm-3.0-pytorch"
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"  # ревизия весов на Hugging Face, 03.09.2026
CONTEXT_CAP = 1024  # дней публикации в контексте (≈ 4 года); в начале ряда — сколько есть
HORIZON = 20  # шагов прогноза = CALIBRATION_H, месячное решение клиента
FORECAST_START = "2017-01-01"  # год до ANALYSIS_START: обучение ML начинается с 2018
FORECAST_CURRENCIES: tuple[str, ...] = (*CORRIDORS, "USD")
STEPS: tuple[int, ...] = (1, 5, 10, 20)  # шаги, на которых хранятся медиана и полоса 10–90 %
MEDIAN_COL = 4  # индекс медианы среди девяти квантилей 0,1…0,9
FC_TAG = "_fc_"  # `<валюта>_fc_<признак>` — прогнозный столбец панели
SELF_CHECK_TOL_BPS = 0.1

# признаки для обучаемого индикатора (build_features); все в бп к курсу действия a_T
ML_FEATURES: tuple[str, ...] = ("mean5_bps", "mean20_bps", "min20_bps", "q10_h20_bps", "q90_h20_bps")
USD_FEATURE = "mean20_bps"  # от доллара берётся один признак: коридоры и так почти его функция

DERIVED_DIR = repo_root() / "data" / "derived"
FORECAST_CSV = DERIVED_DIR / "timesfm_daily.csv"
FORECAST_META = DERIVED_DIR / "timesfm_daily.meta.json"


def feature_names() -> tuple[str, ...]:
    names = ["mean5_bps", "mean20_bps", "min20_bps"]
    for s in STEPS:
        names += [f"med_h{s}_bps", f"q10_h{s}_bps", f"q90_h{s}_bps"]
    return tuple(names)


def features_from_quantiles(a_t: float, quantiles: np.ndarray) -> dict[str, float]:
    """Признаки одного прогноза: `quantiles` — (шаги ≥ HORIZON, 9), столбец MEDIAN_COL — медиана.
    Всё в бп к курсу действия a_T: положительное — модель ждёт курс выше сегодняшнего, то есть
    отправителю «сейчас выгодно»."""
    q = np.asarray(quantiles, dtype=float)[:HORIZON]
    if q.shape != (HORIZON, 9):
        raise ValueError(f"ожидались квантили формы ({HORIZON}, 9), получены {q.shape}")
    med = q[:, MEDIAN_COL]

    def rel(v: float) -> float:
        return float((v / a_t - 1) * 1e4)

    out = {"mean5_bps": rel(med[:5].mean()), "mean20_bps": rel(med.mean()), "min20_bps": rel(med.min())}
    for s in STEPS:
        out[f"med_h{s}_bps"] = rel(med[s - 1])
        out[f"q10_h{s}_bps"] = rel(q[s - 1, 0])
        out[f"q90_h{s}_bps"] = rel(q[s - 1, -1])
    return out


def column_name(currency: str, feature: str) -> str:
    return f"{currency}{FC_TAG}{feature}"


def is_forecast_column(col: object) -> bool:
    return FC_TAG in str(col)


def split_column(col: str) -> tuple[str, str]:
    """`TJS_fc_mean20_bps` → («TJS», «mean20_bps»)."""
    ccy, feat = str(col).split(FC_TAG, 1)
    return ccy, feat


# ------------------------------------------------------------------------------- модель


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str | None = None, batch_size: int = 64) -> Any:
    """Модель с закреплённой ревизией весов; сеть нужна только при первой загрузке в кэш HF."""
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    cfg = ModelConfig(
        checkpoint_path=MODEL_ID,
        revision=MODEL_REVISION,
        per_core_batch_size=batch_size,
        device=pick_device(device),
    )
    return TimesFM3Evaluator(cfg)


def _contexts(rate: pd.Series, dates: pd.DatetimeIndex, cap: int) -> list[np.ndarray]:
    """Контекст на дату T — курсы до T включительно и не длиннее cap: ничего после T."""
    values = rate.to_numpy(dtype=np.float32)
    pos = rate.index.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("дата прогноза отсутствует в ряду")
    return [values[max(0, p + 1 - cap) : p + 1] for p in pos]


def forecast_rows(model: Any, rate: pd.Series, dates: pd.DatetimeIndex, cap: int = CONTEXT_CAP) -> list[dict]:
    """Строки снимка для одной валюты и пачки дат. Настройки — умолчания официального
    зеро-шот прогона (symmetric averaging, квантили отсортированы)."""
    outs = model.predict_batch(_contexts(rate, dates, cap), horizon=HORIZON, return_quantiles=True)
    rows = []
    for t, out in zip(dates, outs, strict=True):
        feats = features_from_quantiles(float(rate.loc[t]), out.quantiles)
        rows.append({"currency": rate.name, "pub_date": t, **feats})
    return rows


def build_snapshot(
    panel: pd.DataFrame,
    model: Any,
    currencies: tuple[str, ...] = FORECAST_CURRENCIES,
    start: str = FORECAST_START,
    cap: int = CONTEXT_CAP,
    batch: int = 256,
    log: Any = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for ccy in currencies:
        rate = panel[ccy].dropna()
        dates = rate.index[rate.index >= pd.Timestamp(start)]
        for i in range(0, len(dates), batch):
            rows.extend(forecast_rows(model, rate, dates[i : i + batch], cap))
        if log:
            log(f"{ccy}: {len(dates)} дат, контекст до {cap}")
    return pd.DataFrame(rows).sort_values(["currency", "pub_date"]).reset_index(drop=True)


def self_check(
    model: Any, panel: pd.DataFrame, snapshot: pd.DataFrame, n: int = 8, seed: int = 0, cap: int = CONTEXT_CAP
) -> float:
    """Пересчёт случайных строк снимка поодиночке с контекстом `panel.loc[:T]`: наибольшее
    расхождение в бп. Проверяет сразу два свойства — в контексте нет ничего после T и результат
    не зависит от состава батча."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(snapshot), size=min(n, len(snapshot)), replace=False)
    worst = 0.0
    for i in idx:
        row = snapshot.iloc[int(i)]
        t = pd.Timestamp(row["pub_date"])
        rate = panel.loc[:t, str(row["currency"])].dropna()
        fresh = forecast_rows(model, rate, pd.DatetimeIndex([t]), cap)[0]
        worst = max(worst, max(abs(fresh[k] - float(row[k])) for k in feature_names()))
    return worst


# ------------------------------------------------------------------------------- снимок


def _sha256(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def save_snapshot(df: pd.DataFrame, extra: dict[str, Any]) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FORECAST_CSV, index=False, date_format="%Y-%m-%d", float_format="%.4f")
    cbr = load_meta()
    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": "TimesFM Non-Commercial License v1.0 — только замер, в продукт не идёт",
        "context_cap": CONTEXT_CAP,
        "horizon": HORIZON,
        "currencies": sorted(df["currency"].unique().tolist()),
        "rows": int(len(df)),
        "first_pub_date": str(pd.Timestamp(df["pub_date"].min()).date()),
        "last_pub_date": str(pd.Timestamp(df["pub_date"].max()).date()),
        "cbr_snapshot_sha256": _sha256(RAW_CSV),
        "cbr_fetched_at_utc": cbr.get("fetched_at_utc"),
        **extra,
    }
    FORECAST_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot() -> pd.DataFrame:
    if not FORECAST_CSV.exists():
        raise FileNotFoundError(f"нет снимка {FORECAST_CSV}: `uv run fxmoment fetch-forecast`")
    return pd.read_csv(FORECAST_CSV, parse_dates=["pub_date"])


def load_forecast_meta() -> dict:
    return json.loads(FORECAST_META.read_text(encoding="utf-8")) if FORECAST_META.exists() else {}


def attach_forecast(panel: pd.DataFrame, snapshot: pd.DataFrame | None = None) -> pd.DataFrame:
    """Панель с прогнозными столбцами `<валюта>_fc_<признак>` по дате публикации. Строка T несёт
    прогноз, построенный на T; пропуски не заполняются — день без прогноза для ML не существует."""
    snap = snapshot if snapshot is not None else load_snapshot()
    feats = [f for f in feature_names() if f in snap.columns]
    wide = snap.pivot(index="pub_date", columns="currency", values=feats)
    wide.columns = [column_name(str(ccy), str(feat)) for feat, ccy in wide.columns]
    wide.index = pd.DatetimeIndex(wide.index)
    return panel.join(wide, how="left")
