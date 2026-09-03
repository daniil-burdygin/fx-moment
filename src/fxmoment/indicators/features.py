"""Каузальные признаки для обучаемого индикатора: производные базовых индикаторов и контекст USD."""

from __future__ import annotations

import pandas as pd

from fxmoment.data.forecast import ML_FEATURES, USD_FEATURE, is_forecast_column, split_column
from fxmoment.indicators.base import down_streak, rolling_pct_rank, up_streak

RANK_WINDOWS = (20, 60, 120, 250)


CACHED_WINDOWS = (60, 120, 250)  # окна сеток `level` и `reversal`, в шагах дневного ряда


def enrich_context(rate: pd.Series, context: pd.DataFrame | None, scale: int = 1) -> pd.DataFrame:
    """Добавляет в контекст предвычисленные каузальные ряды (`_rank_w`, `_dsm_w`) для скорости.

    Значения не зависят от будущего, поэтому предвычисление на полном ряду и на срезе совпадают.
    `scale` — масштаб шага ряда (ADR-0010): кэш обязан совпасть с окнами масштабированной сетки,
    иначе индикатор молча пересчитает их заново и прогон замедлится в разы."""
    from fxmoment.indicators.base import _scale_step, rolling_days_since_min

    ctx = context.copy() if context is not None else pd.DataFrame(index=rate.index)
    ctx = ctx.reindex(rate.index)
    for w0 in CACHED_WINDOWS:
        w = _scale_step(w0, scale)
        ctx[f"_rank_{w}"] = rolling_pct_rank(rate, w)
        ctx[f"_dsm_{w}"] = rolling_days_since_min(rate, w)
    # прогнозные столбцы снимка TimesFM (замер): свой коридор → `_fc_<признак>`, доллар → один признак.
    # Чужие коридоры в признаки не идут: пять коридоров — почти один фактор, и это был бы тот же USD.
    for col in [c for c in ctx.columns if is_forecast_column(c)]:
        ccy, feat = split_column(str(col))
        if ccy == rate.name:
            ctx[f"_fc_{feat}"] = ctx[col]
        elif ccy == "USD" and feat == USD_FEATURE:
            ctx[f"_usd_fc_{feat}"] = ctx[col]
    return ctx


def build_features(rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
    f = pd.DataFrame(index=rate.index)
    ret1 = rate.pct_change()
    f["ret1"] = ret1
    f["ret5"] = rate.pct_change(5)
    f["ret20"] = rate.pct_change(20)
    f["down_streak"] = down_streak(rate).astype(float)
    f["up_streak"] = up_streak(rate).astype(float)
    for w in RANK_WINDOWS:
        key = f"_rank_{w}"
        if context is not None and key in context.columns:
            f[f"rank{w}"] = context[key].reindex(rate.index)
        else:
            f[f"rank{w}"] = rolling_pct_rank(rate, w)
    f["dist_min60"] = rate / rate.rolling(60).min() - 1
    f["dist_max60"] = rate / rate.rolling(60).max() - 1
    f["vol20"] = ret1.rolling(20).std()
    f["vol_rank250"] = f["vol20"].rolling(250).apply(lambda w: float((w <= w[-1]).mean()), raw=True)
    f["month"] = rate.index.month.astype(float)
    f["dow"] = rate.index.dayofweek.astype(float)
    if context is not None and "USD" in context.columns:
        usd = context["USD"].reindex(rate.index)
        f["usd_ret1"] = usd.pct_change()
        f["usd_ret5"] = usd.pct_change(5)
        f["usd_rank60"] = rolling_pct_rank(usd, 60)
        f["usd_vol20"] = usd.pct_change().rolling(20).std()
        local = rate / usd
        f["local_ret5"] = local.pct_change(5)
        f["local_rank60"] = rolling_pct_rank(local, 60)
    if context is not None:
        # прогноз TimesFM на дату T (снимок `data/derived/`, замер): в бп к курсу действия
        for feat in ML_FEATURES:
            key = f"_fc_{feat}"
            if key in context.columns:
                f[f"fc_{feat.removesuffix('_bps')}"] = context[key].reindex(rate.index)
        key = f"_usd_fc_{USD_FEATURE}"
        if key in context.columns:
            f[f"usd_fc_{USD_FEATURE.removesuffix('_bps')}"] = context[key].reindex(rate.index)
    return f
