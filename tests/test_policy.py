import pandas as pd

from fxmoment.combine import PolicyParams, apply_policy, rank_from_history


def _events(idx):
    return pd.DataFrame(
        [
            (idx[0], "TJS", "level", "BUY_NOW", 0.8),
            (idx[0], "TJS", "momentum", "BUY_NOW", 0.9),  # прореживается: ранг хуже
            (idx[1], "TJS", "level", "BUY_NOW", 0.7),  # охлаждение
            (idx[6], "TJS", "reversal", "WINDOW_CLOSING", 0.6),  # BUY_NOW был недавно → закрытие окна
            (idx[6], "TJS", "level", "BUY_NOW", 0.9),
        ],
        columns=["date", "corridor", "indicator", "scenario", "strength"],
    )


def test_policy_cooldown_and_conflict():
    idx = pd.bdate_range("2026-01-05", periods=20)
    out = apply_policy(_events(idx), {"level": 0, "reversal": 1, "momentum": 2}, idx, PolicyParams())
    d = dict(zip(zip(out["date"], out["indicator"], strict=True), out["decision"], strict=True))
    assert d[(idx[0], "level")] == "sent" and d[(idx[0], "momentum")] == "thinned"
    assert d[(idx[1], "level")] == "cooldown"
    assert d[(idx[6], "reversal")] == "sent" and d[(idx[6], "level")] == "thinned"


def test_policy_carries_history_across_window_boundary():
    """Пуш, отправленный в предыдущем окне, охлаждает первые дни следующего."""
    idx = pd.bdate_range("2026-01-05", periods=30)
    prior = {"TJS": [(idx[9], "BUY_NOW")]}
    ev = pd.DataFrame([(idx[10], "TJS", "level", "BUY_NOW", 0.8)], columns=_events(idx).columns)
    out = apply_policy(ev, {"level": 0}, idx, PolicyParams(), prior_sent=prior)
    assert out["decision"].iloc[0] == "cooldown"
    out2 = apply_policy(ev, {"level": 0}, idx, PolicyParams())
    assert out2["decision"].iloc[0] == "sent"


def _matrix_rows(split, hit, base, n, excess=None):
    excess = excess or {}
    return [
        {
            "corridor": "TJS",
            "indicator": ind,
            "split": split,
            "h": 20,
            "tol_bps": 25.0,
            "hit_mean_trunc": h,
            "base_mean_trunc": base,
            "n_scored_trunc": n,
            "benefit_excess_trunc": excess.get(ind, 10.0),
        }
        for ind, h in hit.items()
    ]


def test_rank_from_history_uses_only_past_windows_with_truncated_outcomes():
    rows = _matrix_rows(0, {"level": 0.7, "momentum": 0.4, "reversal": 0.6}, 0.55, 40, {"reversal": -5.0})
    rows += _matrix_rows(1, {"level": 0.6, "momentum": 0.45, "reversal": 0.6}, 0.55, 40, {"reversal": -5.0})
    clean = pd.DataFrame(rows)
    rank, muted = rank_from_history(clean, "TJS", before_split=2)
    assert rank["level"] < rank["momentum"]
    # моментум — lift < 1; разворот — lift > 1, но выгода сверх случайного дня ≤ 0
    assert set(muted) == {"momentum", "reversal"}
    # порча текущего и будущих окон ничего не меняет
    spoiled = pd.concat([clean, pd.DataFrame(_matrix_rows(2, {"level": 0.0, "momentum": 1.0}, 0.5, 400))])
    assert rank_from_history(spoiled, "TJS", before_split=2) == (rank, muted)
    # без усечённых столбцов — порядок по умолчанию, ничего не отключено
    _rank0, muted0 = rank_from_history(clean.drop(columns=["hit_mean_trunc"]), "TJS", before_split=2)
    assert muted0 == ()


def test_storm_blocks_push():
    idx = pd.bdate_range("2026-01-05", periods=10)
    ev = pd.DataFrame([(idx[5], "TJS", "level", "BUY_NOW", 0.8)], columns=_events(idx).columns)
    storm = pd.Series(False, index=idx)
    storm.iloc[5] = True
    out = apply_policy(ev, {"level": 0}, idx, PolicyParams(), storm=storm)
    assert out["decision"].iloc[0] == "storm"


def _row(ind, hit, base, n, split=0, excess=10.0):
    return {
        "corridor": "TJS",
        "indicator": ind,
        "split": split,
        "h": 20,
        "tol_bps": 25.0,
        "hit_mean_trunc": hit,
        "base_mean_trunc": base,
        "n_scored_trunc": n,
        "benefit_excess_trunc": excess,
    }


def test_rank_orders_by_lift_not_hit_and_weights_by_events():
    # уровень: hit 0,60 при базе 0,55 (lift 1,09); моментум: hit 0,50 при базе 0,30 (lift 1,67)
    rank, muted = rank_from_history(
        pd.DataFrame([_row("level", 0.6, 0.55, 40), _row("momentum", 0.5, 0.3, 40)]), "TJS", 1
    )
    assert rank["momentum"] < rank["level"] and muted == ()
    # взвешивание по событиям: окно с 400 событиями перевешивает окно с 5 (простое среднее дало бы lift 1,3)
    rows = [_row("level", 0.9, 0.5, 5, split=0), _row("level", 0.4, 0.5, 400, split=1)]
    _rank, muted = rank_from_history(pd.DataFrame(rows), "TJS", 2)
    assert muted == ("level",)
    # NaN в базе одного окна не роняет индикатор в конец ранга
    rows = [
        _row("level", 0.6, float("nan"), 40),
        _row("level", 0.6, 0.5, 40, split=1),
        _row("momentum", 0.4, 0.5, 40),
    ]
    rank, muted = rank_from_history(pd.DataFrame(rows), "TJS", 2)
    assert rank["level"] == 0 and muted == ("momentum",)
    # ничья по lift — порядок по умолчанию (по скорости), не алфавит
    rank, _ = rank_from_history(
        pd.DataFrame([_row("momentum", 0.6, 0.5, 40), _row("level", 0.6, 0.5, 40)]), "TJS", 1
    )
    assert rank["level"] < rank["momentum"]


def test_history_status_names_the_problem():
    import pytest

    from fxmoment.combine import history_status

    assert history_status(pd.DataFrame()) is not None
    full = pd.DataFrame([_row("level", 0.6, 0.5, 40)])
    assert history_status(full) is None
    broken = full.drop(columns=["benefit_excess_trunc"])
    assert "benefit_excess_trunc" in history_status(broken)
    with pytest.raises(ValueError):
        rank_from_history(broken, "TJS", 1, strict=True)
