import pandas as pd

from fxmoment.combine import PolicyParams, apply_policy


def test_policy_cooldown_and_conflict():
    idx = pd.bdate_range("2026-01-05", periods=20)
    events = pd.DataFrame(
        [
            (idx[0], "TJS", "level", "BUY_NOW", 0.8),
            (idx[0], "TJS", "momentum", "BUY_NOW", 0.9),  # прореживается: ранг хуже
            (idx[1], "TJS", "level", "BUY_NOW", 0.7),  # охлаждение
            (idx[6], "TJS", "reversal", "WINDOW_CLOSING", 0.6),  # BUY_NOW был недавно → закрытие окна
            (idx[6], "TJS", "level", "BUY_NOW", 0.9),
        ],
        columns=["date", "corridor", "indicator", "scenario", "strength"],
    )
    out = apply_policy(events, {"level": 0, "reversal": 1, "momentum": 2}, idx, PolicyParams())
    d = dict(zip(zip(out["date"], out["indicator"], strict=True), out["decision"], strict=True))
    assert d[(idx[0], "level")] == "sent" and d[(idx[0], "momentum")] == "thinned"
    assert d[(idx[1], "level")] == "cooldown"
    assert d[(idx[6], "reversal")] == "sent" and d[(idx[6], "level")] == "thinned"
