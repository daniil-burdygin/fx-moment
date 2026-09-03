"""Машина времени (demo.py): вердикты пушей — теми же функциями, что разметка; страница собирается
офлайн и не раскрывает данных после снимка."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fxmoment import labels
from fxmoment.backtest.walkforward import Split
from fxmoment.config import BUY_NOW, WINDOW_CLOSING
from fxmoment.demo import PLACEHOLDER, render_timemachine, timemachine_payload, write_timemachine

H, TOL = 20, 25.0


def _panel(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n, name="pub_date")
    rate = 10 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"TJS": rate, "USD": 70 + rng.normal(0, 0.3, n)}, index=idx)


def _decisions(idx: pd.DatetimeIndex, n: int) -> pd.DataFrame:
    facts = json.dumps({"pct_rank": 0.08, "window": 120, "days_since_min": 0})
    rows = [
        (idx[100], "TJS", "level", BUY_NOW, 0, 10.0, facts, "sent", BUY_NOW),
        (
            idx[150],
            "TJS",
            "reversal",
            WINDOW_CLOSING,
            0,
            10.2,
            json.dumps({"rise_pct": 0.8, "min_rate": 9.9, "window": 120}),
            "sent",
            WINDOW_CLOSING,
        ),
        (idx[n - 5], "TJS", "level", BUY_NOW, 0, 9.5, facts, "sent", BUY_NOW),
        (
            idx[120],
            "TJS",
            "momentum",
            BUY_NOW,
            0,
            10.1,
            json.dumps({"streak": 4, "drop_pct": -1.2}),
            "muted",
            None,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "corridor",
            "indicator",
            "scenario",
            "split",
            "rate",
            "facts",
            "decision",
            "push_scenario",
        ],
    ).assign(direction="down", strength=0.5, speed="medium", params="{}")


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("level", "TJS", 0, BUY_NOW, H, TOL, 0.5),
            ("level", "TJS", 0, BUY_NOW, H, 0.0, 0.4),
            ("reversal", "TJS", 0, WINDOW_CLOSING, H, 0.0, 0.3),
            ("reversal", "TJS", 0, WINDOW_CLOSING, H, TOL, 0.2),
        ],
        columns=["indicator", "corridor", "split", "scenario", "h", "tol_bps", "base_mean"],
    )


def test_payload_verdicts_match_labels() -> None:
    panel = _panel()
    idx = panel.index
    n = len(idx)
    splits = [Split(0, idx[80], idx[90], idx[n - 1])]
    payload = timemachine_payload(
        panel,
        _decisions(idx, n),
        _matrix(),
        splits,
        {"code": "abc1234", "last_eff_date": "x"},
        h=H,
        tol_bps=TOL,
    )
    assert payload["dates"] == [f"{d:%Y-%m-%d}" for d in idx]
    assert set(payload["series"]) == {"TJS", "USD"}
    rate = panel["TJS"]
    pushes = {p["i"]: p for p in payload["pushes"]}
    assert set(pushes) == {100, 150, n - 5}
    assert pushes[100]["hit"] is bool(labels.hit_buy_now(rate, H, TOL, mode="mean").iloc[100])
    assert pushes[100]["hit_min"] is bool(labels.hit_buy_now(rate, H, TOL, mode="min").iloc[100])
    assert pushes[100]["benefit"] == float(f"{labels.benefit_fwd_bps(rate, H).iloc[100]:.6g}")
    assert pushes[150]["hit"] is bool(labels.hit_window_closing(rate, H, 0.0).iloc[150])
    assert pushes[n - 5]["hit"] is None and pushes[n - 5]["benefit"] is None  # горизонт не закрыт
    assert pushes[100]["title"] and "TJS" not in pushes[100]["title"]  # текст отрисован библиотекой
    assert payload["held"] == [{"i": 120, "c": "TJS", "ind": "momentum", "sc": BUY_NOW, "why": "muted"}]
    w = payload["windows"][0]
    assert (w["start"], w["end"]) == (f"{idx[90]:%Y-%m-%d}", f"{idx[n - 1]:%Y-%m-%d}")
    assert w["base"]["TJS"] == {BUY_NOW: 0.5, WINDOW_CLOSING: 0.3}  # WC — строки с допуском 0
    assert payload["meta"]["code"] == "abc1234" and payload["meta"]["n_pushes"] == 3


def test_render_embeds_payload_offline() -> None:
    panel = _panel(200)
    idx = panel.index
    payload = timemachine_payload(
        panel,
        _decisions(idx, 200).iloc[:1],
        _matrix(),
        [Split(0, idx[80], idx[90], idx[-1])],
        {"code": "deadbee"},
    )
    html = render_timemachine(payload)
    assert PLACEHOLDER not in html and "deadbee" in html and payload["pushes"][0]["title"] in html
    assert "<script src" not in html and "http" not in html.split("<script>")[1].split("const D")[0]


def test_write_from_real_report_if_present(tmp_path: Path) -> None:
    from fxmoment.data.store import load_panel, repo_root

    out_dir = repo_root() / "reports" / "latest"
    if not (out_dir / "stream_decisions.csv").exists():
        import pytest

        pytest.skip("нет reports/latest")
    meta = write_timemachine(load_panel(), out_dir, tmp_path / "tm.html")
    sent = pd.read_csv(out_dir / "stream_decisions.csv")["decision"].eq("sent").sum()
    assert meta["n_pushes"] == int(sent) and (tmp_path / "tm.html").stat().st_size > 10_000
