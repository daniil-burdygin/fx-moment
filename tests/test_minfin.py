"""Режим бюджетного правила: каузальность и разбор снимка (строка Ш14 плана)."""

from __future__ import annotations

import pandas as pd
import pytest

from fxmoment import minfin


@pytest.fixture
def ann() -> pd.DataFrame:
    return minfin.load_announcements()


def test_snapshot_is_well_formed(ann: pd.DataFrame) -> None:
    assert len(ann) > 90
    assert ann["announce_date"].is_monotonic_increasing
    assert ann["announce_date"].notna().all()
    assert set(ann["direction"]) <= set(minfin.REGIMES)
    assert (ann["source_url"].str.startswith("https://")).all()
    named = ann[ann["direction"] != "none"]
    assert named["rub_per_day_bln"].notna().all()
    assert (named["rub_per_day_bln"] > 0).all()
    assert (named["from_date"] <= named["to_date"]).all()
    # период всегда начинается не раньше объявления: иначе признак смотрел бы назад в собственную дату
    assert (named["from_date"] >= named["announce_date"]).all()


def test_regime_uses_only_announcements_published_by_t(ann: pd.DataFrame) -> None:
    """Признак на дату T не меняется, если из снимка убрать всё, что вышло после T."""
    idx = pd.date_range("2021-01-01", "2026-06-30", freq="B")
    full = minfin.regime_series(idx, ann)
    for t in ("2021-03-15", "2022-02-10", "2023-01-20", "2026-03-10", "2026-05-20"):
        cut = pd.Timestamp(t)
        as_of = minfin.regime_series(idx[idx <= cut], ann, as_of=cut)
        assert (as_of == full.loc[as_of.index]).all(), t


def test_regime_marks_the_2022_pause(ann: pd.DataFrame) -> None:
    idx = pd.date_range("2021-12-01", "2023-02-28", freq="B")
    reg = minfin.regime_series(idx, ann)
    assert reg[pd.Timestamp("2021-12-20")] == "buy"
    assert reg[pd.Timestamp("2022-01-25")] == "none"  # ЦБ прекратил покупки 24.01.2022
    assert (reg.loc["2022-06-01":"2022-12-31"] == "none").all()
    assert reg[pd.Timestamp("2023-02-01")] == "sell"


def test_regime_marks_the_2026_pause(ann: pd.DataFrame) -> None:
    idx = pd.date_range("2026-02-01", "2026-06-30", freq="B")
    reg = minfin.regime_series(idx, ann)
    assert (reg.loc["2026-03-10":"2026-04-30"] == "none").all()
    assert reg[pd.Timestamp("2026-06-01")] == "buy"


def test_unknown_direction_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "announce_date,from_date,to_date,direction,rub_per_day_bln,source_url,verified,note\n"
        "2024-01-10,2024-01-15,2024-02-06,hold,1.0,https://example.org,yes,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="неизвестное направление"):
        minfin.load_announcements(path)
