import pandas as pd

from fxmoment.data.calendar import as_of, to_publication_panel


def test_publication_axis_and_unit_rate():
    long = pd.DataFrame(
        {
            "currency": ["TJS", "TJS", "TJS", "USD", "USD", "USD"],
            # вт 25.08, ср 26.08, сб 29.08 (опубликован в пятницу 28.08); понедельника нет
            "eff_date": pd.to_datetime(["2026-08-25", "2026-08-26", "2026-08-29"] * 2),
            "nominal": [10, 10, 10, 1, 1, 1],
            "value": [90.0, 91.0, 92.0, 80.0, 81.0, 82.0],
            "unit_rate": [9.0, 9.1, 9.2, 80.0, 81.0, 82.0],
        }
    )
    panel = to_publication_panel(long)
    assert list(panel.index) == list(pd.to_datetime(["2026-08-24", "2026-08-25", "2026-08-28"]))
    assert panel.loc["2026-08-28", "TJS"] == 9.2
    assert set(panel.columns) == {"TJS", "USD"}


def test_as_of_excludes_future():
    idx = pd.bdate_range("2026-01-01", periods=10)
    panel = pd.DataFrame({"TJS": range(10)}, index=idx)
    cut = as_of(panel, idx[4])
    assert cut.index[-1] == idx[4] and len(cut) == 5
