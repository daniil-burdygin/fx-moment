"""Отчёт внутридневного прогона: провенанс каталога.

Каталог отчётов без `provenance.json` неотличим для `analysis.backtest_provenance` и
`variants._provenance` от каталога, собранного неизвестно каким кодом и на каком снимке:
обе функции ищут файл с этим именем по каталогу. `reports/intraday` называл свой провенанс
`provenance_bars.json` и потому для них не существовал (найдено 06.09).

Ось здесь дневная (`step_scale = 1`): проверяется запись отчёта, а не масштабирование сеток —
оно в `test_profiles.py`. Прогон урезан до одного коридора и одного индикатора, чтобы тест
стоил секунды.
"""

import json
from dataclasses import replace
from datetime import datetime

from fxmoment.indicators import Level
from fxmoment.intraday import run_profile, write_intraday_report
from fxmoment.profiles import DAILY

SMALL = replace(
    DAILY,
    name="test-report",
    corridors=("TJS",),
    context=("USD",),
    indicators=(Level,),
    horizons=(5, 20),
    first_test="2019-01-01",
)


def test_intraday_report_writes_provenance(panel, tmp_path):
    results = run_profile(panel, SMALL)
    assert set(results) == {"TJS"}
    out = write_intraday_report(results, panel, tmp_path, SMALL)

    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["code"]  # хеш HEAD, при грязном дереве с суффиксом -dirty
    assert datetime.strptime(prov["built_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    # вход прогона — снимок часовых свечей, а не дневной снимок ЦБ
    assert set(prov) >= {"fetched_at_utc", "interval_length", "profile", "bars_per_day", "windows"}
    assert prov["profile"] == SMALL.name and prov["bars_per_day"] == SMALL.step_scale
    assert prov["windows"]["TJS"] == [s.label() for s in results["TJS"].splits]
    # дневного отчёта не дали — чужой штамп не выдуман
    assert prov["daily_report"] is None
    # имя одно на все каталоги отчётов: старое рядом не остаётся
    assert not (out / "provenance_bars.json").exists()
