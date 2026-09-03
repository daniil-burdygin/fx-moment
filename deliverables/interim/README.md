# Промежуточные артефакты — сдать в Talent Track до 04.09.2026 10:00

Загрузку делает Даниил вручную, крайний срок — 04.09.2026 10:00 МСК. Ниже — что именно грузить и в каком состоянии оно сейчас.

| Артефакт в карточке | Что загружаем | Статус |
|---|---|---|
| Описание проекта (Markdown, обязательно) | `Описание_проекта_промежуточное.md` | готово: числа сверены с `reports/` двумя волнами роя после аудита кода, `prosaic check` чист |
| Продуктовые материалы (обязательно, до 10 файлов) | `../../docs/product/concept.md`, `texts.md`, `pilot.md`, `limitations.md`, `star-task.md` | готовы: числа сверены роем агентов, `prosaic check` прогнан по каждому файлу |
| Презентация (обязательно) | `Презентация_промежуточная.pptx` | не начата |
| Дополнительные материалы | ссылка на репозиторий, `../../reports/latest/README.md`, `../../reports/latest/analysis/README.md`, `../../reports/intraday/README.md` (сверка источников) и `README_bars.md` (прогон на часовой оси), графики `chart_*.png`, `frontier.png`, `day_of_month.png`, `cbr_vs_moex_by_hour.png` | готовы |

Все отчёты пересобраны на одном коммите: `reports/latest`, `reports/latest/analysis`, `reports/fixed` и `reports/intraday` согласованы между собой, `provenance.json` без суффикса `-dirty`.

Проверка перед загрузкой — три команды и один взгляд:

```bash
uv run pytest && uv run ruff check . && git status --short
```

Тестов должно быть 96, `git status` — пустым, а в трёх файлах провенанса
(`reports/latest/provenance.json`, `reports/fixed/provenance.json`,
`reports/intraday/provenance_bars.json`) должен стоять один и тот же хеш кода
без суффикса `-dirty`.

Совпадения хешей между собой мало: они говорят, что пять отчётов собраны на одном коммите,
но не о том, что с тех пор не правили код. Это отдельная команда:

```bash
CODE=$(python3 -c "import json; print(json.load(open('reports/latest/provenance.json'))['code'])")
git diff --stat "$CODE"..HEAD -- src tests pyproject.toml uv.lock
```

Пустой вывод значит, что числа отчётов получены тем кодом, который лежит в репозитории.
Непустой — отчёты пересобирать: `provenance.json` называет коммит сборки, а не содержимое
кода, и молча устаревает от любой правки в `src/`.
