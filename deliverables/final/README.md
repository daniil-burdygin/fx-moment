# Финальные артефакты к сдаче 07.09.2026 10:00 МСК

Папка повторяет четыре пункта карточки Talent Track: в подпапке лежит ровно то, что грузится в этот пункт. Один комплект на всю команду.

Копии в `3_…` и `4_…` делаются одной командой после финальной пересборки отчётов (раздел «Команда копирования»); в `1_…` и `2_…` лежат оригиналы. Скопировали — сверьте, что числа в копиях совпадают с `reports/`.

Ограничения формы сняты с самой карточки (атрибут `accept` полей загрузки, осмотр 04.09 02:05), а не с описания пункта: продуктовые материалы принимают `.pdf`, `.xlsx`, `.md`, но не `.html` и не `.png` — интерактивные страницы и графики идут в дополнительные.

| Пункт карточки | Форма | Подпапка | Что грузить |
|---|---|---|---|
| Презентация проекта | `.pdf`, `.pptx`, до 2 файлов | `1_Презентация_проекта/` | финальная колода собирается командой утром 07.09; продуктовые слайды кладутся сюда же. Печать PDF из HTML — командой из README промежуточных артефактов |
| └ продуктовые слайды 0a и 0b | вклад в колоду | `1_Презентация_проекта/` | `Слайд_00_продукт.html` — исходник обоих слайдов 1280×720; `Слайд_00_продукт.pdf` — 2 страницы, печать headless Chrome; `Слайд_00_продукт-1.jpg`, `-2.jpg` — контрольные рендеры 110 dpi; `Слайд_00_речь.md` — речь по минуте на слайд. Числа — из `docs/product/customer-journey.md` и `reports/latest/analysis/` |
| └ слайды-дополнение | вклад в колоду | `1_Презентация_проекта/` | `Слайды_дополнение.html/.pdf` (3 страницы: цена ожидания · что не сработало · финал с фразой дословно), `-1…-3.jpg` — контрольные рендеры, `Слайды_дополнение_речь.md` — речь и источники чисел. Собраны в ночь на 07.09 под регламент организатора (`demo-day-pitch-guide.pdf`) |
| └ замечания к v4 | для напарников | `1_Презентация_проекта/` | `Замечания_к_v4.md` — построчная сверка колоды v4 с репозиторием: что чинить, что вставить, тайминг, README |
| Описание проекта | `.md`, 1 файл | `2_Описание_проекта/` | `Описание_проекта_финальное.md` — готово |
| Продуктовые материалы | `.pdf`, `.xlsx`, `.md`, до 10 | `3_Продуктовые_материалы/` | девять файлов из списка ниже |
| Дополнительные материалы | любой формат, до 10 | `4_Дополнительные_материалы/` | десять файлов из списка ниже |
| Репозиторий команды | ссылка, обязательно | — | <https://github.com/daniil-burdygin/fx-moment> |

## Продуктовые материалы (9 из 10 слотов)

| Файл в папке | Источник | Статус |
|---|---|---|
| `Продуктовая_рамка.md` | собрана заново 06.09 из промежуточной версии под `pilot.md`, `concept.md`, `customer-journey.md`; лежит в папке | есть, копировать не надо |
| `concept.md` | `docs/product/concept.md` | есть |
| `texts.md` | `docs/product/texts.md` | есть |
| `pilot.md` | `docs/product/pilot.md` | есть |
| `limitations.md` | `docs/product/limitations.md` | есть |
| `star-task.md` | `docs/product/star-task.md` | есть |
| `market.md` | `docs/product/market.md` | есть |
| `Лист_ответов.md` | `docs/defense/qa.md` (строка Ш16) | есть |
| `Запуск_с_чистого_клона.md` | `docs/process/clean-clone-check.md` (строка Ш15) | есть |

Отдельного `customer-journey.md` нет и не нужно: клиентский путь с развилками закрыт таблицей в `concept.md`, разделом описания и `star-task.md`. Десятый слот свободен под финальный отчёт по бэктесту, если он будет собран отдельным файлом.

## Дополнительные материалы (10 из 10 слотов)

| Файл в папке | Источник |
|---|---|
| `Машина_времени_сигналы_на_дату.html` | `prototype/timemachine.html` (пересобрать `uv run fxmoment timemachine` после финального прогона) |
| `Прототип_момент_изменился.html` | `prototype/index.html` — три состояния, «почему это сообщение», журнал попаданий |
| `Окна_и_метрики_интерактивный_разбор.html` | `deliverables/промежуточные_артефакты/4_Дополнительные_материалы/` |
| `reports_latest_README.md` | `reports/latest/README.md` |
| `reports_latest_analysis_README.md` | `reports/latest/analysis/README.md` |
| `reports_intraday_README.md` | `reports/intraday/README.md` |
| `reports_intraday_README_bars.md` | `reports/intraday/README_bars.md` |
| `chart_TJS.png` | `reports/latest/chart_TJS.png` |
| `frontier.png` | `reports/latest/analysis/frontier.png` |
| `day_of_month.png` | `reports/latest/analysis/day_of_month.png` |

Слотов ровно десять. `cbr_vs_moex_by_hour.png` и графики остальных четырёх коридоров в комплект не помещаются и остаются в репозитории.

## Команда копирования (запускать утром 07.09, после финального прогона)

```bash
cd ~/projects/fx-moment && cp docs/product/{concept,texts,pilot,limitations,star-task,market}.md deliverables/final/3_Продуктовые_материалы/ && cp prototype/timemachine.html deliverables/final/4_Дополнительные_материалы/Машина_времени_сигналы_на_дату.html && cp deliverables/промежуточные_артефакты/4_Дополнительные_материалы/Окна_и_метрики_интерактивный_разбор.html deliverables/final/4_Дополнительные_материалы/ && cp reports/latest/README.md deliverables/final/4_Дополнительные_материалы/reports_latest_README.md && cp reports/latest/analysis/README.md deliverables/final/4_Дополнительные_материалы/reports_latest_analysis_README.md && cp reports/intraday/README.md deliverables/final/4_Дополнительные_материалы/reports_intraday_README.md && cp reports/intraday/README_bars.md deliverables/final/4_Дополнительные_материалы/reports_intraday_README_bars.md && cp reports/latest/chart_TJS.png reports/latest/analysis/frontier.png reports/latest/analysis/day_of_month.png deliverables/final/4_Дополнительные_материалы/
```

Вторая строка, для файлов из работы 06.09: `cp docs/defense/qa.md deliverables/final/3_Продуктовые_материалы/Лист_ответов.md && cp docs/process/clean-clone-check.md deliverables/final/3_Продуктовые_материалы/Запуск_с_чистого_клона.md && cp prototype/index.html deliverables/final/4_Дополнительные_материалы/Прототип_момент_изменился.html`.

## Загрузка

Через настоящий Chrome (`upload_file`), не через внутреннюю панель браузера: она не отдаёт файл в системный диалог. Два условия — путь файла внутри рабочего каталога сессии и скрытые `input[type=file]` предварительно сделаны видимыми скриптом. После загрузки перезагрузить страницу карточки и убедиться, что всё сохранилось на сервере.
