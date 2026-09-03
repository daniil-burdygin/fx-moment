from fxmoment.texts import check_message, check_text, library_texts, render


def test_forbidden_phrases_are_caught():
    for text in (
        "Курс скоро вырастет",
        "Успейте, пока не подорожало",
        "Гарантируем лучший курс",
        "Заработайте на курсе",
        "Сейчас выгодно перевести",  # «выгодно» без измерения
        "Инвестируйте в валюту",
        "Курс упадет завтра",  # без «ё»
        "Завтра курс будет выше",
        "Не пропустите момент",
        "Поспешите с переводом",
        "Последний шанс перевести",
        "Ниже уже не будет",
        "Курс продолжит снижение",
        "Мы ожидаем роста курса",
        "Самое выгодное предложение",
        # регрессия 03.09: закрывающий \b после основы пропускал словоформы
        "Идеальный курс",
        "Идеальный момент для перевода",
        "Оптимальный момент для перевода",
        "Прогнозируем снижение",
        "Курсы будут ниже",
        "Цены вырастут",
        "Не упускайте момент",
        "Успеешь перевести",
        "Курс на дне",
        "Сэкономьте на переводе",
        "Выгода очевидна",
        "Выгодно, комиссия 0 %",  # процент есть, окна нет
        "Только сегодня такой курс",
        "Дальше только дороже",
        "Самое время перевести",
        "Пора действовать",
        "Курс близок к минимуму — дальше отскок",
    ):
        assert check_text(text), text
    for text in (
        "Сейчас курс выгоднее, чем в 85 % дней за три месяца",
        "Сейчас курс выгоднее, чем в 85 % дней за три месяца",  # неразрывный пробел перед %
        "Ниже, чем в 85 % дней за три месяца, — выгоднее обычного",
        "Курс сомони 9,12 ₽ — ниже, чем в 88 % дней за 3 месяца",
        "Курс сомони пошёл вверх: +0,8 % от минимума за 3 месяца (9,05 → 9,12 ₽)",
        "Снижение остановилось 4 дн. назад",
    ):
        assert not check_text(text), text


def test_all_hits_reported_not_only_first():
    hits = check_text("Курс вырастет и упадет")
    assert [h[0] for h in hits] == ["вырастет", "упадет"]


def test_message_checked_as_a_whole():
    assert not check_message("Выгодный курс", "Ниже, чем в 85 % дней за 3 месяца")
    assert check_message("Выгодный курс", "Переведите сегодня")
    # оборот, разорванный границей заголовок/тело
    assert check_message("Ниже уже", "не будет")


def test_library_is_clean():
    for name, title, body in library_texts():
        assert not check_message(title, body), (name, title, body)


def test_render_level_uses_facts():
    title, body = render(
        "TJS", "BUY_NOW", "level", 9.1234, {"pct_rank": 0.08, "window": 120, "days_since_min": 0}
    )
    assert "92 %" in body and "9,12" in body and "6 мес" in body


def test_seasonality_title_carries_the_number_and_the_base():
    title, body = render(
        "TJS", "BUY_NOW", "seasonality", 9.12, {"k_years": 5, "n_years": 6, "target_month": 12, "streak": 2}
    )
    assert "5 из 6" in title and "декабре" in title and "последних" in title and "в среднем" in title
    assert "среднего за предыдущий месяц" in body


def test_dip_text_names_publication_days_not_weeks():
    title, body = render(
        "TJS", "BUY_NOW", "dip_vs_trend", 9.12, {"dev_pct": -1.4, "pct_rank": 0.1, "window": 120, "span": 80}
    )
    assert "80 рабочих дней" in title and "включая сегодняшний" in body and "нед" not in title
