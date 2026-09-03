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
    ):
        assert check_text(text), text
    for text in (
        "Сейчас курс выгоднее, чем в 85 % дней за три месяца",
        "Ниже, чем в 85 % дней за три месяца, — выгоднее обычного",
        "Курс сомони 9,12 ₽ — ниже, чем в 88 % дней за 3 месяца",
        "Курс сомони пошёл вверх: +0,8 % от минимума за 3 месяца (9,05 → 9,12 ₽)",
    ):
        assert not check_text(text), text


def test_message_checked_as_a_whole():
    assert not check_message("Выгодный курс", "Ниже, чем в 85 % дней за 3 месяца")
    assert check_message("Выгодный курс", "Переведите сегодня")


def test_library_is_clean():
    for name, title, body in library_texts():
        assert not check_message(title, body), (name, title, body)


def test_render_level_uses_facts():
    title, body = render(
        "TJS", "BUY_NOW", "level", 9.1234, {"pct_rank": 0.08, "window": 120, "days_since_min": 0}
    )
    assert "92 %" in body and "9,12" in body and "6 мес" in body


def test_seasonality_title_carries_the_number():
    title, _body = render(
        "TJS", "BUY_NOW", "seasonality", 9.12, {"k_years": 5, "n_years": 6, "target_month": 12}
    )
    assert "5 из 6" in title
