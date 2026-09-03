from fxmoment.texts import check_text, library_texts, render


def test_forbidden_phrases_are_caught():
    assert check_text("Курс скоро вырастет")
    assert check_text("Успейте, пока не подорожало")
    assert check_text("Гарантируем лучший курс")
    assert check_text("Заработайте на курсе")
    assert check_text("Сейчас выгодно перевести")  # «выгодно» без измерения
    assert not check_text("Сейчас курс выгоднее, чем в 85 % дней за три месяца")


def test_library_is_clean():
    for name, title, body in library_texts():
        assert not check_text(title), (name, title)
        assert not check_text(body), (name, body)


def test_render_level_uses_facts():
    title, body = render(
        "TJS", "BUY_NOW", "level", 9.1234, {"pct_rank": 0.08, "window": 120, "days_since_min": 0}
    )
    assert "92 %" in body and "9,12" in body and "6 мес" in body
