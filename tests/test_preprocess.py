from otzyv.preprocess import normalize


def test_lowercase_and_yo():
    assert normalize("ЁЛКА Всё") == "елка все"


def test_urls_replaced():
    assert normalize("смотри https://bit.ly/abc123 и dohod-x.ru") == "смотри urltoken и urltoken"


def test_phone_and_handle():
    out = normalize("звони +7 912 345-67-89 или пиши @money_bot")
    assert "phonetoken" in out and "handletoken" in out
    assert "912" not in out


def test_numbers_and_signs_kept():
    assert normalize("скидка 90%!!!") == "скидка num % ! ! !"
    assert normalize("5000 ₽") == "num ₽"


def test_repeated_letters_collapsed():
    assert normalize("оооочень круууто") == "оочень " + "кр" + "уу" + "то"


def test_junk_punctuation_removed():
    assert normalize("  «Хорошо» — (очень)...  ") == "хорошо очень"


def test_email_like_handle_not_broken():
    assert normalize("пишите на hello@site.ru").startswith("пишите на")
