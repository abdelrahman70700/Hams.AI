from hamsrag.text import analyze, clean_text, detect_language, stem_ar, tokenize


def test_tashkeel_tatweel_and_presentation_forms():
    assert clean_text("سِيَاسَةُ الاسْتِرْجَاعِ") == "سياسة الاسترجاع"
    assert clean_text("الــــمنتج") == "المنتج"
    # Presentation forms (typical broken PDF extraction) fold back to base letters
    assert clean_text("ﻣﻨﺘﺠ") == "منتج"


def test_numerals_are_interchangeable():
    assert analyze("خلال ١٤ يوماً") == analyze("خلال 14 يوما")
    assert "1500" in analyze("السعر ١٬٥٠٠ ريال")
    assert "1500" in analyze("price SAR 1,500")
    assert "2.5" in analyze("٢٫٥ كجم")


def test_alef_ya_ta_marbuta_folding():
    assert analyze("إرجاع") == analyze("ارجاع") == analyze("أرجاع")
    assert analyze("مستشفى") == analyze("مستشفي")
    assert analyze("الباقة المؤسسية") == analyze("الباقه المؤسسيه")


def test_script_boundary_split_mixed_query():
    toks = tokenize("أبغى أعرف الـpricing حق الباقة المؤسسية")
    assert "pricing" in toks and "ال" in toks
    assert "price" in analyze("أبغى أعرف الـpricing حق الباقة المؤسسية")  # stemmed EN token


def test_light_stemming():
    assert stem_ar("والمنتجات") == "منتج"
    assert stem_ar("بالضمان") == "ضمان"
    assert analyze("الضمان") == analyze("ضمان")
    assert analyze("الطلبات المتأخرة")[0] == analyze("طلبات")[0]


def test_english_stemming_and_stopwords():
    assert analyze("What are the refunds policies?") == analyze("refund policy")


def test_language_detection():
    assert detect_language("ما هي سياسة الاسترجاع للطلبات المتأخرة؟")["lang"] == "ar"
    assert detect_language("What is the SLA for premium support tickets?")["lang"] == "en"
    d = detect_language("أبغى أعرف الـpricing حق الباقة المؤسسية")
    assert d["lang"] == "ar" and d["mixed"]
    assert detect_language("ما هو الضمان؟ answer in English")["answer_lang"] == "en"


def test_no_overstemming_of_short_stems():
    for w in ["ضمان", "مكان", "بيان", "زمان"]:
        assert stem_ar(w) == w
    assert stem_ar("منتجاتها") == "منتج"
    assert stem_ar("الرياض") == "رياض"
