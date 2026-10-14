"""Tests for per-source text cleaning."""

import pytest

from tinylm import clean


def lines(src: list[str], source: str, **kw: bool) -> list[str]:
    return list(clean.apply(src, source, clean.Options(**kw)))


def test_typography_quotes_dashes_controls_and_whitespace():
    assert clean.typography("“Hi,” she said — it’s ok") == (
        '"Hi," she said - it\'s ok'
    )
    assert clean.typography("  a \t b   c \n") == "a b c"
    assert clean.typography("a\x00b\x07c") == "abc"
    assert clean.typography("   ") == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "WE HAVE THE CO-HOSTS OF WAKE UP.",
            "We have the co-hosts of wake up.",
        ),
        ("I THINK I CAN, AND I WILL", "I think I can, and I will"),
        ("Hello World, this is NASA", "Hello World, this is NASA"),
        ("OK", "OK"),
        ("123 ...", "123 ..."),
    ],
)
def test_repair_case(text: str, expected: str):
    assert clean.recase(text) == expected


def test_only_real_speaker_tags_are_removed():
    assert lines(["*MOT:\tmore what?"], "childes") == ["more what?"]
    assert lines(["A:\tYeah, uh,"], "switchboard") == ["Yeah, uh,"]
    assert lines(["Note: this matters"], "childes") == ["Note: this matters"]
    assert lines(["Right."], "switchboard") == ["Right."]


def test_childes_pipeline_and_keep_tags_option():
    src = ["*MOT:\tmore what?", "[leaves room.]", "*CHI:\tmore tapioca."]
    assert lines(src, "childes") == ["more what?", "more tapioca."]
    assert lines(src[:1], "childes", keep_speaker_tags=True) == [
        "MOT: more what?"
    ]


def test_wiki_headings_case_blank_lines_and_unknown_source():
    assert lines(
        ["= = = Bullet the Blue Sky = = =", "It is a song."], "simple_wiki"
    ) == ["Bullet the Blue Sky", "It is a song."]
    assert lines(["WHO ARE RIGHT HERE."], "open_subtitles") == [
        "Who are right here."
    ]
    assert lines(["", "  ", "Hi."], "gutenberg") == ["Hi."]
    with pytest.raises(ValueError, match="source"):
        lines(["x"], "tweets")


def test_cleaning_is_idempotent():
    src = ["*MOT:\t“Well,” she said.", "= = = T = = ="]
    once = lines(src, "childes")
    assert once == lines(once, "childes")


def test_drop_duplicate_long_lines_keeps_short_repeats():
    long_line = " ".join(["word"] * 25)
    out = list(
        clean.dedupe([long_line, "Yeah.", long_line, "Yeah."], min_words=20)
    )
    assert out == [long_line, "Yeah.", "Yeah."]
    with pytest.raises(ValueError, match="min_words"):
        list(clean.dedupe(["a"], min_words=0))


def test_dedupe_can_share_state_across_sources():
    long_line = " ".join(["word"] * 25)
    seen: set[bytes] = set()
    first = list(clean.dedupe([long_line, "a."], min_words=20, seen=seen))
    second = list(clean.dedupe([long_line, "a."], min_words=20, seen=seen))
    assert first == [long_line, "a."]
    assert second == ["a."]


def test_unicode_is_normalised_to_nfc():
    decomposed = "café au lait"
    assert clean.typography(decomposed) == "café au lait"
    assert clean.typography("café au lait") == clean.typography(decomposed)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello big world", 3),
        ("", 0),
        ("   ", 0),
        ("你好世界", 4),
        ("hello 你好 world", 4),
        ("こんにちは", 5),
        ("今日は天気", 5),
        ("naïve café", 2),
    ],
)
def test_words_counts_spaced_words_and_unspaced_characters(
    text: str, expected: int
):
    assert clean.words(text) == expected


def test_chinese_lines_are_deduplicated_by_character_length():
    line = "这是一个很长的句子" * 4
    seen: set[bytes] = set()
    kept = list(
        clean.dedupe([line, "短。", line, "短。"], min_words=20, seen=seen)
    )
    assert kept == [line, "短。", "短。"]
