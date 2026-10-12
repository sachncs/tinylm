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


def test_speaker_tags():
    assert clean.untag("*MOT:\tmore what?") == "more what?"
    assert clean.untag("A:\tYeah, uh,") == "Yeah, uh,"
    assert clean.untag("Right.") == "Right."
    assert clean.untag("Note: this matters") == "Note: this matters"


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
