"""Pin the pre-checked, cached abbr_match to the regex-per-pair code it replaced."""

from __future__ import annotations

import re
import sys

import numpy as np
import pytest

from emm.features.features_name import WORD_SPLIT, abbr_match, original_abbr_match


def reference_extract_abbr_merged_initials(abbr, name):
    regex = r"\b"
    for char in abbr.lower():
        regex += char + r"\w+\s?"
    return re.search(regex, name.lower(), re.UNICODE)


def reference_abbr_match(str_with_abbr, str_with_open_form):
    if any(c.islower() for c in str_with_abbr) and any(c.isupper() for c in str_with_abbr):
        return original_abbr_match(str_with_abbr, str_with_open_form)
    return any(
        2 <= len(token) <= 5 and reference_extract_abbr_merged_initials(token, str_with_open_form) is not None
        for token in WORD_SPLIT.split(str_with_abbr)
    )


# characters whose lower case form is longer, or which sit on the \w and \b borders
ODD_NAMES = ["İstanbul AB", "ß ss", "a.b c", "ÀB ÉC", "12 ab", "ab_c d", "ǅ x", "ﬁ ab", "SS", "ſa", "", " ", "x y"]
NAMES = [
    *ODD_NAMES,
    "ING BANK B.V.",
    "FC Barcelona",
    "PetroBras",
    "Petroleo Brasileiro B.V.",
    "SK",
    "Fenerbahce Spor Klubu",
    "fenerbahce spor klubu",
    "fc barcelona",
    "abc bcd",
    "sa ka",
    "ska",
    "sakil",
    "1a 2b",
    "_a _b",
    "İab bcd",
]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("open_form", NAMES)
def test_abbr_match_identical_on_listed_names(name, open_form):
    assert abbr_match(name, open_form) == reference_abbr_match(name, open_form)


@pytest.mark.parametrize("seed", range(5))
def test_abbr_match_identical_on_random_names(seed):
    rng = np.random.default_rng(seed)
    alphabet = list("abcdeABCDE  .-_19") + ["İ", "ß", "ﬁ", "é"]

    def random_name():
        return "".join(rng.choice(alphabet, size=rng.integers(0, 18)))

    for _ in range(2000):
        name, open_form = random_name(), random_name()
        assert abbr_match(name, open_form) == reference_abbr_match(name, open_form), (name, open_form)


def test_abbr_match_finds_open_form():
    assert abbr_match("sk", "fenerbahce spor klubu")
    assert not abbr_match("sk", "klubu spor")
    # the letters may sit in one word, as long as the first starts it
    assert abbr_match("sk", "sakil")
    assert not abbr_match("sk", "asakil")


def test_token_first_letter_is_always_a_word_character():
    # the pre-check looks the first letter up among word initials, which holds only if \b before it means a word start
    word = re.compile(r"\w", re.UNICODE)
    for code in range(sys.maxunicode + 1):
        char = chr(code)
        if word.fullmatch(char):
            assert word.fullmatch(char.lower()[0]), hex(code)
