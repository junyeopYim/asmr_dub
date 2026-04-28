from __future__ import annotations

import re


class MeCab:
    """Tiny fallback for g2pk2 when python-mecab-ko cannot be installed.

    The real Mecab tokenizer improves Korean pronunciation rules, but g2pk2 only
    needs a ``pos`` method to avoid failing at runtime. Returning noun-like tags
    keeps the text path usable when system ``mecab-config`` is unavailable.
    """

    _token_pattern = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")

    def pos(self, text: str) -> list[tuple[str, str]]:
        return [(match.group(0), "NNG") for match in self._token_pattern.finditer(text)]
