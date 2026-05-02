from __future__ import annotations

import re

from asmr_dub_pipeline.schemas import KoreanTranslation

COLLOQUIAL_REWRITE_NOTE = "korean_colloquial_postprocess"

_SPACE_RE = re.compile(r"\s+")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")

_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("이것은", "이건"),
    ("그것은", "그건"),
    ("저것은", "저건"),
    ("이것을", "이걸"),
    ("그것을", "그걸"),
    ("저것을", "저걸"),
    ("이것이", "이게"),
    ("그것이", "그게"),
    ("저것이", "저게"),
    ("저는", "전"),
    ("나는", "난"),
    ("너는", "넌"),
    ("나를", "날"),
    ("너를", "널"),
    ("것은", "건"),
    ("것을", "걸"),
    ("것이", "게"),
    ("무엇을", "뭘"),
    ("무엇이", "뭐가"),
    ("무엇은", "뭐는"),
    ("무엇", "뭐"),
)

_FORMAL_ENDINGS: tuple[tuple[str, str], ...] = (
    ("것입니다", "거예요"),
    ("해드리겠습니다", "해드릴게요"),
    ("드리겠습니다", "드릴게요"),
    ("시작하겠습니다", "시작할게요"),
    ("해보겠습니다", "해볼게요"),
    ("하겠습니다", "할게요"),
    ("보겠습니다", "볼게요"),
    ("가겠습니다", "갈게요"),
    ("오겠습니다", "올게요"),
    ("알겠습니다", "알겠어요"),
    ("모르겠습니다", "잘 모르겠어요"),
    ("했습니다", "했어요"),
    ("하였습니다", "했어요"),
    ("합니다", "해요"),
    ("드립니다", "드려요"),
    ("됩니다", "돼요"),
    ("되었습니다", "됐어요"),
    ("됐습니다", "됐어요"),
    ("있습니다", "있어요"),
    ("없습니다", "없어요"),
    ("같습니다", "같아요"),
    ("싶습니다", "싶어요"),
    ("괜찮습니다", "괜찮아요"),
    ("좋습니다", "좋아요"),
    ("아닙니다", "아니에요"),
    ("맞습니다", "맞아요"),
    ("해드립니까", "해드릴까요"),
    ("드립니까", "드릴까요"),
    ("하겠습니까", "할까요"),
    ("합니까", "해요"),
    ("됩니까", "돼요"),
    ("있습니까", "있어요"),
    ("없습니까", "없어요"),
    ("괜찮습니까", "괜찮아요"),
    ("좋습니까", "좋아요"),
    ("맞습니까", "맞아요"),
)

_COPULA_NOUNS = (
    "거",
    "문장",
    "번역",
    "소리",
    "느낌",
    "시간",
    "부분",
    "곳",
    "중",
    "편",
    "상태",
    "준비",
    "마지막",
    "처음",
    "다음",
    "여기",
    "거기",
    "저기",
)


def _has_final_consonant(text: str) -> bool:
    for char in reversed(text):
        codepoint = ord(char)
        if 0xAC00 <= codepoint <= 0xD7A3:
            return (codepoint - 0xAC00) % 28 != 0
    return False


def _replace_word(text: str, source: str, replacement: str) -> str:
    pattern = re.compile(rf"(?<![가-힣]){re.escape(source)}(?![가-힣])")
    return pattern.sub(replacement, text)


def _replace_formal_endings(text: str) -> str:
    for source, replacement in sorted(_FORMAL_ENDINGS, key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"{re.escape(source)}(?=$|[\s,.!?])", replacement, text)
    return text


def _replace_common_copula(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        noun = match.group(1)
        ending = "이에요" if _has_final_consonant(noun) else "예요"
        return noun + ending

    nouns = "|".join(re.escape(noun) for noun in _COPULA_NOUNS)
    return re.sub(rf"({nouns})입니다(?=$|[\s,.!?])", repl, text)


def _tidy_space(text: str) -> str:
    text = _SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])(?=[^\s,.!?])", r"\1 ", text)
    return text.strip()


def colloquialize_korean_text(text: str) -> str:
    """Rewrite common stiff Korean written forms into polite spoken ASMR text."""
    rewritten = text.strip()
    if not rewritten or not _HANGUL_RE.search(rewritten):
        return rewritten
    for source, replacement in _CONTRACTIONS:
        rewritten = _replace_word(rewritten, source, replacement)
    rewritten = _replace_formal_endings(rewritten)
    rewritten = _replace_common_copula(rewritten)
    return _tidy_space(rewritten)


def colloquialize_korean_translation(translation: KoreanTranslation) -> KoreanTranslation:
    rewritten = colloquialize_korean_text(translation.ko_natural)
    if rewritten == translation.ko_natural:
        return translation
    notes = list(translation.notes)
    if COLLOQUIAL_REWRITE_NOTE not in notes:
        notes.append(COLLOQUIAL_REWRITE_NOTE)
    return translation.model_copy(update={"ko_natural": rewritten, "notes": notes})
