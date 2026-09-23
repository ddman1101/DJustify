#!/usr/bin/env python3
"""Lyrics-aware song selection helpers for the three-song story pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from zhnorm import normalize as _zhnorm


REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = Path(
    os.environ.get("AIDJ_LYRICS_POOL_ROOT", "lyrics")
)
CHORUS_DATABASES = (
    Path(os.environ.get("AIDJ_CHORUS_DB", str(REPO_ROOT / "chorus_db.json"))),
)
GAG_PROMPT_PATH = Path(os.environ.get("AIDJ_GAG_PROMPT", str(Path(__file__).resolve().parent / "prompts" / "lyric_hook_system.txt")))
GAG_LRC_ROOT = Path(os.environ.get("AIDJ_LRC_ROOT", str(POOL_ROOT / "lyrics_lrc")))
LLMCall = Callable[[str, list[dict[str, str]], float, bool], dict[str, Any] | None]


class LyricsStoryError(RuntimeError):
    """A fail-closed error that should be shown directly in the Demo UI."""


@dataclass(frozen=True)
class TimedLine:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SelectionBundle:
    selected_tid: str
    shortlist_tids: tuple[str, ...]
    analyses: dict[str, dict[str, str]]

    def selected_analysis(self) -> dict[str, str]:
        return self.analyses[self.selected_tid]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_text(value: Any, limit: int = 0) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit] if limit > 0 else text


def load_gag_prompt(path: Path = GAG_PROMPT_PATH) -> str:
    if not path.is_file():
        raise LyricsStoryError(f"Cannot find the strict Gag System Prompt: {path}")
    text = path.read_text(encoding="utf-8").strip()
    # The supplied text was once pasted out of an HTML/Python snippet.
    text = re.sub(r"</script>\s*\"\"\"\s*$", "", text).strip()
    text = text.split("[Task Requirements]", 1)[0].strip()
    if not text:
        raise LyricsStoryError(f"Gag System Prompt is empty: {path}")
    return text


COMPACT_HEAD_LINES = 8
COMPACT_HOOK_LINES = 6
COMPACT_TAIL_LINES = 6
_COMPACT_HOOK_MIN_DISTINCT = 4
COMPACT_MAX_CHARS = 1000


def _dedup_key(line: str) -> str:
    return re.sub(r"[\s\W_]+", "", line, flags=re.UNICODE).casefold()


def _trim_to_budget(
    head: list[int], hook: list[int], tail: list[int], lines: list[str], budget: int
) -> set[int]:
    """Drop the least informative lines until the view fits ``budget``."""
    kept = {*head, *hook, *tail}

    def size() -> int:
        return sum(len(lines[index]) + 1 for index in kept)

    while size() > budget:
        if len(head) > 1:
            kept.discard(head.pop())
        elif len(tail) > 1:
            kept.discard(tail.pop(0))
        elif len(hook) > 1:
            kept.discard(hook.pop())
        else:
            break
    return kept


def _render_view(
    lines: list[str], head: list[int], hook: list[int], tail: list[int], budget: int
) -> str:
    kept = {*head, *hook, *tail}
    if sum(len(lines[index]) + 1 for index in kept) > budget:
        kept = _trim_to_budget(list(head), list(hook), list(tail), lines, budget)
    return "\n".join(lines[index] for index in sorted(kept))


def compact_story_view(
    lyrics: str,
    head: int = COMPACT_HEAD_LINES,
    hook: int = COMPACT_HOOK_LINES,
    tail: int = COMPACT_TAIL_LINES,
    max_chars: int = COMPACT_MAX_CHARS,
) -> str:
    """Reduce a full lyric to head + most-repeated block + tail, in song order."""
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines:
        return ""
    # Index of every line's first appearance, so a kept line always resolves to
    # the same position no matter which section selected it.
    first_index: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for index, line in enumerate(lines):
        key = _dedup_key(line)
        if not key:
            continue
        counts[key] += 1
        first_index.setdefault(key, index)
    if not first_index:
        return ""
    unique = sorted(first_index.values())
    if len(unique) <= head + hook + tail:
        middle = len(unique) // 2
        return _render_view(lines, unique[:middle], [], unique[middle:], max_chars)
    head_idx = unique[:head]
    tail_idx = [index for index in unique[-tail:] if index not in set(head_idx)]
    hook_idx: list[int] = []
    repeated = [
        key for key, count in counts.items()
        if count >= 2 and len(set(key)) >= _COMPACT_HOOK_MIN_DISTINCT
    ]
    if repeated:
        # Longest wins a tie on repeat count: a chorus line carries more of the
        # song than the short refrain that usually repeats just as often.
        anchor = max(repeated, key=lambda key: (counts[key], len(key), -first_index[key]))
        seen: set[str] = set()
        for index in range(first_index[anchor], len(lines)):
            key = _dedup_key(lines[index])
            if not key or key in seen:
                continue
            seen.add(key)
            if first_index[key] not in head_idx and first_index[key] not in tail_idx:
                hook_idx.append(first_index[key])
            if len(hook_idx) >= hook:
                break
    return _render_view(lines, head_idx, hook_idx, tail_idx, max_chars)


class LyricCatalog:
    """Read compact, full and timed lyrics under one canonical TID contract."""

    def __init__(
        self,
        pool_root: Path = POOL_ROOT,
        chorus_paths: Sequence[Path] = CHORUS_DATABASES,
        gag_lrc_root: Path = GAG_LRC_ROOT,
    ) -> None:
        self.pool_root = Path(pool_root)
        self.gag_lrc_root = Path(gag_lrc_root)
        self.chorus_path = next((Path(p) for p in chorus_paths if Path(p).is_file()), None)
        if self.chorus_path is None:
            choices = "、".join(str(p) for p in chorus_paths)
            raise LyricsStoryError(f"Cannot find the chorus database; checked: {choices}")
        raw = _read_json(self.chorus_path)
        if not isinstance(raw, dict):
            raise LyricsStoryError(f"Chorus title database format error: {self.chorus_path}")
        self.chorus: dict[str, dict[str, str]] = {
            str(tid): row
            for tid, row in raw.items()
            if isinstance(row, dict)
            and _clean_text(row.get("title"))
            and _clean_text(row.get("chorus_lyrics"))
        }

    def has_full_lyrics(self, tid: str) -> bool:
        return ((self.pool_root / "lyrics_txt" / f"{tid}.txt").is_file()
                or (self.gag_lrc_root / f"{tid}.txt").is_file())

    def has_timed_lyrics(self, tid: str) -> bool:
        return ((self.pool_root / "whisper_json" / f"{tid}.json").is_file()
                or (self.gag_lrc_root / f"{tid}.txt").is_file())

    def has_song(self, tid: str) -> bool:
        return self.has_full_lyrics(tid) and self.has_timed_lyrics(tid)

    def title(self, tid: str) -> str:
        title = _clean_text(self.chorus.get(tid, {}).get("title"))
        if title:
            return title
        label = re.sub(r"^\d+\s+", "", tid)
        label = re.sub(r"\s*\[[^]]+\]\s*$", "", label)
        return _clean_text(label) or tid

    def compact_lyrics(self, tid: str) -> str:
        """The ranking pass' view of a song: head + hook + tail, in song order."""
        try:
            lyrics = self.full_lyrics(tid)
        except LyricsStoryError:
            try:
                lyrics = "\n".join(line.text for line in self.timed_lines(tid))
            except LyricsStoryError:
                return ""
        return compact_story_view(lyrics)

    def full_lyrics(self, tid: str, max_chars: int = 6000) -> str:
        path = self.pool_root / "lyrics_txt" / f"{tid}.txt"
        if not path.is_file():
            path = self.gag_lrc_root / f"{tid}.txt"
        if not path.is_file():
            raise LyricsStoryError(f"Incomplete lyrics missing: {tid}")
        text = path.read_text(encoding="utf-8").strip()
        if path.parent == self.gag_lrc_root:
            text = "\n".join(line.text for line in self._read_lrc(path))
        if not text:
            raise LyricsStoryError(f"Full lyrics is empty: {tid}")
        return text[:max_chars]

    def timed_lines(self, tid: str) -> list[TimedLine]:
        path = self.pool_root / "whisper_json" / f"{tid}.json"
        if not path.is_file():
            lrc_path = self.gag_lrc_root / f"{tid}.txt"
            if lrc_path.is_file():
                return self._read_lrc(lrc_path)
            raise LyricsStoryError(f"Time-aligned lyrics missing: {tid}")
        raw = _read_json(path)
        lines: list[TimedLine] = []
        for segment in raw.get("segments", []):
            if not isinstance(segment, dict):
                continue
            text = _clean_text(segment.get("text"))
            if not text:
                continue
            try:
                start = max(0.0, float(segment.get("start", 0)))
                end = max(start, float(segment.get("end", start)))
            except (TypeError, ValueError):
                continue
            lines.append(TimedLine(start, end, text))
        return lines

    @staticmethod
    def _read_lrc(path: Path) -> list[TimedLine]:
        stamped: list[tuple[float, str]] = []
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$", raw_line.strip())
            if not match:
                continue
            text = _clean_text(match.group(3))
            if text:
                stamped.append((int(match.group(1)) * 60 + float(match.group(2)), text))
        lines: list[TimedLine] = []
        for index, (start, text) in enumerate(stamped):
            end = stamped[index + 1][0] if index + 1 < len(stamped) else start + 4.0
            lines.append(TimedLine(start, max(start, end), text))
        return lines

    def lyrics_between(
        self,
        tid: str,
        start: float,
        end: float,
        max_lines: int = 30,
        max_chars: int = 4000,
    ) -> str:
        if end <= start:
            return ""
        selected = [
            line.text
            for line in self.timed_lines(tid)
            if line.end > start and line.start < end
        ]
        text = "\n".join(selected[:max_lines]).strip()
        return text[:max_chars]

    def eligible_registry_tids(
        self,
        registry: dict[str, dict[str, Any]],
        quarantined: Iterable[str] = (),
        suspect: Iterable[str] = (),
        in_pool: Callable[[str], bool] | None = None,
    ) -> list[str]:
        blocked = set(map(str, quarantined)) | set(map(str, suspect))
        allowed: list[str] = []
        for tid, card in registry.items():
            tid = str(tid)
            if tid in blocked or (in_pool is not None and not in_pool(tid)):
                continue
            if not isinstance(card, dict) or not card.get("bpm") or not card.get("segments"):
                continue
            if self.has_song(tid):
                allowed.append(tid)
        return sorted(allowed, key=str.casefold)


def make_story_acts(story: str, curve: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(curve) != 3:
        raise LyricsStoryError(f"A 3-song set requires 3 story stages, currently got {len(curve)}")
    story_body = re.sub(r"^\s*Set\s*\d*\s*[:：]\s*", "", story, flags=re.I)
    clauses = [
        _clean_text(piece, 120)
        for piece in re.split(r"\s*(?:,|，|;|；|→|->|⇒)\s*", story_body)
        if _clean_text(piece)
    ]
    descriptions: list[str] = []
    acts: list[dict[str, Any]] = []
    if len(clauses) >= 3:
        quotient, remainder = divmod(len(clauses), 3)
        cursor = 0
        for index in range(3):
            size = quotient + (1 if index < remainder else 0)
            descriptions.append("，".join(clauses[cursor : cursor + size]))
            cursor += size
    for index, segment in enumerate(curve):
        description = (
            descriptions[index]
            if len(descriptions) == 3
            else _clean_text(segment.get("description"), 160)
        )
        if not description:
            raise LyricsStoryError(f"story Act {index + 1} lacks text description")
        acts.append(
            {
                "act": index + 1,
                "description": description,
                "story": story_body.strip(),
                "V": segment.get("V"),
                "A": segment.get("A"),
            }
        )
    return acts


ENERGY_CHOICES = ("up", "peak", "down", "calm", "neutral")


def act_energy(description: str) -> str:
    """How this act's energy was read when the act was first analysed."""
    row = load_act_themes().get(act_theme_key(description)) or {}
    value = str(row.get("energy", "")).strip().lower()
    return value if value in ENERGY_CHOICES else "neutral"


def act_demands_peak_energy(description: str) -> bool:
    """True when the act names the peak of the arc, not the climb toward it."""
    return act_energy(description) == "peak"


def act_energy_direction(description: str) -> str | None:
    """"up" / "down" / None -- which way this act's arousal should lean."""
    energy = act_energy(description)
    if energy in ("up", "peak"):
        return "up"
    if energy in ("down", "calm"):
        return "down"
    return None


def energy_rank_key(description: str, arousal: Any) -> float:
    """Sort weight for this act's energy direction; 0.0 when the act is neutral."""
    direction = act_energy_direction(description)
    if direction is None:
        return 0.0
    try:
        value = float(arousal)
    except (TypeError, ValueError):
        return 0.0
    return value if direction == "up" else -value


def folded_bpm_compatible(bpm_a: Any, bpm_b: Any, tolerance: float = 0.08) -> bool:
    try:
        ratio = float(bpm_b) / float(bpm_a)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    while ratio > 1.5:
        ratio /= 2
    while ratio < 0.75:
        ratio *= 2
    return abs(ratio - 1.0) <= tolerance
_RELATION_CHOICES = ("romantic", "friends or crew", "family", "self or striving", "other")
REQUIRE_DIRECT_FIT = os.environ.get("AIDJ_REQUIRE_DIRECT_FIT", "0") == "1"
THEME_CHOICES = (
    "love",        # in love, courtship, infatuation -- the relationship is alive
    "breakup",        # it ended, or it is breaking: fights, betrayal, heartbreak
    "longing",        # the other person is gone; longing and regret after the fact
    "party",     # partying, clubbing, drinking, celebration
    "friendship",        # friends, crew, brothers -- being beside someone
    "striving",     # striving, the come-up, refusing to give up
    "family life",     # family, parents, home, growing up
    "society",     # society, money, the street, status, politics
    "loneliness",     # alone, lost, numb -- with no other person in the frame
    "other",
)

RELATION_CHOICES_THEME = (
    "romantic", "friends or crew", "family", "self or striving",
    "strangers or a crowd", "no named subject",
)

_LYRICS_USABLE: dict[str, bool] = {}
MIN_DISTINCT_LYRIC_LINES = 4
MIN_DISTINCT_LYRIC_CHARS = 40

_LYRIC_SCRIPT_RANGES = (
    (0x0020, 0x024F),
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x2E80, 0x9FFF),   # CJK radicals through unified ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0x3000, 0x30FF),   # CJK punctuation, hiragana, katakana
    (0xFF00, 0xFFEF),   # halfwidth/fullwidth forms
)
MAX_OFF_SCRIPT_RATIO = 0.05


def _off_script_ratio(text: str) -> float:
    """Share of non-space characters outside the pool's four writing systems."""
    body = [ch for ch in text if not ch.isspace()]
    if not body:
        return 1.0
    inside = sum(1 for ch in body
                 if any(lo <= ord(ch) <= hi for lo, hi in _LYRIC_SCRIPT_RANGES))
    return 1.0 - inside / len(body)


def lyrics_are_usable(tid: str, catalog: "LyricCatalog | None" = None) -> bool:
    if tid in _LYRICS_USABLE:
        return _LYRICS_USABLE[tid]
    try:
        text = (catalog or LyricCatalog()).full_lyrics(tid)
    except Exception:
        _LYRICS_USABLE[tid] = False
        return False
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    ok = (len(lines) >= MIN_DISTINCT_LYRIC_LINES
          and len("".join(lines)) >= MIN_DISTINCT_LYRIC_CHARS
          and _off_script_ratio("".join(lines)) <= MAX_OFF_SCRIPT_RATIO)
    _LYRICS_USABLE[tid] = ok
    return ok


def act_theme_store_path() -> str:
    return os.environ.get(
        "AIDJ_ACT_THEMES",
        str(REPO_ROOT / "reports" / "act_themes.json"),
    )


_ACT_THEME_STORE: dict[str, dict[str, Any]] | None = None


def act_theme_key(description: str) -> str:
    return hashlib.sha256(
        _normalized_evidence(description).encode("utf-8")).hexdigest()[:16]


def load_act_themes(refresh: bool = False) -> dict[str, dict[str, Any]]:
    global _ACT_THEME_STORE
    if _ACT_THEME_STORE is None or refresh:
        try:
            with open(act_theme_store_path(), encoding="utf-8") as handle:
                _ACT_THEME_STORE = json.load(handle)
        except Exception:
            _ACT_THEME_STORE = {}
    return _ACT_THEME_STORE


def build_act_theme_prompt(description: str, story: str) -> tuple[str, str]:
    """Ask what a song must be about to belong in this act."""
    system = (
        "You are setting song selection criteria for a DJ set. The user will give you the full story, and one scene.\n"
        "Question: for a song to belong in this act, which theme must its lyrics have at the very least?\n"
        "Rules:\n"
        "1. Only fill in themes that absolutely cannot be used if this is not the theme. Prefer filling in only 1."
        "For each additional theme filled in, a whole batch of titles unrelated to this act will be included.\n"
        "2. Do not fill theme just because it might appear in this scene."
        "Ask yourself: Would a song with only that theme and absolutely no previous theme feel weird in this act?"
        "If yes, leave blank.\n"
        "3. required_relations: Only fill in when this act explicitly specifies character relationships."
        "Otherwise leave the array empty. Side by side, brothers, the two of us together mean friends or crew; "
        "falling in love, a crush and a breakup mean romantic; "
        "a party or a celebration says nothing about who with, so leave it blank.\n"
        "4. If the act only describes a change of energy or pace, such as building all the way up, "
        "Read it in the context of the full story to see what precedes and follows, then decide the theme; do not approve everything just because it is abstract.\n"
        "5. energy: fill in the musical energy of this act: peak = this act is the highest point of the entire set (rave, explosion, party);"
        "up=energy climbing but not yet at the peak; calm=quiet, restrained, concluding; down=emotionally low but volume not necessarily small;"
        "neutral=this act has no energy specified.\n"
        "Example:\n"
        "  An act that opens on the noise of a party gives themes=[\"party\"], required_relations=[]\n"
        "  An act about being deep in love gives themes=[\"love\"], required_relations=[\"romantic\"]\n"
        "  An act about friends grinding side by side gives themes=[\"friendship\",\"striving\"], "
        "required_relations=[\"friends or crew\"]\n"
        f"Themes to choose from: {', '.join(THEME_CHOICES)}.\n"
        f"Energy to choose from: {', '.join(ENERGY_CHOICES)}.\n"
        f"Relations to choose from: {', '.join(RELATION_CHOICES_THEME)}."
    )
    user = json.dumps({"full story": story, "this act": description},
                      ensure_ascii=False)
    return system, user


def build_act_relation_prompt(description: str) -> tuple[str, str]:
    """Ask, about this act's own words only, whether it names a relationship."""
    system = (
        "You are setting song selection criteria for a DJ set. Only look at the text of this scene sentence, answer: Does this sentence itself have"
        "Does the named song describe a relationship between who and whom?"
        "Only an explicit subject counts, such as side by side, brothers, falling in love, or a breakup. "
        "A crowd or everyone is not a named relationship.\n"
        "If not specified, fill required_relations as an empty array.\n"
        "themes fill in: for a song to be placed in this act, the themes must at least include which ones (max 3, fewer is better).\n"
        f"Themes to choose from: {', '.join(THEME_CHOICES)}.\n"
        f"Energy to choose from: {', '.join(ENERGY_CHOICES)}.\n"
        f"Relations to choose from: {', '.join(RELATION_CHOICES_THEME)}."
    )
    return system, json.dumps({"this act": description}, ensure_ascii=False)


def act_allowed_themes(description: str
                       ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """(allowed themes, required relations), or None if the act constrains neither."""
    cached = load_act_themes().get(act_theme_key(description))
    if cached:
        themes = tuple(str(t) for t in (cached.get("themes") or ()))
        relations = tuple(str(r) for r in (cached.get("required_relations") or ()))
        if themes:
            return themes, relations
    return None


_THEME_STORE: dict[str, dict[str, str]] | None = None


def _theme_store_path() -> str:
    return os.environ.get(
        "AIDJ_SONG_THEMES",
        str(REPO_ROOT / "reports" / "song_themes.json"),
    )


def load_song_themes(refresh: bool = False) -> dict[str, dict[str, str]]:
    """The blind per-song theme verdicts, read from disk once."""
    global _THEME_STORE
    if _THEME_STORE is None or refresh:
        try:
            with open(_theme_store_path(), encoding="utf-8") as handle:
                _THEME_STORE = json.load(handle)
        except Exception:
            _THEME_STORE = {}
    return _THEME_STORE


def song_theme(tid: str) -> dict[str, str] | None:
    return load_song_themes().get(tid)
def theme_conflicts_with_act(description: str, tid: str) -> str | None:
    """Why this song's subject does not belong to this act, or None if it does."""
    rule = act_allowed_themes(description)
    if rule is None:
        return None
    allowed, relations = rule
    row = song_theme(tid)
    if not row:
        return None
    themes = [str(t) for t in (row.get("themes") or [])]
    lead = themes[0] if themes else ""
    if lead not in allowed:
        if any(t in allowed for t in themes):
            return (f"this song is mainly about {lead} "
                    f"({row.get('summary', '')[:40]}); "
                    f"{' / '.join(allowed)} is only a secondary colour in it, "
                    f"and this act needs a song that is about it")
        return (f"this song is about {' / '.join(themes)} "
                f"({row.get('summary', '')[:40]}), "
                f"and this act needs {' / '.join(allowed)}")
    if relations and str(row.get("relation", "")) not in relations:
        return (f"this song is about {row.get('relation')} "
                f"({row.get('summary', '')[:40]}), "
                f"and this act needs {' / '.join(relations)}")
    return None


def build_theme_prompt(lyrics: str) -> tuple[str, str]:
    """The blind question: what is this song about?  No act, no story, no hint."""
    system = (
        "You analyse lyrics in Mandarin and English pop."
        " Judge what the song as a whole is about from the full lyrics alone."
        " Do not guess at a use for it and do not favour any expected answer; describe only the lyrics."
        " A song usually belongs to two or three themes at once, so list them from the main one down rather than picking a single label."
        " Keep these distinctions strictly: love is a relationship that is alive, the crush, the courting, the sweetness;"
        " breakup is a relationship breaking or broken, the arguments, the betrayal, the heartbreak, the goodbye;"
        " longing is when the other person is already gone, the missing and the regret afterwards;"
        " friendship is between friends, brothers or crew, not lovers;"
        " loneliness is only for lyrics with no other person in them, just the singer's own state."
        " If the lyrics are fragmentary, repetitive or unreadable, set themes to other."
    )
    user = json.dumps({
        "task": "Read the full lyrics and determine what it is about.",
        "themes": (f"Select 1 to 3 from these categories, ranked from primary to secondary: {list(THEME_CHOICES)}."
                   "For example, a song about friends partying at a club should be [friendship, party]."),
        "summary": "Describe what this song is about in one sentence (within 40 characters)",
        "relation": f"lyrics main character relationships, choose one from these categories: {list(RELATION_CHOICES_THEME)}",
        "evidence": "Copy verbatim one lyrics sentence that best represents the theme",
        "lyrics": lyrics[:6000],
    }, ensure_ascii=False)
    return system, user


def build_act_subject_prompt(description: str, lyrics: str) -> tuple[str, str]:
    """Ask the one question that matters, about one act, with the full lyrics."""
    system = (
        "You analyse lyrics in Mandarin and English pop."
        " You are given the full lyrics of a song and a description of a theme."
        " Answer only this: is the song as a whole about that theme?"
        " Judge by the core content of the whole song, not by words that appear in passing:"
        " a song about rapping is not about friendship just because the word friend appears in it,"
        " and a song about first love is not about striving side by side just because the word brother appears."
        " If it only touches on the theme, or only a line or two relate to it, answer partial."
        " If the lyrics are fragmentary, repetitive or unreadable, answer no."
        " Do not accommodate, and do not find excuses for the song."
    )
    user = json.dumps({
        "Theme": description,
        "Issue": "Is this song overall about the above theme?",
        "verdict": "Yes / Part / No",
        "reason": "one-sentence reason (within 40 chars), must be based on lyrics content",
        "evidence": "Copy a lyrics sentence verbatim to support your judgment; if the judgment is negative, copy the sentence that best reveals the true theme of this song",
        "lyrics": lyrics[:6000],
    }, ensure_ascii=False)
    return system, user


ACT_SUBJECT_ACCEPT = tuple(
    v.strip() for v in os.environ.get("AIDJ_ACT_SUBJECT_ACCEPT", "is").split(",")
    if v.strip()
)


def act_subject_verdict(description: str, tid: str, catalog: "LyricCatalog",
                        llm_call: "LLMCall") -> dict[str, str] | None:
    """Does this song's subject match this act?  None when unanswerable."""
    if not description:
        return None
    try:
        lyrics = catalog.full_lyrics(tid)
    except LyricsStoryError:
        return None
    if not lyrics.strip():
        return None
    system, user = build_act_subject_prompt(description, lyrics)
    try:
        row = llm_call("act subject", [{"role": "system", "content": system},
                                       {"role": "user", "content": user}], 0.0, False)
    except Exception:
        return None
    if not isinstance(row, dict) or "verdict" not in row:
        return None
    return {
        "verdict": str(row.get("verdict", "")),
        "reason": str(row.get("reason", "")),
        "evidence": str(row.get("evidence", "")),
    }
def set2_viable_openers(
    tids: Sequence[str], registry: dict[str, dict[str, Any]]
) -> list[str]:
    """Return A candidates that have at least one legal A→B→C BPM path."""
    viable: list[str] = []
    for tid_a in tids:
        neighbors_b = [
            tid_b for tid_b in tids
            if tid_b != tid_a
            and folded_bpm_compatible(registry[tid_a].get("bpm"), registry[tid_b].get("bpm"))
        ]
        if any(
            tid_c not in (tid_a, tid_b)
            and folded_bpm_compatible(registry[tid_b].get("bpm"), registry[tid_c].get("bpm"))
            for tid_b in neighbors_b
            for tid_c in tids
        ):
            viable.append(tid_a)
    return viable


def _candidate_payload(
    tids: Iterable[str],
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    full: bool,
    full_max_chars: int = 2600,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid in tids:
        card = registry[tid]
        compact_tid = tid.split(" ", 1)[0]
        rows.append(
            {
                "tid": compact_tid,
                "title": catalog.title(tid) or _clean_text(card.get("title")) or tid,
                "bpm": card.get("bpm"),
                "valence_q": card.get("valence"),
                "arousal_q": card.get("arousal"),
                "lyrics": (
                    catalog.full_lyrics(tid, max_chars=full_max_chars)
                    if full else catalog.compact_lyrics(tid)
                ),
            }
        )
    return rows


def _require_dict(result: dict[str, Any] | None, stage: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise LyricsStoryError(f"{stage}: Qwen did not return parseable JSON")
    return result


def _canonical_tid(value: Any, allowed: Iterable[str]) -> str | None:
    raw = str(value or "").strip()
    allowed_list = list(map(str, allowed))
    if raw in allowed_list:
        return raw
    # Compatibility with the existing UI/model convention that returns the
    # numeric prefix, while still resolving strictly inside this allowlist.
    matches = [tid for tid in allowed_list if tid.split(" ", 1)[0] == raw]
    if len(matches) == 1:
        return matches[0]
    matches = [tid for tid in allowed_list if tid.rsplit(" [", 1)[0] == raw]
    return matches[0] if len(matches) == 1 else None


def _analysis_row(row: dict[str, Any], tid: str) -> dict[str, str]:
    return {
        "storytelling_analysis": _clean_text(row.get("storytelling_analysis"), 300)
        or "model returned no analysis",
        "gag_analysis": _clean_text(row.get("gag_analysis"), 300) or "None",
        "gag_confidence": _clean_text(row.get("gag_confidence"), 10) or "None",
        "story_fit": _clean_text(row.get("story_fit"), 10) or "unverified",
        "story_evidence": _clean_text(row.get("story_evidence"), 160),
        "relation": _clean_text(row.get("relation"), 20),
    }


def _normalized_evidence(text: Any) -> str:
    """Canonical matching form, shared with the GAG retrieval stage."""
    value = _clean_text(text)
    value = value.strip("「」『』\"' `，。！？；：,.!?;:")
    return _zhnorm(value)


def story_evidence_is_grounded(evidence: Any, lyrics: str) -> bool:
    """Require a short, verbatim lyric span instead of model-made rationale."""
    needle = _normalized_evidence(evidence)
    haystack = _normalized_evidence(lyrics)
    return len(needle) >= 2 and needle in haystack


def _canonical_grounded_evidence(evidence: Any, lyrics: str) -> str | None:
    """Return an actual source lyric line contained in the model's quote."""
    cleaned = _clean_text(evidence, 160)
    if story_evidence_is_grounded(cleaned, lyrics):
        return cleaned
    quoted = _normalized_evidence(cleaned)
    matches: list[str] = []
    for raw_line in lyrics.splitlines():
        line = _clean_text(raw_line, 160)
        normalized = _normalized_evidence(line)
        if len(normalized) >= 4 and normalized in quoted:
            matches.append(line)
    return max(matches, key=lambda value: len(_normalized_evidence(value))) if matches else None


def require_grounded_story_analysis(
    row: dict[str, Any], tid: str, catalog: LyricCatalog, stage: str
) -> dict[str, str]:
    analysis = _analysis_row(row, tid)
    if analysis["story_fit"] not in ("direct fit", "partial fit"):
        raise LyricsStoryError(f"{stage}: {catalog.title(tid)} was judged as no fitstory for this act")
    # Evidence may come from either the representative span or the full text.
    lyrics = catalog.compact_lyrics(tid) + "\n" + catalog.full_lyrics(tid)
    grounded = _canonical_grounded_evidence(analysis["story_evidence"], lyrics)
    if grounded is None:
        raise LyricsStoryError(
            f"{stage}: story_evidence of {catalog.title(tid)} not lyrics original"
        )
    analysis["story_evidence"] = grounded
    return analysis


def _parse_ranked_rows(
    raw_rows: Any,
    allowed: Sequence[str],
) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not isinstance(raw_rows, list):
        return [], {}
    tids: list[str] = []
    analyses: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        tid = _canonical_tid(
            row.get("tid", row.get("song_id", row.get("selected_song_id"))), allowed
        )
        if tid is None or tid in tids:
            continue
        tids.append(tid)
        analyses[tid] = _analysis_row(row, tid)
    return tids, analyses


def _parse_ranked_tid_list(raw_tids: Any, allowed: Sequence[str]) -> list[str]:
    """Parse a compact ordered TID list under the same strict allowlist."""
    if not isinstance(raw_tids, list):
        return []
    tids: list[str] = []
    for value in raw_tids:
        tid = _canonical_tid(value, allowed)
        if tid is not None and tid not in tids:
            tids.append(tid)
    return tids


def _score_story_candidates(
    act: dict[str, Any],
    tids: Sequence[str],
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    *,
    full: bool,
    tag: str,
    history_context: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Have Qwen assess every song, then let code compute the ordering."""
    allowed = [tid for tid in tids if tid in registry and catalog.has_song(tid)]
    chunk = 1 if full else 5
    if len(allowed) > chunk:
        merged_analyses: dict[str, dict[str, str]] = {}
        for offset in range(0, len(allowed), chunk):
            _, chunk_analyses = _score_story_candidates(
                act, allowed[offset : offset + chunk], catalog, registry, llm_call,
                full=full, tag=tag, history_context=history_context,
            )
            merged_analyses.update(chunk_analyses)
        merged_order = sorted(
            allowed,
            key=lambda tid: int(merged_analyses[tid].get("_story_score", "-999")),
            reverse=True,
        )
        return merged_order, merged_analyses
    payload = {
        "current_act_only": {
            "act": act.get("act"),
            "description": act.get("description"),
        },
        # Context for *reading* the clause, never for widening the verdict.
        "Full story (used only to understand what this act is about, not for scoring)": act.get("story", ""),
        "history_context": history_context or {},
        "task": (
            "Score each song independently with exactly one row per song; do not select a champion first and then provide reasons."
            "Read the full story first to place this act in it and see which way it moves: an act about building up, "
            "followed by an act about a party, means the energy climbs rather than the mood sinking. "
            "Re-score only for current_act_only."
            "relation: decide first what relationship the song mainly describes, using only "
            + "/".join(_RELATION_CHOICES) + "。"
            "Judge from what the lyrics actually describe; a single word such as friend or brother does not make it a friendship song. "
            "『friend of friend』『dont rush to say you love me』overall is romantic."
            "subject_score: whether character/relationship themes match; event_score: whether lyrics actual events describe this scene;"
            "tone_score: whether the emotional direction is the same. Each item: 0=mismatch, 1=partial, 2=clear."
            "If lyrics main events contradict this act, contradiction=true; e.g., insults, betrayal, arguments in a sweet act,"
            "Or a simple party/career flex in a breakup scene, or a confession, romance, or ambiguity in a friendship/inspirational scene."
            "Shared words and song title associations do not count as semantic evidence."
            "story_evidence must copy a single continuous original sentence from the candidate lyrics verbatim; if no fit, use an empty string."
        ),
        "candidates": _candidate_payload(
            allowed, catalog, registry, full=full, full_max_chars=2600
        ),
        "output_schema": {
            "candidate_scores": [
                {
                    "tid": "candidate tid",
                    "relation": "/".join(_RELATION_CHOICES),
                    "subject_score": "0/1/2",
                    "event_score": "0/1/2",
                    "tone_score": "0/1/2",
                    "contradiction": "true/false",
                    "story_fit": "direct fit / partial fit / no fit",
                    "story_evidence": "A verbatim original sentence from candidate lyrics; no fit can be left blank",
                }
            ]
        },
    }
    result = _require_dict(
        llm_call(
            tag,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict auditor of lyric meaning. Use the full story to understand what this "
                        "act means and which way its energy goes, but score only current_act_only, never another "
                        "act, and never let a song title stand in for its lyrics. Output only JSON following the "
                        "user's output_schema."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            0.1,
            True,
        ),
        tag,
    )
    raw_rows = result.get("candidate_scores")
    if not isinstance(raw_rows, list):
        raise LyricsStoryError(f"{tag}: Qwen did not return candidate_scores")
    scored: list[tuple[int, int, str]] = []
    analyses: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for position, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            continue
        tid = _canonical_tid(row.get("tid"), allowed)
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        try:
            subject = max(0, min(2, int(row.get("subject_score", 0))))
            event = max(0, min(2, int(row.get("event_score", 0))))
            tone = max(0, min(2, int(row.get("tone_score", 0))))
        except (TypeError, ValueError):
            subject = event = tone = 0
        contradiction = bool(row.get("contradiction"))
        analysis = _analysis_row(row, tid)
        positive = analysis["story_fit"] in ("direct fit", "partial fit")
        if positive:
            try:
                analysis = require_grounded_story_analysis(analysis, tid, catalog, tag)
            except LyricsStoryError:
                subject = event = tone = 0
                contradiction = True
                analysis["story_fit"] = "no fit"
                analysis["story_evidence"] = ""
            if arousal_conflicts_with_act(str(act.get("description", "")),
                                          (registry.get(tid) or {}).get("arousal")):
                contradiction = True
                analysis["story_fit"] = "no fit"
                analysis["storytelling_analysis"] += (
                    "this act requires burst high energy but song arousal below threshold"
                )
        # Subject is foundational, then actual event, then emotional tone.
        # A contradiction dominates any generous component score.
        total = subject * 4 + event * 3 + tone * 2 - (20 if contradiction else 0)
        if not positive:
            total -= 8
        analysis["storytelling_analysis"] = (
            f"Characters/theme {subject}/2, events {event}/2, mood {tone}/2"
            + ("; contradicts this act" if contradiction else "")
        )
        analysis["gag_analysis"] = "None"
        analysis["gag_confidence"] = "None"
        analysis["_story_score"] = str(total)
        analysis["_component_score"] = str(total)
        analyses[tid] = analysis
        scored.append((total, -position, tid))
    if len(seen) != len(allowed):
        raise LyricsStoryError(f"{tag}: Only evaluated {len(seen)}/{len(allowed)} candidates")
    if full:
        for tid in allowed:
            if analyses[tid]["story_fit"] not in ("direct fit", "partial fit"):
                continue
            description = str(act.get("description", ""))
            full_lyrics = catalog.full_lyrics(tid)
        scored = [
            (int(analyses[tid]["_story_score"]), -position, tid)
            for position, tid in enumerate(allowed)
        ]
        def _verify(evidence_by_tid: dict[str, str]) -> dict[str, bool]:
            verify_payload = {
                "current_act_only": {
                    "act": act.get("act"),
                    "description": act.get("description"),
                },
                "Full story (used only to understand the direction of this act)": act.get("story", ""),
                "task": (
                    "Evaluate each row to determine if the evidence sentence itself matches the situation described by current_act."
                    "An act often has several stages in it, such as a rift turning into a breakup.，"
                    "If this sentence supports any one of the stages, it counts as true; it does not need to cover all stages simultaneously."
                    "The criterion is the semantics of this sentence, not the song title or common words:"
                    "Only positive, high-energy, party does not mean love; only sadness does not mean breakup;"
                    "Only shared words with different contexts count as false; also false if it clearly contradicts this act."
                ),
                "claims": [{"tid": tid.split(" ", 1)[0], "evidence": evidence}
                           for tid, evidence in evidence_by_tid.items()],
                "output_schema": {
                    "verdicts": [{
                        "tid": "claim tid",
                        "evidence_entails_act": "true/false",
                        "reason": "Max 20 characters",
                    }]
                },
            }
            verified = _require_dict(
                llm_call(
                    "evidence check",
                    [
                        {"role": "system", "content": (
                            "You are an independent auditor of natural-language inference. Do not look at song "
                            "titles and do not add anything the quoted line does not say; judge only whether the "
                            "line itself entails this act. Output only JSON following the user's output_schema."
                        )},
                        {"role": "user", "content": json.dumps(verify_payload, ensure_ascii=False)},
                    ],
                    0.1,
                    True,
                ),
                "evidence check",
            )
            rows = verified.get("verdicts")
            out: dict[str, bool] = {}
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    resolved = _canonical_tid(row.get("tid"), allowed)
                    if resolved is not None:
                        out[resolved] = row.get("evidence_entails_act") is True
            return out

        claim_tids = [
            tid for tid in allowed
            if analyses[tid]["story_fit"] in ("direct fit", "partial fit")
            and analyses[tid]["story_evidence"]
        ]
        if claim_tids:
            primary = {tid: analyses[tid]["story_evidence"] for tid in claim_tids}
            verdicts = _verify(primary)
            if set(verdicts) != set(claim_tids):
                raise LyricsStoryError(
                    f"evidence check only returns {len(verdicts)}/{len(claim_tids)} tracks"
                )
            description = str(act.get("description", ""))
            retry = {}
            for tid in claim_tids:
                if verdicts[tid]:
                    continue
            for tid in claim_tids:
                if not verdicts[tid]:
                    analyses[tid]["story_fit"] = "no fit"
                    analyses[tid]["storytelling_analysis"] += "; evidence does not imply this act"
                    analyses[tid]["_story_score"] = "-999"
            scored = [
                (int(analyses[tid]["_story_score"]), -position, tid)
                for position, tid in enumerate(allowed)
            ]
    if (len(allowed) > 1 and len({row[0] for row in scored}) == 1
            and scored[0][0] >= 0):
        isolated: dict[str, dict[str, str]] = {}
        for tid in allowed:
            _, one = _score_story_candidates(
                act, [tid], catalog, registry, llm_call,
                full=full, tag=tag, history_context=history_context,
            )
            isolated.update(one)
        order = sorted(
            allowed,
            key=lambda tid: int(isolated[tid].get("_story_score", "-999")),
            reverse=True,
        )
        return order, isolated
    scored.sort(reverse=True)
    return [tid for _, _, tid in scored], analyses


def _opening_system(gag_prompt: str) -> str:
    return (
        "You are the music director choosing the opening track of a DJ set.\n"
        "The opening track has nothing before it, so there is no lyric hook to judge: choose on act 1 and the "
        "candidate lyrics alone. Fitting the story is a requirement, and a similar title, one shared word or a "
        "vague claim of an emotional turn does not count. story_evidence must be a continuous short line copied "
        "verbatim from the candidate lyrics that supports the verdict; with no such evidence, mark it as no fit "
        "rather than inventing a reason. Never choose a TID that is not in the candidate list. Follow the single "
        "output_schema the user gives you and output JSON only."
    )


def _gag_stage_system(gag_prompt: str, contract: str) -> str:
    return (
        "【Decision Priority】First check if candidate lyrics are direct fit for current_act; Storytelling is mandatory."
        "Only use Gag as a sorting bonus when both songs fit the story. Do not choose a song for a literal pun if it does not fit the story."
        "story_evidence must copy a continuous short sentence from candidate.lyrics verbatim; sharing only common words but with different context does not count.\n\n"
        + gag_prompt
        + "\n\n[Only output rule for this session]\n"
        + contract
        + "\nOutput only a single JSON object. No Markdown, tables, code fences, preamble, postamble, or reasoning process."
    )


def select_opening_song(
    story: str,
    act: dict[str, Any],
    eligible_tids: Sequence[str],
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    gag_prompt: str,
    batch_size: int = 30,
) -> SelectionBundle:
    if not eligible_tids:
        raise LyricsStoryError("A title valid set is empty (A.REG and lyrics library have no available intersection)")

    representative_order, representative_analyses = _score_story_candidates(
        act, list(eligible_tids), catalog, registry, llm_call,
        full=False, tag="opening act scoring",
    )
    nominee_cap = max(3, int(os.environ.get("AIDJ_OPENER_NOMINEES", "15")))
    nominees = representative_order[: min(nominee_cap, len(representative_order))]
    full_order, full_analyses = _score_story_candidates(
        act, nominees, catalog, registry, llm_call,
        full=True, tag="opening act scoring",
    )
    def _opener_key(tid: str) -> tuple[int, int]:
        analysis = full_analyses[tid]
        passed = int(analysis.get("story_fit") in ("direct fit", "partial fit"))
        return passed, int(analysis.get("_story_score", "-999"))

    full_order = sorted(full_order, key=_opener_key, reverse=True)
    selected = next(
        (tid for tid in full_order
         if full_analyses[tid]["story_fit"] in ("direct fit", "partial fit")),
        None,
    )
    if selected is None:
        # Lyrics order the pool, they never abort it: with no candidate the
        # judge accepts, take its best-scoring one and label it honestly.
        full_order = sorted(
            full_order,
            key=lambda tid: int(full_analyses[tid].get("_component_score", "-999")),
            reverse=True,
        )
        selected = full_order[0]
        for tid in full_order:
            full_analyses[tid]["storytelling_analysis"] += "; no strong lyrics match for Act 1, selected by relative rank"
            full_analyses[tid]["story_fit"] = "no fit (best available)"
    analyses = dict(representative_analyses)
    analyses.update(full_analyses)
    analyses[selected]["gag_analysis"] = "Does not apply: A is the opening title, with no previous title to form a transition joke."
    analyses[selected]["gag_confidence"] = "None"
    ordered = [selected] + [tid for tid in full_order if tid != selected]
    ordered += [tid for tid in nominees if tid not in ordered]
    return SelectionBundle(selected, tuple(ordered), analyses)


def select_lyric_shortlist(
    story: str,
    act: dict[str, Any],
    history_context: dict[str, Any],
    musical_tids: Sequence[str],
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    gag_prompt: str,
) -> SelectionBundle:
    """Backward-compatible Top-5 helper used by older callers/tests."""
    return _select_representative_ranking(
        story, act, history_context, musical_tids, catalog, registry,
        llm_call, gag_prompt, limit=5, tag="lyric top 5",
    )


def select_lyric_ranking(
    story: str,
    act: dict[str, Any],
    history_context: dict[str, Any],
    musical_tids: Sequence[str],
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    gag_prompt: str,
) -> SelectionBundle:
    """Rank the complete BPM-qualified pool in one representative-lyrics call."""
    return _select_representative_ranking(
        story, act, history_context, musical_tids, catalog, registry,
        llm_call, gag_prompt, limit=None, tag="full lyric ranking",
    )


def _select_representative_ranking(
    story: str,
    act: dict[str, Any],
    history_context: dict[str, Any],
    musical_tids: Sequence[str],
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    gag_prompt: str,
    limit: int | None,
    tag: str,
) -> SelectionBundle:
    allowed = [tid for tid in musical_tids if tid in registry and catalog.has_song(tid)]
    if not allowed:
        raise LyricsStoryError("No candidate titles after intersecting the qualified BPM pool with A.REG/lyrics library")
    wanted = len(allowed) if limit is None else min(limit, len(allowed))
    scored_tids, scored_analyses = _score_story_candidates(
        act, allowed, catalog, registry, llm_call,
        full=False, tag="lyricsstory score per track", history_context=history_context,
    )
    scored_tids = scored_tids[:wanted]
    return SelectionBundle(
        scored_tids[0], tuple(scored_tids),
        {tid: scored_analyses[tid] for tid in scored_tids},
    )

    payload = {
        "current_act": act,
        "history_context": history_context,
        "task": f"Based solely on the candidate representative lyrics, fully rank the {wanted} songs from most suitable to least suitable; do not omit any titles.",
        "hard_rule": (
            "Only use tids within candidates; the song has already passed the BPM music threshold. Sorting must first check if the overall lyrics"
            "Describe current_act first, then check if it continues history_context, finally use gag as tie-breaker."
            "Word collisions, song title associations, or vague emotions must not precede titles that directly describe this act"
        ),
        "candidates": _candidate_payload(allowed, catalog, registry, full=False),
        "output_schema": {
            "ranked_candidate_tids": [f"Exactly {wanted} complete TIDs within candidates, sorted by suitability"]
        },
    }
    result = _require_dict(
        llm_call(
            tag,
            [
                {"role": "system", "content": _gag_stage_system(
                    gag_prompt,
                    "Return the fields of the output_schema the user gave you.",
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            0.35,
            False,
        ),
        "full lyric ranking" if limit is None else "lyric top 5",
    )
    tids = _parse_ranked_tid_list(result.get("ranked_candidate_tids"), allowed)
    legacy_tids, analyses = _parse_ranked_rows(result.get("ranked_candidates"), allowed)
    if not tids:
        tids = legacy_tids
    if len(tids) < wanted:
        remaining = [tid for tid in allowed if tid not in tids]
        missing = wanted - len(tids)
        repair_payload = {
            "current_act": act,
            "history_context": history_context,
            "already_selected_tids": tids,
            "task": f"Previously returned too few candidates; select exactly {missing} more from remaining_candidates, no duplicates with already_selected_tids",
            "remaining_candidates": _candidate_payload(
                remaining, catalog, registry, full=False
            ),
            "output_schema": {
                "ranked_candidate_tids": [f"Exactly {missing} complete TIDs within remaining_candidates"]
            },
        }
        repaired = _require_dict(
            llm_call(
                "ranking top-up" if limit is None else "lyric top 5 refill",
                [
                    {"role": "system", "content": _gag_stage_system(
                    gag_prompt,
                    "Return the fields of the output_schema the user gave you.",
                )},
                    {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)},
                ],
                0.2,
                False,
            ),
            "ranking top-up",
        )
        extra_tids = _parse_ranked_tid_list(
            repaired.get("ranked_candidate_tids"), remaining
        )
        legacy_extra, extra_analyses = _parse_ranked_rows(
            repaired.get("ranked_candidates"), remaining
        )
        if not extra_tids:
            extra_tids = legacy_extra
        tids.extend(extra_tids[:missing])
        analyses.update(extra_analyses)
    if len(tids) < wanted:
        raise LyricsStoryError(f"lyrics sorting returned only {len(tids)}/{wanted} valid candidates")
    tids = tids[:wanted]
    analyses = {
        tid: analyses.get(tid, {
            "storytelling_analysis": "full lyric finalists not updated, waiting for complete lyrics judgment",
            "gag_analysis": "None",
            "gag_confidence": "None",
        })
        for tid in tids
    }
    # selected_tid is provisional; the full-lyrics round makes the decision.
    return SelectionBundle(tids[0], tuple(tids), analyses)


def select_from_full_lyrics(
    story: str,
    act: dict[str, Any],
    history_context: dict[str, Any],
    shortlist: SelectionBundle,
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    gag_prompt: str,
) -> SelectionBundle:
    allowed = list(shortlist.shortlist_tids)
    full_analyses: dict[str, dict[str, str]] = {}
    for tid in allowed:
        _, one = _score_story_candidates(
            act, [tid], catalog, registry, llm_call,
            full=True, tag="lyricsstory score per track", history_context=history_context,
        )
        full_analyses.update(one)
    def _final_key(tid: str) -> tuple[int, int]:
        analysis = full_analyses[tid]
        passed = int(analysis.get("story_fit") in ("direct fit", "partial fit"))
        return passed, int(analysis.get("_story_score", "-999"))

    full_order = sorted(allowed, key=_final_key, reverse=True)
    selected = next(
        (tid for tid in full_order
         if full_analyses[tid]["story_fit"] in ("direct fit", "partial fit")),
        None,
    )
    if selected is None:
        full_order = sorted(
            full_order,
            key=lambda tid: int(full_analyses[tid].get("_component_score", "-999")),
            reverse=True,
        )
        selected = full_order[0]
        for tid in full_order:
            full_analyses[tid]["storytelling_analysis"] += "; no strong lyrics match for this act, selected by relative rank"
            full_analyses[tid]["story_fit"] = "no fit (best available)"
    analyses = dict(shortlist.analyses)
    analyses.update(full_analyses)
    return SelectionBundle(selected, tuple(full_order), analyses)

    # Legacy direct-winner implementation retained below for reference only;
    # the component-score ordering above prevents post-hoc winner rationales.
    payload = {
        "current_act": act,
        "history_context": history_context,
        "task": "After reading the full lyrics of the Top 5, select the single song; only analyze the finally selected song.",
        "hard_rule": (
            "Only select tid from candidates; do not add new songs. Storytelling is mandatory; Gag is only allowed when"
            "story also gets bonus points for matching. If the lyrics theme is not current_act, even with identical words, it must be marked no fit"
        ),
        "candidates": _candidate_payload(allowed, catalog, registry, full=True),
        "output_schema": {
            "selected_song_id": "The complete TID of the candidate",
            "storytelling_analysis": "Explain why the overall lyrics of the selected song fit this act, max 30 chars",
            "story_fit": "direct fit / partial fit / no fit",
            "story_evidence": "Copy verbatim a continuous short phrase from the selected song lyrics",
            "gag_analysis": "Select song specific line connecting to History line, max 30 chars; no reliable hook fill none",
            "gag_confidence": "High/Medium/Low/None",
        },
    }
    result = _require_dict(
        llm_call(
            "full lyric finalists",
            [
                {"role": "system", "content": _gag_stage_system(
                    gag_prompt,
                    "Return the fields of the output_schema the user gave you.",
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            0.25,
            False,
        ),
        "full lyric finalists",
    )
    selected = _canonical_tid(result.get("selected_song_id"), allowed)
    if selected is None:
        raise LyricsStoryError("full lyric finalists returned a TID outside the Top 5")
    analyses = dict(shortlist.analyses)
    _, legacy_analyses = _parse_ranked_rows(result.get("ranked_candidates"), allowed)
    analyses.update(legacy_analyses)
    direct_analysis = require_grounded_story_analysis(
        result, selected, catalog, "full lyric finalists"
    )
    if (direct_analysis["storytelling_analysis"] != "model returned no analysis"
            or selected not in analyses):
        analyses[selected] = direct_analysis
    for tid in allowed:
        analyses.setdefault(
            tid,
            {
                "storytelling_analysis": "full lyric finalists not updated, relying on representative lyrics judgment",
                "gag_analysis": "None",
                "gag_confidence": "None",
                "story_fit": "unverified",
                "story_evidence": "",
            },
        )
    return SelectionBundle(selected, tuple(allowed), analyses)


ENERGY_FLOOR_UP = float(os.environ.get("AIDJ_ENERGY_FLOOR_UP", "0.60"))
ENERGY_CEIL_DOWN = float(os.environ.get("AIDJ_ENERGY_CEIL_DOWN", "0.55"))
def act_demands_calm(description: str) -> bool:
    """True when the act names a subdued sound, not merely a sad subject."""
    return act_energy(description) == "calm"


def arousal_conflicts_with_act(description: str, arousal: Any) -> bool:
    """True when the recording's energy contradicts what the act asks for."""
    peak = act_demands_peak_energy(description)
    calm = act_demands_calm(description)
    if not peak and not calm:
        return False
    try:
        value = float(arousal)
    except (TypeError, ValueError):
        return False        # no measurement is not evidence against the song
    return value < ENERGY_FLOOR_UP if peak else value > ENERGY_CEIL_DOWN


def apply_act_backstops(analysis: dict[str, str], description: str,
                        tid: str, catalog: LyricCatalog,
                        registry: dict[str, dict[str, Any]] | None = None,
                        subject_verdict: dict[str, str] | None = None,
                        ) -> dict[str, str]:
    """Apply the code-side act checks to one already-grounded analysis."""
    if analysis.get("story_fit") not in ("direct fit", "partial fit"):
        return analysis

    def _reject(reason: str) -> dict[str, str]:
        analysis["storytelling_analysis"] = (
            str(analysis.get("storytelling_analysis", "")) + reason
        )
        analysis["story_fit"] = "no fit"
        analysis["story_evidence"] = ""
        analysis["_story_score"] = "-999"
        return analysis

    arousal = (registry or {}).get(tid, {}).get("arousal")
    if arousal_conflicts_with_act(description, arousal):
        return _reject(
            f"this act requires burst high energy but song arousal={float(arousal):.3f}"
            f" below threshold {ENERGY_FLOOR_UP:.2f}"
        )
    try:
        full_lyrics = catalog.full_lyrics(tid)
    except LyricsStoryError:
        return analysis
    conflict = theme_conflicts_with_act(description, tid)
    if conflict:
        return _reject(f"；{conflict}")
    return analysis


def analyze_single_full_lyrics(
    story: str,
    act: dict[str, Any],
    history_context: dict[str, Any],
    tid: str,
    catalog: LyricCatalog,
    registry: dict[str, dict[str, Any]],
    llm_call: LLMCall,
    gag_prompt: str,
) -> dict[str, str]:
    """Analyze an acoustically accepted ranked replacement without re-ranking."""
    if tid not in registry or not catalog.has_song(tid):
        raise LyricsStoryError(f"Planner switched to song: missing complete lyrics or Song Card: {tid}")
    payload = {
        "current_act": act,
        "history_context": history_context,
        "task": (
            "song has been finalized by the Music Planner; only analyze this fixed song, do not change the title."
            "storytelling_analysis must cite the actual semantics of candidate.lyrics;"
            "story_fit must determine direct fit / partial fit / no fit; story_evidence must copy the original lyrics verbatim;"
            "gag_analysis must reference sentences actually present in candidate.lyrics and history_context."
            "Do not copy the field description; when the evidence is weak, say plainly that there is no reliable lyric hook."
        ),
        "candidate": _candidate_payload([tid], catalog, registry, full=True)[0],
    }
    analysis = None
    messages = [
        {"role": "system", "content": _gag_stage_system(
                    gag_prompt,
                    "Return the fields of the output_schema the user gave you.",
                )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    last_error: LyricsStoryError | None = None
    for attempt in range(3):
        result = _require_dict(
            llm_call("Planner switched to lyrics analysis", messages, 0.2, False),
            "Planner switched to lyrics analysis",
        )
        returned_tid = _canonical_tid(result.get("tid", tid), [tid])
        if returned_tid is None:
            raise LyricsStoryError("Planner switched to analysis: returned incorrect TID")
        try:
            analysis = require_grounded_story_analysis(
                result, tid, catalog, "Lyrics review for acoustically qualified songs"
            )
            break
        except LyricsStoryError as exc:
            last_error = exc
            if "story_evidence is not a direct quote from the lyrics" not in str(exc):
                raise
            messages = messages + [
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                {"role": "user", "content":
                 f"The story_evidence you filled in: 「{_clean_text(result.get('story_evidence'), 80)}」"
                 "Not found in candidate.lyrics. Please from candidate.lyrics"
                 "Copy the entire line exactly as story_evidence, without changing a single character."
                 "Do not merge two lines; do not translate. Re-output the JSON."}]
    if analysis is None:
        raise last_error or LyricsStoryError("Lyrics review failed for acoustically qualified songs")
    description = str(act.get("description", ""))
    # Ask the subject question only when the analysis would otherwise stand:
    # a song already judged no fit costs nothing more to confirm.
    subject = None
    if analysis.get("story_fit") in ("direct fit", "partial fit"):
        subject = act_subject_verdict(description, tid, catalog, llm_call)
    return apply_act_backstops(
        analysis, description, tid, catalog, registry, subject
    )


def validate_planner_swap(
    requested_tid: Any,
    shortlist_tids: Sequence[str],
    used_tids: Iterable[str],
    registry: dict[str, Any],
) -> str | None:
    allowed = [tid for tid in shortlist_tids if tid in registry and tid not in set(used_tids)]
    return _canonical_tid(requested_tid, allowed)


def opening_history(catalog: LyricCatalog, tid: str, expected_seconds: float = 45.0) -> dict[str, Any]:
    lyrics = catalog.lyrics_between(tid, 0.0, expected_seconds)
    return {
        "committed_songs": [],
        "current_song": {
            "tid": tid,
            "title": catalog.title(tid),
            "known_entry_sec": 0.0,
            "exit_status": "A→B cue_out not yet planned",
            "expected_lyrics": lyrics or catalog.compact_lyrics(tid),
        },
    }


def history_for_c(
    catalog: LyricCatalog,
    tid_a: str,
    tid_b: str,
    plan_ab: dict[str, Any],
    expected_b_seconds: float = 40.0,
) -> dict[str, Any]:
    cut = float(plan_ab["cut"])
    entry = float(plan_ab["entry"])
    a_lyrics = catalog.lyrics_between(tid_a, 0.0, cut)
    b_lyrics = catalog.lyrics_between(tid_b, entry, entry + expected_b_seconds)
    return {
        "committed_songs": [
            {
                "tid": tid_a,
                "title": catalog.title(tid_a),
                "played_range_sec": [0.0, round(cut, 2)],
                "actually_played_lyrics": a_lyrics or catalog.compact_lyrics(tid_a),
            }
        ],
        "current_song": {
            "tid": tid_b,
            "title": catalog.title(tid_b),
            "known_entry_sec": round(entry, 2),
            "expected_range_sec": [round(entry, 2), round(entry + expected_b_seconds, 2)],
            "exit_status": "B→C cue_out not planned",
            "expected_lyrics": b_lyrics or catalog.compact_lyrics(tid_b),
        },
    }
