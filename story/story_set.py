#!/usr/bin/env python3
"""Lyrics-aware three-song set on top of the existing tech-first planner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import song_library as A
import transition_core as LD
import retrieval as LT
import lyric_gate as LS
import render_plan as PF
import regression_check as RC
from plan_sim import (
    anchor_table,
    b_lead_need_beats,
    c1c_min_ratio,
    resolve_cue,
    safe_cut_near,
    sim_c1,
    sim_c1b,
    sim_c1c,
    sim_c2b,
    sim_c2c,
    sim_c6b,
    sim_c9,
    sim_plan,
    validate_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_MUSICAL_CANDIDATES = max(6, int(os.environ.get("AIDJ_TF_TOPK", "12")))
PLANS_PER_SONG = max(1, int(os.environ.get("AIDJ_PLANS_PER_SONG", "5")))
# How many of a song's already-legal entry points become separate routes.
ENTRIES_PER_SONG = max(1, int(os.environ.get("AIDJ_ENTRIES_PER_SONG", "4")))
REQUIRE_DIRECT_FIT = os.environ.get("AIDJ_REQUIRE_DIRECT_FIT", "0") == "1"


def forced_order() -> list[str]:
    """Three tids to plan, instead of searching for them."""
    raw = os.environ.get("AIDJ_FORCE_ORDER", "")
    return [part for part in (piece.strip() for piece in raw.split("|")) if part]
_ACCEPTED_FITS = ("direct fit",) if REQUIRE_DIRECT_FIT else ("direct fit", "partial fit")

_FIT_RANK = {"direct fit": 2, "partial fit": 1, "no fit": 0}

SET_TARGET_SEC = float(os.environ.get("AIDJ_SET_TARGET_SEC", "180"))
SET_SLACK_SEC = float(os.environ.get("AIDJ_SET_SLACK_SEC", "8"))


MIN_SONG_SHARE_SEC = float(os.environ.get("AIDJ_MIN_SONG_SHARE_SEC", "30"))
MAX_SONG_SHARE_SEC = float(os.environ.get("AIDJ_MAX_SONG_SHARE_SEC", "90"))


def share_window(n_songs: int = 3) -> tuple[float, float]:
    """The band a middle song's audible share may occupy and still hit target."""
    low = MIN_SONG_SHARE_SEC
    high = min(MAX_SONG_SHARE_SEC,
               SET_TARGET_SEC - (n_songs - 1) * MIN_SONG_SHARE_SEC)
    return low, max(low + SET_SLACK_SEC, high)


def _song_budget(n_songs: int = 3) -> float:
    """Seconds of audible programme each song gets in the target-length set."""
    return SET_TARGET_SEC / max(1, n_songs)


def _opener_start(ta: str, cut: float, budget: float) -> float:
    """Where song 1 enters the set so that its share matches the budget."""
    target = max(0.0, float(cut) - budget)
    if target <= 0.5:
        return 0.0
    heads = [
        float(row["start"]) for row in A.REG.get(ta, {}).get("segments", [])
        if row.get("label") not in ("start", "silence", "end")
        and 0.0 < float(row["start"]) < float(cut) - 10.0
        and abs(float(row["start"]) - target) <= SET_SLACK_SEC
    ]
    if heads:
        return round(min(heads, key=lambda head: abs(head - target)), 2)
    downbeats = A.REG.get(ta, {}).get("downbeat_times") or []
    if downbeats:
        return round(float(min(downbeats, key=lambda d: abs(float(d) - target))), 2)
    return round(target, 2)


def _apply_cut_budget(accepted: list[dict[str, Any]],
                      target: float | None) -> list[dict[str, Any]]:
    """Order already-legal cue_out proposals by how well they fit the budget."""
    if target is None or not accepted:
        return accepted
    def distance(proposal: dict[str, Any]) -> float:
        return abs(float(proposal["cue_out"]) - target)
    inside = sorted((p for p in accepted if distance(p) <= SET_SLACK_SEC), key=distance)
    outside = sorted((p for p in accepted if distance(p) > SET_SLACK_SEC), key=distance)
    return inside + outside


def _emit(rid: str, kind: str, data: dict[str, Any]) -> None:
    LD.emit(rid, kind, data)


def _selector_schema(tag: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Constrain the selector to the exact song IDs sent in this request."""
    try:
        payload = json.loads(messages[-1]["content"])
    except Exception:
        return {"type": "object"}
    short = {"type": "string", "maxLength": 180}
    fit = {"type": "string", "enum": ["direct fit", "partial fit", "no fit"]}
    if tag in ("opening act scoring", "lyricsstory score per track"):
        allowed = [str(row["tid"]) for row in payload.get("candidates", []) if row.get("tid")]
        score = {"type": "integer", "minimum": 0, "maximum": 2}
        row = {
            "type": "object",
            "properties": {
                "tid": {"type": "string", "enum": allowed},
                "relation": {"type": "string", "enum": list(LS._RELATION_CHOICES)},
                "subject_score": score,
                "event_score": score,
                "tone_score": score,
                "contradiction": {"type": "boolean"},
                "story_fit": fit,
                "story_evidence": short,
            },
            "required": ["tid", "relation", "subject_score", "event_score",
                         "tone_score", "contradiction", "story_fit",
                         "story_evidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"candidate_scores": {
                "type": "array", "items": row,
                "minItems": len(allowed), "maxItems": len(allowed),
            }},
            "required": ["candidate_scores"], "additionalProperties": False,
        }
    if tag == "song theme for act":
        # One narrow question with the full lyrics, nothing to score and nothing
        # to justify -- the load under which the big per-song call rationalises.
        return {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["is", "Partial", "No"]},
                "reason": {"type": "string", "maxLength": 120},
                "evidence": short,
            },
            "required": ["verdict", "reason", "evidence"],
            "additionalProperties": False,
        }
    if tag == "act theme":
        return {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(LS.THEME_CHOICES)},
                    "minItems": 1, "maxItems": 3,
                },
                "required_relations": {
                    "type": "array",
                    "items": {"type": "string",
                              "enum": list(LS.RELATION_CHOICES_THEME)},
                    "minItems": 0, "maxItems": 2,
                },
                "energy": {"type": "string", "enum": list(LS.ENERGY_CHOICES)},
                "reason": {"type": "string", "maxLength": 160},
            },
            "required": ["themes", "required_relations", "energy", "reason"],
            "additionalProperties": False,
        }
    if tag == "song theme":
        return {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(LS.THEME_CHOICES)},
                    "minItems": 1, "maxItems": 3,
                },
                "summary": {"type": "string", "maxLength": 120},
                "relation": {"type": "string",
                             "enum": list(LS.RELATION_CHOICES_THEME)},
                "evidence": short,
            },
            "required": ["themes", "summary", "relation", "evidence"],
            "additionalProperties": False,
        }
    if tag == "evidence check":
        allowed = [str(row["tid"]) for row in payload.get("claims", []) if row.get("tid")]
        row = {
            "type": "object",
            "properties": {
                "tid": {"type": "string", "enum": allowed},
                "evidence_entails_act": {"type": "boolean"},
                "reason": short,
            },
            "required": ["tid", "evidence_entails_act", "reason"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"verdicts": {
                "type": "array", "items": row,
                "minItems": len(allowed), "maxItems": len(allowed),
            }},
            "required": ["verdicts"], "additionalProperties": False,
        }
    if tag == "Planner switched to lyrics analysis":
        tid = str((payload.get("candidate") or {}).get("tid", ""))
        return {
            "type": "object",
            "properties": {
                "tid": {"type": "string", "const": tid},
                "storytelling_analysis": short,
                "story_fit": fit,
                "story_evidence": short,
                "gag_analysis": short,
                "gag_confidence": {"type": "string", "enum": ["High", "Middle", "Low", "None"]},
            },
            "required": ["tid", "storytelling_analysis", "story_fit", "story_evidence",
                         "gag_analysis", "gag_confidence"],
            "additionalProperties": False,
        }
    return {"type": "object"}


def _satisfies_schema(result: Any, schema: dict[str, Any]) -> bool:
    """Whether a reply carries the keys its stage declared as required."""
    if not isinstance(result, dict):
        return False
    required = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(required, list):
        return True
    return all(key in result for key in required)


def lyric_llm_call(rid: str):
    """Adapter with prompt caching and decode-time allowlists."""
    def call(tag: str, messages: list[dict[str, str]], temperature: float,
             reasoning: bool) -> dict[str, Any] | None:
        _emit
        cache_dir = REPO_ROOT / "runtime" / "llm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        effective_reasoning = reasoning and tag not in {
            "opening act scoring", "lyricsstory score per track", "evidence check",
            "Planner switched to lyrics analysis", "song theme", "song theme for act", "act theme",
        }
        schema = _selector_schema(tag, messages)
        material = json.dumps({
            "model": os.environ.get("AIDJ_LLM_MODEL_ID", "unknown"),
            "messages": messages,
            "temperature": temperature,
            "reasoning": effective_reasoning,
            "schema": schema,
            "version": 5,
        }, ensure_ascii=False, sort_keys=True)
        path = cache_dir / (hashlib.sha256(material.encode()).hexdigest() + ".json")
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if _satisfies_schema(cached, schema):
                    _emit(rid, "stage", {"msg": f"{tag}: Using cached Qwen with the same Prompt."})
                    return cached
                path.unlink()      # poisoned entry: drop it and ask the model again
            except Exception:
                pass
        result, _ = LD.llm_stream(
            rid, tag, messages,
            max_tokens=1800 if "Score each song" in tag else 1000,
            temperature=temperature, reasoning=effective_reasoning,
            json_only=True, json_schema=schema,
        )
        if _satisfies_schema(result, schema):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            os.replace(tmp, path)
        return result
    return call


def story_acts(story: str) -> list[dict[str, Any]]:
    """Split user wording deterministically; V/A never rewrites the story."""
    dummy = [{"V": None, "A": None, "description": ""} for _ in range(3)]
    return LS.make_story_acts(story, dummy)


def _runtime_song_ok(tid: str) -> bool:
    if not LS.lyrics_are_usable(tid):
        return False
    return (
        tid in A.REG and tid in A.MAN and tid in A.VOICED
        and bool(A.REG[tid].get("segments"))
        and os.path.isfile(A.MAN[tid].get("audio_path", ""))
    )


def _eligible_lyrics(catalog: LS.LyricCatalog) -> list[str]:
    tids = catalog.eligible_registry_tids(A.REG, A.QUAR, A.SUSPECT, A.in_pool)
    return [tid for tid in tids if _runtime_song_ok(tid)]


def _exit_schema(ta: str, cut_floor: float,
                 cut_ceiling: float | None = None) -> dict[str, Any]:
    rows = [row for row in anchor_table(ta)
            if cut_floor <= float(row["Seconds"]) < A.REG[ta]["duration_sec"] - 1]
    if cut_ceiling is not None:
        for widen in (0.0, 30.0, 60.0, 120.0):
            capped = [row for row in rows if float(row["Seconds"]) <= cut_ceiling + widen]
            if capped:
                rows = capped
                break
    ids = [row["id"] for row in rows]
    if not ids:
        raise LS.LyricsStoryError(f"{A.REG[ta]['title']} has no exit anchors matching the playback track")
    item = {
        "type": "object",
        "properties": {
            "cue_out": {
                "type": "object",
                "properties": {
                    "anchor": {"type": "string", "enum": ids},
                    "offset_beats": {"type": "integer", "minimum": -8, "maximum": 8},
                },
                "required": ["anchor", "offset_beats"], "additionalProperties": False,
            },
            "exit_tool": {"type": "string", "enum": list(PF.FX_CARD["exit effect"])},
            "b_entry_need": {"type": "string", "enum": ["lead16", "section_head", "drum_entry"]},
            "va_dir": {"type": "string", "enum": ["near", "up", "down"]},
            "why": {"type": "string", "maxLength": 180},
        },
        "required": ["cue_out", "exit_tool", "b_entry_need", "va_dir", "why"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"option": {"type": "array", "items": item, "minItems": 2, "maxItems": 3}},
        "required": ["option"], "additionalProperties": False,
    }


def _gate_exit_plans(rid: str, ta: str, act: dict[str, Any],
                     prev_entry: float | None,
                     cut_target: float | None = None) -> list[dict[str, Any]]:
    cut_floor = max(45.0, 0.2 * float(A.REG[ta]["duration_sec"]))
    if prev_entry is not None:
        cut_floor = max(cut_floor, prev_entry + 15.0)
    if cut_target is not None:
        entry = float(prev_entry or 0.0)
        low, _high = share_window()
        cut_floor = min(cut_floor, max(entry + 15.0, entry + low))
    if cut_target is None:
        ceilings: list[float | None] = [None, None]
    else:
        _entry = float(prev_entry or 0.0)
        ceilings = [_entry + share_window()[1]] * 2
    base_user = {
        "A (outgoing)": PF.song_card(ta),
        "technique card": PF.FX_CARD,
        "current act": act,
        "Music priority": (
            "story is only used to select va_dir and aesthetic tendency; feasibility of Cue, vocals, beat grid, and effect trails must not be compromised."
        ),
    }
    if cut_target is not None:
        base_user["set length budget"] = (
            f"The entire set has 3 tracks, total duration should be ≈ {SET_TARGET_SEC:.0f} seconds. This track can play "
            f"{share_window()[0]:.0f}~{share_window()[1]:.0f} seconds (menu restricted accordingly),"
            f"The average allocation is {_song_budget():.0f} seconds; cue_out is most ideal near {cut_target:.0f}s;"
            f"But anchor validity takes priority; a clean end of phrase is more important than padding seconds."
        )
    messages = [{"role": "system", "content": PF.GUIDE + LT.S1_TASK},
                {"role": "user", "content": json.dumps(base_user, ensure_ascii=False)}]
    for attempt, ceiling in enumerate(ceilings):
        schema = _exit_schema(ta, cut_floor, ceiling)
        label = f"Transition entry option {attempt + 1}"
        _emit
        raw, _ = LD.llm_stream(rid, label, messages, max_tokens=1200,
                               temperature=0.55, reasoning=False,
                               json_only=True, json_schema=schema)
        accepted, rejected = [], []
        for proposal in (raw or {}).get("option", [])[:3]:
            cut, err = resolve_cue(ta, proposal.get("cue_out"))
            tool = str(proposal.get("exit_tool", "echo"))
            if err or cut is None:
                rejected.append({"reason": err or "cue_out cannot be parsed", "option": proposal})
                continue
            if cut < cut_floor:
                rejected.append({"reason": f"cue_out {cut:.1f}s < runway lower bound {cut_floor:.1f}s", "option": proposal})
                continue
            why9, fail9 = sim_c9(ta, cut)
            ratio1, fail1 = sim_c1(ta, cut, tool)
            grid = validate_grid(ta, cut)
            if fail9 or fail1 or not grid["ok"]:
                rejected.append({
                    "reason": why9 if fail9 else (
                        f"C1 exit cuts a sung phrase {ratio1:.2f}x" if fail1 else "cue_out not on beat grid"
                    ),
                    "Nearest safe cut": (safe_cut_near(ta, cut, tool) or [None])[0],
                    "option": proposal,
                })
                continue
            clean = dict(proposal)
            clean["cue_out"] = float(cut)
            accepted.append(clean)
        accepted = _widen_exit_tools(ta, _apply_cut_budget(accepted, cut_target))
        _emit(rid, "tfgate", {"attempt": attempt, "ok": accepted, "rejected": rejected})
        if accepted:
            return accepted
        if not rejected:
            break
        messages = messages[:2] + [
            {"role": "assistant", "content": json.dumps(raw or {}, ensure_ascii=False)},
            {"role": "user", "content": json.dumps({
                "all options rejected by original Tech-first gate": rejected,
                "instruction": "Only re-generate the full option using valid anchors from the title card; do not relax any conditions.",
            }, ensure_ascii=False)},
        ]
    fallback = _enumerate_exit_plans(ta, act, cut_floor,
                                     ceilings[-1] if ceilings else None, cut_target)
    _emit(rid, "tfgate", {"attempt": "Deterministic Fallback", "ok": fallback, "rejected": []})
    return fallback


def _enumerate_exit_plans(ta: str, act: dict[str, Any], cut_floor: float,
                          ceiling: float | None,
                          cut_target: float | None) -> list[dict[str, Any]]:
    """Every legal cue_out in A's own anchor table, cheapest exit tool first."""
    try:
        rows = anchor_table(ta)
    except Exception:
        return []
    direction = LS.act_energy_direction(str(act.get("description", "")))
    va_dir = {"up": "up", "down": "down"}.get(direction or "", "near")
    duration = float(A.REG[ta].get("duration_sec") or 0)
    plans: list[dict[str, Any]] = []
    for row in rows:
        try:
            cut = float(row["Seconds"])
        except (TypeError, ValueError, KeyError):
            continue
        if cut < cut_floor or cut > duration - 1:
            continue
        if ceiling is not None and cut > ceiling:
            continue
        if sim_c9(ta, cut)[1] or not validate_grid(ta, cut)["ok"]:
            continue
        for tool in sorted(_ALT_EXIT_TOOLS, key=lambda name: _EXIT_TOOL_RISK.get(name, 9)):
            if sim_c1(ta, cut, tool)[1]:
                continue
            plans.append({
                "cue_out": cut,
                "exit_tool": tool,
                "b_entry_need": LT.NEED_OF.get(tool, "section_head"),
                "va_dir": va_dir,
                "why": "A: anchor point table, verified legal exit points (fallback when all LLM proposals are rejected)",
            })
    return _widen_exit_tools(
        ta, _apply_cut_budget(plans, cut_target))[:MAX_EXIT_PROPOSALS]


_ALT_EXIT_TOOLS = ("blendecho", "filter_lpf", "echo",
                   "blend16", "filter_hpf", "reverb",
                   "loop_in", "loop", "introstack")
# Lower = less likely to leave a silent seam.  Ride tools overlap the two
# tracks; filter sweeps stay audible; echo/reverb decay into a gap.
_EXIT_TOOL_RISK = {"blendecho": 0, "blend16": 0, "introstack": 0, "loop_in": 0,
                   "loop": 0, "filter_lpf": 1, "filter_hpf": 1,
                   "echo": 2, "reverb": 2}


def _widen_exit_tools(ta: str, accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Offer each legal cue_out under every exit tool that is *also* legal there."""
    widened: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    preferred = [str(p.get("exit_tool", "")) for p in accepted]
    def _tool_order(name: str) -> tuple[int, int]:
        try:
            model_rank = preferred.index(name)
        except ValueError:
            model_rank = len(preferred)
        return _EXIT_TOOL_RISK.get(name, 9), model_rank
    tools = sorted(_ALT_EXIT_TOOLS, key=_tool_order)
    for _tool_index, _cut_index in sorted(
        ((ti, ci) for ti in range(len(tools)) for ci in range(len(accepted))),
        key=lambda pair: (pair[0] + pair[1], pair[0]),
    ):
        tool = tools[_tool_index]
        proposal = accepted[_cut_index]
        cut = float(proposal["cue_out"])
        if tool in _BLEND_FAMILY and hasattr(LD, "_enforce_exit"):
            cut = float(LD._enforce_exit(ta, cut, tool)[0])
        key = (round(cut, 2), tool)
        if key in seen:
            continue
        if sim_c1(ta, cut, tool)[1]:
            continue
        if sim_c1b(ta, cut, tool, 4)[1]:
            continue
        # C1d, checked here so a ride variant is never offered at a cut the
        # scorecard would fail after a full render.
        if tool in _BLEND_FAMILY and _voiced_coverage(ta, cut, cut + 1.5) > 0.5:
            continue
        seen.add(key)
        variant = dict(proposal)
        variant["cue_out"] = cut
        variant["exit_tool"] = tool
        variant["b_entry_need"] = LT.NEED_OF.get(
            tool, proposal.get("b_entry_need") or "section_head"
        )
        widened.append(variant)
        if len(widened) >= MAX_EXIT_PROPOSALS:
            return widened
    return widened


def story_mode_regression_fails(fails: Iterable[str],
                                plan: dict[str, Any] | None = None) -> list[str]:
    """Apply the branch's own story-mode rule to the rendered-audio checker."""
    kept = [failure for failure in fails if not failure.startswith("C4 ")]
    if plan is not None and str(plan.get("exit_tool")) in _BLEND_FAMILY:
        kept = [failure for failure in kept if not failure.startswith("C2c ")]
    return kept


_TITLE_NOISE = re.compile(
    r"\((?:[^()]*)\)|\[[^\]]*\]|(?:official|music|video|audio|lyrics?|mv|hd|"
    r"full|ver\.?|version|feat\.?|ft\.?|remaster(?:ed)?)",
    re.IGNORECASE,
)
_TITLE_KEEP = re.compile(r"[0-9a-z\u3400-\u9fff\uac00-\ud7af\u3040-\u30ff]+")


def title_key(title: str) -> str:
    """A title reduced to the part that identifies the recording."""
    text = _TITLE_NOISE.sub(" ", str(title or "").replace("\u2019", "'").lower())
    # An artist prefix is written "Artist - Title"; keep the longest segment so
    # the comparison is made on the song name rather than on who uploaded it.
    parts = [segment.strip() for segment in re.split(r"\s[-–—]\s", text)]
    text = max(parts, key=len) if parts else text
    return "".join(_TITLE_KEEP.findall(text))


def same_song(left: str, right: str) -> bool:
    """True when two songs name the same recording."""
    a, b = title_key(left), title_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    # A short key ("crying", "bite") collides by accident, so containment only
    # counts once the shared part is long enough to be a real title.
    return len(short) >= 8 and short in long


def songs_to_skip(used: Iterable[str]) -> set[str]:
    """`used` plus every other pool entry that is the same recording."""
    skip = set(used)
    keys = [A.REG[tid]["title"] for tid in skip if tid in A.REG]
    for tid, row in A.REG.items():
        if tid in skip:
            continue
        if any(same_song(row.get("title", ""), name) for name in keys):
            skip.add(tid)
    return skip


def discover_musical_routes(rid: str, ta: str, act: dict[str, Any],
                            allowed_lyrics: set[str], used: set[str],
                            prev_entry: float | None = None,
                            cut_target: float | None = None) -> dict[str, list[dict[str, Any]]]:
    """Run the unchanged tech-first retrieval, then intersect with lyrics."""
    exits = _gate_exit_plans(rid, ta, act, prev_entry, cut_target)
    if not exits:
        return {}
    routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for plan_index, proposal in enumerate(exits, 1):
        # The effect footprint is authoritative.  Do not let an LLM pairing
        # such as blendecho + section_head bypass the original lead16 gate.
        tool = str(proposal.get("exit_tool"))
        need = LT.NEED_OF.get(tool, proposal.get("b_entry_need") or "section_head")
        if need not in ("lead16", "section_head", "drum_entry"):
            need = "section_head"
        need_beats, _ = b_lead_need_beats(
            proposal.get("exit_tool"), None,
            proposal.get("echo_delay_beats", 1.0), proposal.get("overlap_beats", 0),
        )
        candidates = LT.retrieve(
            ta, need, proposal.get("va_dir", "near"),
            topk=MAX_MUSICAL_CANDIDATES,
            min_gap_s=need_beats * A.beat_of(ta),
        )
        min_gap_s = need_beats * A.beat_of(ta)
        for candidate in candidates:
            tid = str(candidate["b_tid"])
            if tid in used or tid not in allowed_lyrics or not _runtime_song_ok(tid):
                continue
            entries = [
                (float(row[0]), dict(row[1]), float(row[2]))
                for row in LT.entry_candidates(tid, need)
                if float(row[1].get("instrumental lead before vocal (s)", 99)) >= min_gap_s - 0.1
            ]
            entries.sort(key=lambda row: -row[2])
            if not entries:
                entries = [(float(candidate["suggested cue_in"]), {}, 0.0)]
            ride = (str(proposal.get("exit_tool", "")) in _BLEND_FAMILY
                    and hasattr(LD, "_enforce_blend_entry"))
            if ride:
                if not LD._enforce_blend_entry(tid, 0.0)[2]:
                    continue
                snapped: list[tuple[float, dict[str, Any], float]] = []
                for cue_in, evidence, score in entries[:ENTRIES_PER_SONG]:
                    moved, _was_moved, feasible = LD._enforce_blend_entry(tid, cue_in)
                    if feasible:
                        snapped.append((float(moved), evidence, score))
                seen_entries: set[float] = set()
                entries = [row for row in snapped
                           if not (round(row[0], 2) in seen_entries
                                   or seen_entries.add(round(row[0], 2)))]
                if not entries:
                    continue
            for cue_in, evidence, _score in entries[:ENTRIES_PER_SONG]:
                routes[tid].append({
                    "route_no": len(routes[tid]) + 1,
                    "exit_plan_no": plan_index,
                    "cue_out": float(proposal["cue_out"]),
                    "cue_in": cue_in,
                    "exit_tool": str(proposal.get("exit_tool", "echo")),
                    "b_entry_need": need,
                    "va_dir": proposal.get("va_dir", "near"),
                    "musical_evidence": {**candidate, **evidence, "suggested cue_in": cue_in},
                })
    if not routes:
        fallback_cut = None
        for proposal in exits:
            cut = float(proposal["cue_out"])
            if not sim_c9(ta, cut)[1] and not sim_c1(ta, cut, "echo")[1] \
                    and validate_grid(ta, cut)["ok"]:
                fallback_cut = cut
                break
            nearby = safe_cut_near(ta, cut, "echo")
            if nearby:
                candidate_cut = float(nearby[0])
                if (not sim_c9(ta, candidate_cut)[1]
                        and validate_grid(ta, candidate_cut)["ok"]):
                    fallback_cut = candidate_cut
                    break
        if fallback_cut is not None:
            for candidate in LT.retrieve(
                ta, "section_head", "near", topk=MAX_MUSICAL_CANDIDATES
            ):
                tid = str(candidate["b_tid"])
                if tid in used or tid not in allowed_lyrics or not _runtime_song_ok(tid):
                    continue
                routes[tid].append({
                    "route_no": 1, "exit_plan_no": 0,
                    "cue_out": fallback_cut,
                    "cue_in": float(candidate["suggested cue_in"]),
                    "exit_tool": "echo", "b_entry_need": "section_head",
                    "va_dir": "near", "musical_evidence": candidate,
                    "original_live_tf_fallback": True,
                })
    _emit(rid, "tfcands", {
        "songs": len(routes),
        "routes": sum(len(rows) for rows in routes.values()),
        "titles": [A.REG[tid]["title"] for tid in routes],
    })
    return dict(routes)


def _parameter_schema(routes: list[dict[str, Any]], key_shift: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "route_no": {"type": "integer", "enum": [row["route_no"] for row in routes]},
            "entry_tool": {"type": "string", "enum": list(PF.FX_CARD["entry effect"])},
            "exit_len_beats": {"type": "integer", "minimum": 1, "maximum": 16},
            "echo_delay_beats": {"type": "number", "minimum": 0.5, "maximum": 1.5},
            "entry_advance_beats": {"type": "number", "minimum": 0, "maximum": 2},
            "overlap_beats": {"type": "number", "minimum": 0, "maximum": 16},
            "keysync": {"type": "integer", "const": int(key_shift)},
            "bpm_align": {"type": "string", "enum": ["native", "micro"]},
            "loop_echo": {"type": "boolean"},
            "why": {"type": "string", "maxLength": 240},
        },
        "required": ["route_no", "entry_tool", "exit_len_beats", "echo_delay_beats",
                     "entry_advance_beats", "overlap_beats", "keysync", "bpm_align",
                     "loop_echo", "why"],
        "additionalProperties": False,
    }


def _plan_from_choice(choice: dict[str, Any], routes: list[dict[str, Any]],
                      ta: str, tb: str) -> tuple[dict[str, Any] | None, str | None]:
    route = next((row for row in routes if row["route_no"] == choice.get("route_no")), None)
    if route is None:
        return None, "route_no not in the original Tech-first candidates"
    entry_tool = str(choice.get("entry_tool", "direct"))
    if route["exit_tool"] in ("blendecho", "blend16", "loop", "loop_in", "introstack"):
        entry_tool = "direct"
    raw = {
        "cue_out": route["cue_out"], "cue_in": route["cue_in"],
        "exit_tool": route["exit_tool"], "entry_tool": entry_tool,
        "exit_len_beats": choice.get("exit_len_beats", 4),
        "echo_delay_beats": choice.get("echo_delay_beats", 0.75),
        "entry_advance_beats": choice.get("entry_advance_beats", 0),
        "overlap_beats": choice.get("overlap_beats", 0),
        "keysync": choice.get("keysync", 0),
        "loop_echo": choice.get("loop_echo", False),
    }
    return LD.to_plan(raw, ta, tb)


_BLEND_FAMILY = ("blendecho", "blend16", "loop_in", "loop", "introstack")
_V2_CHECKER = hasattr(LD, "_enforce_exit")


def _voiced_coverage(tid: str, start: float, end: float) -> float:
    """Fraction of [start, end] that carries vocal, from the same VOICED spans
    transition_core.scorecard reads for C1d / C2d."""
    span = max(1e-6, end - start)
    return sum(
        max(0.0, min(end, stop) - max(start, begin))
        for begin, stop in A.VOICED.get(tid, [])
        if stop > start and begin < end
    ) / span


def cheap_presim_fails(ta: str, tb: str, plan: dict[str, Any]) -> list[str]:
    """The pre-render checks that need only VOICED spans, segments and the beat
    grid.  No audio is decoded, so this is cheap enough to enumerate over a
    parameter grid."""
    cut, entry = float(plan["cut"]), float(plan["entry"])
    tool = plan.get("exit_tool")
    entry_tool = str(plan.get("entry_tool", "") or "").strip("_")
    exit_len = plan.get("exit_len_beats", 4)
    delay = plan.get("echo_delay_beats", 1.0)
    overlap = plan.get("overlap_beats", 0.0)
    fails: list[str] = []
    base = sim_plan(ta, tb, plan)
    if base.get("_c1_fail"):
        fails.append(f"C1 exit cuts a sung phrase on A: {base.get('C1_prediction')}")
    if base.get("_c2_fail"):
        fails.append(f"C2 entry lands mid-phrase on B: {base.get('C2_prediction')}")
    frac, bad = sim_c1b(ta, cut, tool, exit_len)
    if bad:
        fails.append(f"C1b effect window over vocal: vocal coverage {frac:.2f} ≥ 0.5")
    secs, bad = sim_c2b(ta, cut, tb, entry, tool, overlap)
    if bad:
        fails.append(f"C2b both vocals in overlap：{secs}s > 1.0s")
    gap, need, bad = sim_c2c(ta, cut, tb, entry, tool, entry_tool, delay, overlap)
    if bad and not (_V2_CHECKER and str(tool) in _BLEND_FAMILY):
        fails.append(f"C2c effect tail presses B vocal: gap {gap}s < need {need}s")

    if _V2_CHECKER and str(tool) in _BLEND_FAMILY:
        after = _voiced_coverage(ta, cut, cut + 1.5)
        if after > 0.5:
            fails.append(f"C1d blend entrance cuts mid-phrase: cut + 1.5s A vocal {after:.0%} > 50%")

    # C2d: the hand-off must land on a clean section head, not halfway into a
    # line B is already singing.
    if _V2_CHECKER:
        before = _voiced_coverage(tb, entry - 1.5, entry)
        if before > 0.5:
            fails.append(f"C2d entry mid-phrase: entry 1.5s before B vocal {before:.0%} > 50%")

    if str(tool) in ("blendecho", "blend16", "loop_in", "loop"):
        label = next(
            (row["label"] for row in A.REG.get(tb, {}).get("segments", [])
             if row["start"] <= entry < row["end"]),
            "?",
        )
        if _V2_CHECKER:
            if label not in ("verse", "chorus", "hook", "drop"):
                fails.append(f"C6d B did not enter verse: handoff point at '{label}', must be verse/chorus/hook")
        elif label in ("intro", "start", "outro"):
            fails.append(f"C6d B did not enter verse: entry at '{label}', must be verse/chorus/main")
    duration_b = A.REG.get(tb, {}).get("duration_sec")
    if duration_b and duration_b - entry < 20:
        fails.append(f"C6 entry runway: B has only {duration_b - entry:.0f}s left, short set mode requires ≥ 20s")
    return fails


_EXIT_LEN_GRID = (2, 4, 8)
_DELAY_GRID = (0.5, 0.75, 1.0, 1.25)
_ADVANCE_GRID = (0.0, 1.0, 2.0)
_OVERLAP_GRID = (4.0, 2.0, 8.0, 16.0, 0.0)
MAX_LEGAL_OPTIONS = max(4, int(os.environ.get("AIDJ_MAX_LEGAL_OPTIONS", "18")))
_PER_ROUTE_CAP = 12   # bound the grid walk; routes are already score-ordered
# Each proposal costs one full retrieval.retrieve() scan of the registry.
MAX_EXIT_PROPOSALS = max(2, int(os.environ.get("AIDJ_MAX_EXIT_PROPOSALS", "12")))
# How many top-ranked songs get the full-lyrics, one-call-per-song read.
FINALISTS = max(3, int(os.environ.get("AIDJ_FINALISTS", "8")))


def enumerate_legal_plans(ta: str, tb: str, routes: list[dict[str, Any]],
                          key_shift: int,
                          cut_target: float | None = None) -> list[dict[str, Any]]:
    """Every (route x parameter) combination that already clears the symbolic"""
    per_route: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for route in routes:
        bucket: list[dict[str, Any]] = []
        tool = str(route.get("exit_tool", "echo"))
        ride = tool in ("blendecho", "blend16", "loop", "loop_in", "introstack")
        entry_tools = ["direct"] if ride else ["direct", "spinup", "filter_in", "filter_in_hpf"]
        for exit_len in _EXIT_LEN_GRID:
            for delay in _DELAY_GRID:
                for advance in _ADVANCE_GRID:
                    for overlap in _OVERLAP_GRID:
                        for entry_tool in entry_tools:
                            raw = {
                                "cue_out": route["cue_out"], "cue_in": route["cue_in"],
                                "exit_tool": tool, "entry_tool": entry_tool,
                                "exit_len_beats": exit_len,
                                "echo_delay_beats": delay,
                                "entry_advance_beats": advance,
                                "overlap_beats": overlap,
                                "keysync": key_shift, "loop_echo": False,
                            }
                            plan, error = LD.to_plan(raw, ta, tb)
                            if plan is None or error:
                                continue
                            if cheap_presim_fails(ta, tb, plan):
                                continue
                            if tool in ("echo", "reverb") and overlap <= 0.0:
                                continue
                            signature = json.dumps(
                                {k: plan.get(k) for k in
                                 ("cut", "entry", "exit_tool", "entry_tool",
                                  "exit_len_beats", "echo_delay_beats",
                                  "entry_advance_beats", "overlap_beats")},
                                sort_keys=True,
                            )
                            if signature in seen:
                                continue
                            seen.add(signature)
                            bucket.append({
                                "route_no": route["route_no"],
                                "cue_out": plan["cut"], "cue_in": plan["entry"],
                                "exit_tool": tool, "entry_tool": entry_tool,
                                "exit_len_beats": exit_len,
                                "echo_delay_beats": delay,
                                "entry_advance_beats": advance,
                                "overlap_beats": overlap,
                                "_plan": plan,
                                "musical_evidence": route.get("musical_evidence", {}),
                            })
                            if len(bucket) >= _PER_ROUTE_CAP:
                                break
                        if len(bucket) >= _PER_ROUTE_CAP:
                            break
                    if len(bucket) >= _PER_ROUTE_CAP:
                        break
                if len(bucket) >= _PER_ROUTE_CAP:
                    break
            if len(bucket) >= _PER_ROUTE_CAP:
                break
        per_route.append(bucket)
    def _tier(option: dict[str, Any]) -> int:
        return _EXIT_TOOL_RISK.get(option["exit_tool"], 3)

    def _dead_air_risk(option: dict[str, Any]) -> tuple[int, int]:
        """Tool risk first, then whether the two tracks actually overlap."""
        overlap = float(option.get("overlap_beats", 0.0))
        no_overlap = int(overlap <= 0.0)
        preference = {8.0: 0, 4.0: 1, 16.0: 2, 2.0: 3, 0.0: 4}
        return (_tier(option), no_overlap,
                preference.get(overlap, 5 if overlap > 0 else 6))

    _engine_driven = ("loop", "loop_in")
    for bucket_index, bucket in enumerate(per_route):
        kept, seen_engine = [], set()
        for option in bucket:
            tool = option["exit_tool"]
            if tool in _engine_driven:
                if tool in seen_engine:
                    continue
                seen_engine.add(tool)
            kept.append(option)
        per_route[bucket_index] = kept

    best_tier = min((_tier(o) for bucket in per_route for o in bucket), default=None)
    best_tier_cap = max(4, MAX_LEGAL_OPTIONS // 2)
    options: list[dict[str, Any]] = []
    for wanted_best in (True, False):
        tiers = [[o for o in bucket if (_tier(o) == best_tier) == wanted_best]
                 for bucket in per_route]
        limit = best_tier_cap if wanted_best else MAX_LEGAL_OPTIONS
        for index in range(max((len(bucket) for bucket in tiers), default=0)):
            for bucket in tiers:
                if index < len(bucket):
                    options.append(bucket[index])
                    if len(options) >= limit:
                        break
            if len(options) >= limit:
                break
        if len(options) >= MAX_LEGAL_OPTIONS:
            break
    def _budget_miss(row: dict[str, Any]) -> float:
        if cut_target is None:
            return 0.0
        return abs(float(row["cue_out"]) - cut_target)

    options.sort(key=lambda row: (
        _budget_miss(row) > SET_SLACK_SEC,
        _dead_air_risk(row),
        _budget_miss(row),
    ))
    for number, option in enumerate(options, 1):
        option["option_no"] = number
    return options


def presim_fails(ta: str, tb: str, plan: dict[str, Any]) -> list[str]:
    """Run every pre-render simulator the original branch ships, with the"""
    cut, entry = float(plan["cut"]), float(plan["entry"])
    tool = plan.get("exit_tool")
    entry_tool = str(plan.get("entry_tool", "") or "").strip("_")
    exit_len = plan.get("exit_len_beats", 4)
    delay = plan.get("echo_delay_beats", 1.0)
    overlap = plan.get("overlap_beats", 0.0)
    fails: list[str] = []

    base = sim_plan(ta, tb, plan)
    if base.get("_c1_fail"):
        fails.append(f"C1 exit cuts a sung phrase on A: {base.get('C1_prediction')}")
    if base.get("_c2_fail"):
        fails.append(f"C2 entry lands mid-phrase on B: {base.get('C2_prediction')}")

    frac, bad = sim_c1b(ta, cut, tool, exit_len)
    if bad:
        fails.append(f"C1b effect window over vocal: vocal coverage {frac:.2f} ≥ 0.5")
    ratio, bad = sim_c1c(ta, cut, tool, exit_len)
    if bad:
        fails.append(f"C1c effect source window has no energy: {ratio} < {c1c_min_ratio():.2f}")
    secs, bad = sim_c2b(ta, cut, tb, entry, tool, overlap)
    if bad:
        fails.append(f"C2b both vocals in overlap：{secs}s > 1.0s")
    gap, need, bad = sim_c2c(ta, cut, tb, entry, tool, entry_tool, delay, overlap)
    if bad:
        fails.append(
            f"C2c effect tail presses B vocal: gap {gap}s < need {need}s"
            "(Reduce echo_delay_beats, set overlap_beats to 0, or switch to a route for this song with larger vocal gaps)"
        )
    ratio, bad = sim_c6b(ta, cut, tb, entry, entry_tool, overlap, tool, exit_len)
    if bad:
        fails.append(f"C6b B entry energy cliff: {ratio} < 0.25")
    return fails


def _brief_card(tid: str) -> dict[str, Any]:
    """Just enough of a song card to choose between pre-validated options."""
    card = PF.song_card(tid)
    return {
        "tid": tid, "title": card.get("title"), "bpm": card.get("bpm"),
        "key": card.get("key"), "length": card.get("length"),
        "V_q": card.get("V_q"), "A_q": card.get("A_q"),
        "sections": [
            {"label": row.get("label"), "start": row.get("start"),
             "trend": row.get("trend")}
            for row in (card.get("sections") or [])
        ][:18],
    }


def _has_budgeted_exit(tb: str, entry: float) -> bool:
    """Can B hand over again inside the set's length budget?"""
    try:
        rows = anchor_table(tb)
    except Exception:
        return True          # unknown -> do not veto on a missing anchor table
    low, high = share_window()
    floor = max(entry + 15.0, entry + low)
    ceiling = entry + high
    duration = float(A.REG[tb]["duration_sec"])
    for row in rows:
        cut = float(row["Seconds"])
        if not (floor <= cut <= min(ceiling, duration - 1)):
            continue
        if sim_c9(tb, cut)[1] or not validate_grid(tb, cut)["ok"]:
            continue
        if not sim_c1(tb, cut, "echo")[1]:
            return True
        if "blend exit" not in (row.get("applies to") or []):
            continue
        for tool in ("blendecho", "blend16", "introstack"):
            moved = cut
            if hasattr(LD, "_enforce_exit"):
                try:
                    moved = float(LD._enforce_exit(tb, cut, tool)[0])
                except Exception:
                    moved = cut
            if not (floor <= moved <= min(ceiling, duration - 1)):
                continue
            if sim_c1(tb, moved, tool)[1] or sim_c1b(tb, moved, tool, 4)[1]:
                continue
            if _voiced_coverage(tb, moved, moved + 1.5) > 0.5:
                continue
            return True
    return False


def try_ranked_candidates(
    rid: str,
    hop: int,
    ta: str,
    lyric_order: Iterable[str],
    routes_by_tid: dict[str, list[dict[str, Any]]],
    story: str,
    act: dict[str, Any],
    history: dict[str, Any],
    catalog: LS.LyricCatalog,
    llm_call,
    gag_prompt: str,
    cut_target: float | None = None,
    require_budgeted_exit: bool = False,
    start_rank: int = 0,
) -> dict[str, Any] | None:
    """Try songs in lyric order; every accepted junction is acoustically green."""
    for lyric_rank, tb in enumerate(lyric_order, 1):
        # Resume point for backtracking: a B that led to a dead end one hop
        # later is skipped along with everything the caller already tried.
        if lyric_rank <= start_rank:
            continue
        routes = routes_by_tid.get(tb) or []
        if not routes:
            continue
        advice = PF.key_advice(A.KEYS.get(ta, {}).get("key"), A.KEYS.get(tb, {}).get("key"))
        key_shift = int(advice.get("suggested keysync", 0) or 0)
        options = enumerate_legal_plans(ta, tb, routes, key_shift, cut_target)
        if require_budgeted_exit and options:
            budgeted = [o for o in options
                        if _has_budgeted_exit(tb, float(o["cue_in"]))]
            if not budgeted:
                _emit(rid, "stage", {"msg": (
                    f"✗ {A.REG[tb]['title']} has no valid entry points within length budget"
                    "exit anchors point (selecting it will cause the entire set to exceed time limits); try the next track based on lyrics ranking."
                )})
                continue
            options = budgeted
            for number, option in enumerate(options, 1):
                option["option_no"] = number
        if not options:
            _emit(rid, "stage", {"msg": (
                f"✗ {A.REG[tb]['title']} has no Tech-first path among {len(routes)}"
                "Any parameters combination passing pre-render check; try next song based on lyrics ranking."
            )})
            continue
        _emit(rid, "stage", {"msg": (
            f"Interlude {hop}: lyrics rank {lyric_rank} {A.REG[tb]['title']}——"
            f"{len(routes)} paths enumerated {len(options)} options passing all pre-render checks."
        )})
        payload = {
            "A (outgoing)": _brief_card(ta),
            "B (lyrics ranking is fixed, title cannot be changed)": _brief_card(tb),
            "Options that have passed all pre-render checks (select only from here)": [
                {k: v for k, v in row.items()
                 if k not in ("_plan", "musical_evidence")}
                for row in options
            ],
            "key advice (computed)": advice,
            "task": (
                "Output only option_no. The title, cue_out, cue_in, moves, and parameters of each option are locked."
                "And has passed the pre-rendering checks for C1/C1b/C2/C2b/C2c. Select the one with the best listening experience:"
                "Overlap windows (blendecho/blend16) typically have less dead air than sequential ones (echo);"
                "filter-type leads require the shortest lead-in. Do not use story or lyrics to offset any acoustic conditions."
            ),
        }
        messages = [{"role": "system", "content": PF.GUIDE + PF.BLEND_FEWSHOT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        remaining = {row["option_no"]: row for row in options}
        for attempt in range(min(PLANS_PER_SONG, len(options))):
            label = f"h{hop}-lyrics{lyric_rank}-plan{attempt + 1}"
            if attempt == 0 or REQUIRE_DIRECT_FIT:
                option = remaining.pop(min(remaining))
                choice = {"option_no": option["option_no"],
                          "why": "Try one by one sorted by gap risk"}
            else:
                _emit
                choice, _ = LD.llm_stream(
                    rid, label, messages, max_tokens=400, temperature=0.45,
                    reasoning=False, json_only=True,
                    json_schema={
                        "type": "object",
                        "properties": {
                            "option_no": {"type": "integer", "enum": sorted(remaining)},
                            "why": {"type": "string", "maxLength": 200},
                        },
                        "required": ["option_no", "why"],
                        "additionalProperties": False,
                    },
                )
                picked = (choice or {}).get("option_no")
                option = remaining.pop(picked, None)
                if option is None:
                    if not remaining:
                        break
                    option = remaining.pop(min(remaining))
            plan = option["_plan"]
            _emit(rid, "stage", {"msg": (
                f"Interlude {hop}: lyrics rank {lyric_rank} {A.REG[tb]['title']} / "
                f"option {option['option_no']}（{option['exit_tool']}→{option['entry_tool']}，"
                f"Attempt {attempt + 1}/{min(PLANS_PER_SONG, len(options))})"
            )})
            pre_fails = presim_fails(ta, tb, plan)
            if pre_fails:
                _emit(rid, "stage", {"msg": (
                    f"  Pre-render (with audio) check blocked: {'；'.join(pre_fails)}"
                )})
                messages = messages[:2] + [
                    {"role": "assistant", "content": json.dumps(choice, ensure_ascii=False)},
                    {"role": "user", "content": json.dumps({
                        f"option{option['option_no']} pre-render failure": pre_fails,
                        "instruction": "Do not change the song; select another option_no. Do not relax any threshold.",
                    }, ensure_ascii=False)},
                ]
                continue
            tag = f"{rid}_h{hop}_ly{lyric_rank}_r{attempt + 1}"
            try:
                rendered_checker_fails = PF.render_free(
                    f"{A.VR}/live_tmp", {"ta": ta, "tb": tb, "plan": plan}, tag
                ) or []
                shutil.rmtree(f"{LD.OUT}/{tag}", ignore_errors=True)
                shutil.move(f"{PF.OUT}/s/{tag}", f"{LD.OUT}/{tag}")
            except Exception as exc:
                _emit(rid, "error", {"tag": label, "msg": f"Render failed {exc!r}"})
                break
            scorecard, _ = LD.scorecard(f"{LD.OUT}/{tag}", ta, tb, plan,
                                        mode="short", skip_c4=True)
            score_fails = [name for name, row in scorecard.items() if row.get("verdict") == "FAIL"]
            regression_fails = story_mode_regression_fails(rendered_checker_fails, plan)
            hard_fails = len(score_fails) + len(regression_fails)
            _emit(rid, "verdict", {
                "hop": hop, "lyric_rank": lyric_rank, "plan": plan,
                "scorecard": scorecard, "scorecard_fails": score_fails,
                "rendered_checker_fails": regression_fails, "n_fail": hard_fails,
                "audio": f"/live/audio/{tag}/trans1.wav",
            })
            if hard_fails == 0:
                try:
                    analysis = LS.analyze_single_full_lyrics(
                        story, act, history, tb, catalog, A.REG, llm_call, gag_prompt
                    )
                except LS.LyricsStoryError as exc:
                    if REQUIRE_DIRECT_FIT:
                        _emit(rid, "stage", {"msg": (
                            f"✗ {A.REG[tb]['title']} is fully green acoustically, but focused lyrics review failed"
                            f"({exc}); strict mode skips this, tries next song based on lyrics ranking."
                        )})
                        break
                    _emit(rid, "stage", {"msg": (
                        f"{A.REG[tb]['title']} is fully green acoustically; focused lyrics review failed ({exc}), "
                        "Keep this transition and reuse the lyrics analysis from the sorting stage."
                    )})
                    analysis = None
                else:
                    if (REQUIRE_DIRECT_FIT and analysis is not None
                            and analysis.get("story_fit") not in _ACCEPTED_FITS):
                        _emit(rid, "stage", {"msg": (
                            f"✗ {A.REG[tb]['title']} is fully green acoustically, but review verdict is"
                            f"「{analysis.get('story_fit')}」; Strict mode not adopted."
                        )})
                        break
                return {
                    "tb": tb, "plan": plan, "analysis": analysis,
                    "lyric_rank": lyric_rank, 
                    "preview_tag": tag,
                }
            messages = messages[:2] + [
                {"role": "assistant", "content": json.dumps(choice, ensure_ascii=False)},
                {"role": "user", "content": json.dumps({
                    "Original Scorecard failed": LD.fam_feedback(scorecard),
                    "rendered regression checker failed": regression_fails,
                    "instruction": (
                        "song cannot be changed; only select another option_no for this song (each has been approved"
                        "Pre-render check. No red lights may be waived or ignored."
                    ),
                }, ensure_ascii=False)},
            ]
        _emit(rid, "stage", {"msg": (
            f"✗ {A.REG[tb]['title']} never had a full-green bridge; try next song by lyrics rank."
        )})
    return None


_RIDEABLE_CACHE: dict[str, int] = {}


def _arc_conflicts(description: str, prev_arousal: Any, arousal: Any) -> bool:
    """True when this candidate would move the arc the wrong way."""
    direction = LS.act_energy_direction(description)
    if direction is None:
        return False
    try:
        previous, current = float(prev_arousal), float(arousal)
    except (TypeError, ValueError):
        return False
    tolerance = float(os.environ.get("AIDJ_ARC_TOLERANCE", "0.12"))
    step = float(os.environ.get("AIDJ_MIN_ARC_STEP", "0"))
    if direction == "up":
        if step > 0:
            return current < previous + step
        return current < previous - tolerance
    if step > 0:
        return current > previous - step
    return current > previous + tolerance


def _rank_musical_pool(
    rid: str, story: str, act: dict[str, Any], history: dict[str, Any],
    musical_tids: list[str], catalog: LS.LyricCatalog, llm_call, gag_prompt: str,
    prev_arousal: Any = None, arousal_ceiling: float | None = None,
    arousal_floor: float | None = None,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    ranking = LS.select_lyric_ranking(
        story, act, history, musical_tids, catalog, A.REG, llm_call, gag_prompt
    )
    _emit(rid, "stage", {"msg": (
        "top 8 after ranking："
        + "、".join(f"{i}.{A.REG[tid]['title']}"
                    for i, tid in enumerate(ranking.shortlist_tids[:8], 1))
    )})
    leading_tids = list(ranking.shortlist_tids[:FINALISTS])
    leading = LS.SelectionBundle(
        leading_tids[0], tuple(leading_tids),
        {tid: ranking.analyses[tid] for tid in leading_tids},
    )
    full = LS.select_from_full_lyrics(
        story, act, history, leading, catalog, A.REG, llm_call, gag_prompt
    )
    order = [full.selected_tid]
    order.extend(tid for tid in full.shortlist_tids if tid not in order)
    order.extend(tid for tid in ranking.shortlist_tids if tid not in order)
    def _rideable(tid: str) -> int:
        """Can this song *receive* a ride hand-over?"""
        cached = _RIDEABLE_CACHE.get(tid)
        if cached is not None:
            return cached
        value = 0
        try:
            value = int(bool(list(LT.entry_candidates(tid, "lead16"))))
        except Exception:
            value = 0
        _RIDEABLE_CACHE[tid] = value
        return value

    description = str(act.get("description", ""))

    def _order_key(tid: str) -> tuple[int, float, int]:
        row = analyses_all.get(tid, {})
        passed = int(row.get("story_fit") in ("direct fit", "partial fit"))
        energy = LS.energy_rank_key(description, A.REG.get(tid, {}).get("arousal"))
        return passed, energy, _rideable(tid)

    analyses_all = dict(ranking.analyses)
    analyses_all.update(full.analyses)
    order = sorted(order, key=_order_key, reverse=True)
    analyses = dict(ranking.analyses)
    analyses.update(full.analyses)
    def _within_arc(tid: str) -> bool:
        """The same window verify_set.py holds the finished set to."""
        value = float(A.REG.get(tid, {}).get("arousal") or 0.0)
        if _arc_conflicts(description, prev_arousal, value):
            return False
        if arousal_floor is not None and value < arousal_floor:
            return False
        if arousal_ceiling is not None and value > arousal_ceiling:
            return False
        return True

    climbing = [tid for tid in order if _within_arc(tid)]
    if climbing and len(climbing) < len(order):
        _emit(rid, "stage", {"msg": (
            f"Energy arc: {len(order)} tracks, {len(order) - len(climbing)} will cause"
            f"The arc direction is wrong; this section is excluded."
        )})
        order = climbing
    elif not climbing:
        if arousal_floor is not None or arousal_ceiling is not None:
            _emit(rid, "stage", {"msg": (
                f"energy arc: {len(order)} candidates none fall within this act energy window"
                f"(Requires ≥ {arousal_floor:.2f}"
                + (f"、≤ {arousal_ceiling:.2f}" if arousal_ceiling is not None else "")
                + "), this jump is not viable."
            )})
            return [], analyses
        _emit(rid, "stage", {"msg": (
            "Energy arc: no candidate can follow the correct arc direction; temporarily allowing all candidates."
        )})
    if REQUIRE_DIRECT_FIT:
        kept = [tid for tid in order
                if analyses.get(tid, {}).get("story_fit") in _ACCEPTED_FITS]
        if kept:
            _emit(rid, "stage", {"msg": (
                f"Strict mode: among {len(order)} musically qualified titles, only {len(kept)} are direct fits,"
                "Others are excluded from this act."
            )})
            order = kept
        elif order:
            order = sorted(
                order,
                key=lambda tid: -_FIT_RANK.get(
                    str(analyses.get(tid, {}).get("story_fit", "")), 0),
            )
            _emit(rid, "stage", {"msg": (
                f"Strict mode: in the sorting phase of compressed lyrics, none of the {len(order)} songs are direct fits,"
                "But this stage is stricter than reviewing full lyrics one by one. Change to sending for review one by one after sorting by verdict,"
                "Ultimately only accepts direct fit."
            )})
    _emit(rid, "lyric_shortlist", {
        "act": act, "tids": order,
        "titles": [A.REG[tid]["title"] for tid in order],
    })
    return order, analyses


def pipeline_story_set(rid: str, seed_tid: str | None = None, story: str | None = None) -> None:
    """Generate exactly three songs; any acoustic red light aborts acceptance."""
    del seed_tid  # Story mode always selects A from lyrics.
    try:
        story = str(story or "").strip()
        if not story:
            raise LS.LyricsStoryError("story must not be empty")
        acts = story_acts(story)
        catalog = LS.LyricCatalog()
        gag_prompt = LS.load_gag_prompt()
        llm_call = lyric_llm_call(rid)
        eligible = _eligible_lyrics(catalog)
        if len(eligible) < 3:
            raise LS.LyricsStoryError(f"The intersection of available Song Cards, audio files, and full lyrics contains only {len(eligible)} songs.")
        eligible = LS.set2_viable_openers(eligible, A.REG)
        forced = forced_order()
        if forced:
            missing = [tid for tid in forced if tid not in A.REG]
            if missing:
                raise LS.LyricsStoryError(f"AIDJ_FORCE_ORDER: contains non-existent tid: {missing}")
            if len(forced) > len(acts):
                raise LS.LyricsStoryError(
                    f"AIDJ_FORCE_ORDER: max {len(acts)} tracks, received {len(forced)} tracks")

        def _hop_pool(hop: int) -> set[str]:
            """Which songs this hop is allowed to hand over to."""
            if forced and hop < len(forced):
                return {forced[hop]}
            return set(eligible)
        _emit(rid, "story_acts", {"story": story, "acts": acts, "eligible": len(eligible)})
        _emit(rid, "stage", {"msg": f"Score eligible opening tracks one by one in Act 1 {len(eligible)} titles…"})
        if forced:
            pinned = "、".join(
                f"{chr(ord('A') + i)}={A.REG[tid]['title']}"
                for i, tid in enumerate(forced)
            )
            _emit(rid, "stage", {"msg": (
                f"Specified sequence mode: {pinned} (others selected by planner); still verify lyrics and rendering track by track."
            )})
            opening = LS.SelectionBundle(
                forced[0], (forced[0],),
                {forced[0]: LS.analyze_single_full_lyrics(
                    story, acts[0], LS.opening_history(catalog, forced[0]),
                    forced[0], catalog, A.REG, llm_call, gag_prompt)},
            )
        else:
            opening = LS.select_opening_song(
                story, acts[0], eligible, catalog, A.REG, llm_call, gag_prompt
            )
        opener_order = [opening.selected_tid]
        opener_order.extend(tid for tid in opening.shortlist_tids if tid not in opener_order)
        def _pool_viable(pool: Iterable[str], act: dict[str, Any],
                         prev_arousal: Any) -> int:
            """How many of this opener's successors could actually tell the act."""
            description = str(act.get("description", ""))
            count = 0
            for tid in pool:
                arousal = A.REG.get(tid, {}).get("arousal")
                if LS.arousal_conflicts_with_act(description, arousal):
                    continue
                if _arc_conflicts(description, prev_arousal, arousal):
                    continue
                count += 1
            return count

        trials = max(1, int(os.environ.get("AIDJ_OPENER_TRIALS", "4")))
        scored_openers: list[tuple[tuple[int, int, int], str, dict[str, Any]]] = []
        for candidate_a in opener_order[:trials]:
            _emit(rid, "stage", {"msg": (
                f"Opening candidate {A.REG[candidate_a]['title']}: verify Tech-first next hop and"
                "Evaluate whether its candidate pool can cover the last two acts."
            )})
            candidate_routes = discover_musical_routes(
                rid, candidate_a, acts[1], _hop_pool(1), songs_to_skip([candidate_a])
            )
            if not candidate_routes:
                _emit(rid, "stage", {"msg": (
                    f"{A.REG[candidate_a]['title']} has no next hop passing Tech-first retrieval."
                )})
                continue
            opener_arousal = A.REG.get(candidate_a, {}).get("arousal")
            viable2 = _pool_viable(candidate_routes, acts[1], opener_arousal)
            viable3 = _pool_viable(candidate_routes, acts[2], opener_arousal)
            own_fit = _FIT_RANK.get(
                str((opening.analyses.get(candidate_a) or {}).get("story_fit", "")), 0
            )
            scored_openers.append(
                ((own_fit, min(viable2, viable3), viable2 + viable3,
                  len(candidate_routes)),
                 candidate_a, candidate_routes)
            )
            _emit(rid, "stage", {"msg": (
                f"{A.REG[candidate_a]['title']}: candidate pool {len(candidate_routes)} songs,"
                f"Act 2 can use {viable2} lines, Act 3 can use {viable3} lines."
            )})
        if not scored_openers:
            raise LS.LyricsStoryError(
                f"The top {trials} opening tracks cannot generate any next-hop that passes the Tech-first acoustic retrieval"
            )
        step = float(os.environ.get("AIDJ_MIN_ARC_STEP", "0"))
        if step > 0 and len(acts) >= 3:
            last_peak = LS.act_demands_peak_energy(str(acts[-1]["description"]))
            first_peak = LS.act_demands_peak_energy(str(acts[0]["description"]))
            if last_peak and not first_peak:
                ceiling = 1.0 - 2.0 * step
                roomy = [row for row in scored_openers
                         if float(A.REG[row[1]].get("arousal") or 0.0) <= ceiling]
                _emit(rid, "stage", {"msg": (
                    f"energy arc needs at least +{step:.2f} per act; opening track arousal must not exceed"
                    f"{ceiling:.2f}; among {len(scored_openers)} opening candidates, {len(roomy)} have space."
                )})
                if roomy:
                    scored_openers = roomy
        scored_openers.sort(key=lambda row: row[0], reverse=True)
        if REQUIRE_DIRECT_FIT:
            strict_openers = [row for row in scored_openers
                              if row[0][0] == _FIT_RANK["direct fit"]]
            _emit(rid, "stage", {"msg": (
                f"Strict mode: among {len(scored_openers)} musically qualified openers,"
                f"{len(strict_openers)} are direct fits."
            )})
            if not strict_openers:
                raise LS.LyricsStoryError(
                    f"Strict mode: none of the top {trials} opener candidates are direct fits"
                )
            scored_openers = strict_openers
        opener_failure = ""
        for _opener_score, chosen_a, routes_ab in scored_openers:
            _fit = str((opening.analyses.get(chosen_a) or {}).get("story_fit", "?"))
            _emit(rid, "stage", {"msg": (
                f"Try opening {A.REG[chosen_a]['title']} (Act 1 verdict={_fit}): in "
                f"{len(scored_openers)} musically qualified opening tracks, its own verdict is highest,"
                "The candidate pool best bridges the last two acts."
            )})
            opening_analysis = opening.analyses.get(chosen_a, opening.selected_analysis())
            order = [chosen_a]
            selections = [{
                "position": "A", "tid": chosen_a, "title": A.REG[chosen_a]["title"],
                "act": acts[0], **opening_analysis,
            }]
            _emit(rid, "lyric_selection", selections[-1])
            plans: list[dict[str, Any]] = []
            hops: list[dict[str, Any]] = []

            MAX_BACKTRACKS = max(1, int(os.environ.get("AIDJ_MAX_BACKTRACKS", "6")))
            base_order, base_selections = list(order), list(selections)

            def _run_hop(hop: int, preset_routes: dict[str, list[dict[str, Any]]] | None,
                         skip: int) -> tuple[dict[str, Any], dict[str, dict[str, str]]] | None:
                ta = order[-1]
                history = (
                    LS.opening_history(catalog, ta)
                    if hop == 1 else LS.history_for_c(catalog, order[0], ta, plans[0])
                )
                # Song `ta` entered the set at plans[-1]["entry"], so its audible
                # share is cut - entry: aim the exit one budget later.
                cut_target = (None if hop == 1
                              else float(plans[-1]["entry"]) + _song_budget())
                routes = preset_routes if preset_routes is not None else discover_musical_routes(
                    rid, ta, acts[hop], _hop_pool(hop), songs_to_skip(order),
                    plans[-1]["entry"],
                    cut_target=cut_target,
                )
                if not routes:
                    _emit(rid, "stage", {"msg": (
                        f"No titles in hop {hop} passed Tech-first acoustic retrieval."
                    )})
                    return None
                _emit(rid, "stage", {"msg": (
                    f"Music candidate pool for hop {hop} is final: {len(routes)} tracks; now sorting by lyrics for act {hop + 1}."
                )})
                # How many climbs still have to happen after this act, and so
                # how much arousal headroom this song must leave behind it.
                _step = float(os.environ.get("AIDJ_MIN_ARC_STEP", "0"))
                _ceiling = _floor = None
                if _step > 0 and len(acts) >= 3:
                    if (LS.act_demands_peak_energy(str(acts[-1]["description"]))
                            and not LS.act_demands_peak_energy(
                                str(acts[0]["description"]))):
                        _ceiling = 1.0 - _step * (len(acts) - 1 - hop)
                        _floor = (float(A.REG.get(ta, {}).get("arousal") or 0.0)
                                  + _step)
                lyric_order, analyses = _rank_musical_pool(
                    rid, story, acts[hop], history, list(routes), catalog,
                    llm_call, gag_prompt,
                    prev_arousal=A.REG.get(ta, {}).get("arousal"),
                    arousal_ceiling=_ceiling, arousal_floor=_floor,
                )
                accepted = try_ranked_candidates(
                    rid, hop, ta, lyric_order, routes, story, acts[hop], history,
                    catalog, llm_call, gag_prompt,
                    cut_target=cut_target,
                    # Only the middle song's share is rigid, so B is the one choice
                    # that has to be made with the next hand-off already in mind.
                    require_budgeted_exit=(hop == 1),
                    start_rank=skip,
                )
                if accepted is None:
                    _emit(rid, "stage", {"msg": (
                        f"All songs that passed Tech-first retrieval in hop {hop} fail the full Scorecard."
                    )})
                    return None
                return accepted, analyses

            def _commit(hop: int, accepted: dict[str, Any],
                        analyses: dict[str, dict[str, str]]) -> None:
                ta, tb, plan = order[-1], accepted["tb"], accepted["plan"]
                order.append(tb)
                plans.append(plan)
                analysis = accepted.get("analysis") or analyses[tb]
                selection = {
                    "position": chr(ord("A") + hop), "tid": tb,
                    "title": A.REG[tb]["title"], "act": acts[hop],
                    "lyric_rank": accepted["lyric_rank"], **analysis,
                }
                selections.append(selection)
                _emit(rid, "lyric_selection", selection)
                hops.append({
                    "hop": hop, "a": A.REG[ta]["title"], "b": A.REG[tb]["title"],
                    "plan": plan, "n_fail": 0,
                })

            b_skip = 0
            opener_ok = True
            for backtrack in range(MAX_BACKTRACKS):
                order[:], plans[:], selections[:], hops[:] = (
                    list(base_order), [], list(base_selections), [])
                first = _run_hop(1, routes_ab, b_skip)
                if first is None:
                    opener_failure = (
                        "All songs that passed Tech-first retrieval in hop 1 fail the full Scorecard"
                    )
                    opener_ok = False
                    break
                _commit(1, *first)
                # Anything at or before this B is exhausted for the next attempt.
                b_skip = int(first[0]["lyric_rank"])
                second = _run_hop(2, None, 0)
                if second is not None:
                    _commit(2, *second)
                    break
                _emit(rid, "stage", {"msg": (
                    f"↩ 2nd hop from 《{A.REG[order[-1]]['title']}》 failed;"
                    f"Go back to step 1 to select B after the {b_skip + 1}th rank ({backtrack + 1}th backtrack)."
                )})
            else:
                opener_failure = f"After backtracking {MAX_BACKTRACKS} times, still unable to complete the 2nd hop"
                opener_ok = False
            if opener_ok:
                break
            _emit(rid, "stage", {"msg": (
                f"↩ Opening 《{A.REG[chosen_a]['title']}》 failed ({opener_failure});"
                "Switch to the next opening candidate."
            )})
        else:
            raise LS.LyricsStoryError(
                opener_failure or "All musically qualified opening tracks cannot complete the entire set"
            )

        out_dir = f"{LD.OUT}/{rid}_story_set"
        os.makedirs(out_dir, exist_ok=True)
        PF.M.FREE_MODE = True
        try:
            middle = sum(
                float(plans[index + 1]["cut"]) - float(plans[index]["entry"])
                for index in range(len(plans) - 1)
            )
            end_share = (SET_TARGET_SEC - middle) / 2.0
            end_share = max(MIN_SONG_SHARE_SEC, min(MAX_SONG_SHARE_SEC, end_share))
            entry0 = _opener_start(order[0], float(plans[0]["cut"]), end_share)
            opener_share = float(plans[0]["cut"]) - entry0
            tail_share = SET_TARGET_SEC - middle - opener_share
            tail_share = max(MIN_SONG_SHARE_SEC, min(MAX_SONG_SHARE_SEC, tail_share))
            final_cut = min(
                A.REG[order[-1]]["duration_sec"] - 0.5,
                float(plans[-1]["entry"]) + tail_share,
            )
            marks, duration = PF.M.render_set(
                order, plans, out_dir, entry0_override=entry0, final_cut=final_cut
            )
        finally:
            PF.M.FREE_MODE = False
        shares = [round(float(plans[0]["cut"]) - entry0, 1)] + [
            round(float(plans[index + 1]["cut"]) - float(plans[index]["entry"]), 1)
            for index in range(len(plans) - 1)
        ] + [round(final_cut - float(plans[-1]["entry"]), 1)]
        trace = {
            "order_tids": order,
            "junctions": [{"plan": plan} for plan in plans],
            "junction_times": [round(float(mark), 2) for mark in marks],
            "planner": "lyrics-tech-first-set",
            "story": story,
            "selections": selections,
            "entry0": round(float(entry0), 2),
            "final_cut": round(float(final_cut), 2),
            "per_song_share_sec": shares,
            "set_target_sec": SET_TARGET_SEC,
            "set_duration_sec": round(float(duration), 1),
        }
        with open(f"{out_dir}/set_trace.json", "w", encoding="utf-8") as handle:
            json.dump(trace, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        final_checks = RC.check_run(out_dir)
        # Filter each junction against its *own* plan, so the ride C2c exemption
        # is applied only to the junctions that actually ride.
        final_fails = [
            failure
            for index, row in enumerate(final_checks)
            for failure in story_mode_regression_fails(
                row.get("fails", []),
                plans[index] if index < len(plans) else None,
            )
        ]
        _emit(rid, "final_check", {"junctions": final_checks, "fails": final_fails})
        if final_fails:
            raise LS.LyricsStoryError(
                "Full Set re-render failed regression checker, delivery rejected:"
                + " | ".join(final_fails)
            )
        _emit(rid, "setfinal", {
            "audio": f"/live/audio/{rid}_story_set/set.wav",
            "order": [A.REG[tid]["title"] for tid in order],
            "hops": hops, "junction_times": [round(float(mark), 1) for mark in marks],
            "dur": round(float(duration), 1), "story": story,
            "per_song_share_sec": shares, "set_target_sec": SET_TARGET_SEC,
            "selections": selections, "final_checker": "ALL PASS",
        })
    except Exception as exc:
        import traceback
        _emit(rid, "error", {"tag": "story_set", "msg": str(exc)[:800],
                             "traceback": traceback.format_exc()[-1500:]})
    finally:
        LD.RUNS[rid]["done"] = True
        _emit(rid, "done", {})
