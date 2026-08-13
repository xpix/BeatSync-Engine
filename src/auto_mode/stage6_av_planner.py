#!/usr/bin/env python3
"""Audio-visual clip planner for Auto Mode."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from typing import Dict, List, Sequence

import numpy as np


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _stable_rng(*parts) -> random.Random:
    raw = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    return random.Random(seed)


def _buffered_start_bounds(video_duration: float, source_duration: float, edge_buffer_seconds: float) -> tuple[float, float]:
    """Valid [lo, hi] start-time range that keeps clips out of the video's edge buffer.

    Shrinks gracefully (rather than rejecting the video) when it is too short to fit the buffer.
    """
    max_start_full = max(0.0, video_duration - source_duration)
    buffer = max(0.0, edge_buffer_seconds)
    lo = min(buffer, max_start_full)
    hi = max(lo, max_start_full - buffer)
    return lo, hi


def build_planned_clip_sequence(
    cut_times: Sequence[float],
    segment_durations: Sequence[float],
    beat_info: Dict | None,
    video_files: Sequence[str],
    strict_unique_non_overlap: bool = True,
    preferred_videos: Sequence[str] = (),
    edge_buffer_seconds: float = 5.0,
) -> List[Dict]:
    """Build exact source clip choices for every output segment.

    Returns an empty list when no visual library is present, which tells the
    renderer to keep its old fallback sampling.

    edge_buffer_seconds: portion at the very start/end of each source video that
    is never used for a clip (e.g. to skip intros/outros or unstable footage).
    """
    edge_buffer_seconds = max(0.0, float(edge_buffer_seconds))
    beat_info = beat_info or {}
    video_analysis = beat_info.get("video_analysis") or {}
    candidates = list(video_analysis.get("candidates") or [])
    candidates = [c for c in candidates if c.get("video_file")]
    if not candidates:
        return []

    preferred_set = {str(p) for p in preferred_videos if p}

    cut_times_arr = np.asarray(cut_times, dtype=float)
    durations_arr = np.asarray(segment_durations, dtype=float)
    if cut_times_arr.size < 2 or durations_arr.size == 0:
        return []

    profiles = _build_segment_profiles(cut_times_arr, durations_arr, beat_info)
    recent_ids = deque(maxlen=10)
    recent_videos = deque(maxlen=5)
    usage = Counter()
    used_candidate_ids: set[str] = set()
    occupied_ranges: Dict[str, List[tuple[float, float]]] = {}
    planned: List[Dict] = []

    for i, profile in enumerate(profiles):
        if strict_unique_non_overlap:
            planned_clip = _choose_and_materialize_candidate(
                candidates=candidates,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                used_candidate_ids=used_candidate_ids,
                occupied_ranges=occupied_ranges,
                preferred_videos=preferred_set,
                edge_buffer_seconds=edge_buffer_seconds,
            )
        else:
            candidate = _choose_relaxed_candidate(
                candidates=candidates,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                preferred_videos=preferred_set,
            )
            planned_clip = (
                _materialize_clip(candidate, profile, i, edge_buffer_seconds=edge_buffer_seconds)
                if candidate else None
            )

        if not planned_clip:
            continue
        candidate_id = str(planned_clip.get("candidate_id") or "")
        video_file = str(planned_clip.get("video_file") or "")
        start_time = float(planned_clip.get("start_time") or 0.0)
        source_duration = max(0.05, float(planned_clip.get("source_duration") or 0.05))
        end_time = start_time + source_duration

        planned.append(planned_clip)
        if candidate_id:
            if strict_unique_non_overlap:
                used_candidate_ids.add(candidate_id)
            recent_ids.append(candidate_id)
            usage[candidate_id] += 1
        if video_file:
            recent_videos.append(video_file)
            usage[video_file] += 1
            if strict_unique_non_overlap:
                occupied_ranges.setdefault(video_file, []).append((start_time, end_time))

    if len(planned) != len(durations_arr):
        return []
    return planned


def summarize_clip_plan(plan: Sequence[Dict]) -> Dict:
    if not plan:
        return {"clip_count": 0, "targets": {}, "ai_tagged": 0}
    targets = Counter(str(item.get("target", "flow")) for item in plan)
    ai_tagged = sum(1 for item in plan if item.get("ai_analyzed"))
    source_count = len(set(item.get("video_file") for item in plan))
    return {
        "clip_count": len(plan),
        "targets": dict(targets),
        "ai_tagged": ai_tagged,
        "source_count": source_count,
    }


def _build_segment_profiles(cut_times: np.ndarray, segment_durations: np.ndarray, beat_info: Dict) -> List[Dict]:
    beat_times = np.asarray(beat_info.get("times", []), dtype=float)
    energy_profile = beat_info.get("energy_profile") or {}
    rhythm_data = beat_info.get("rhythm_data") or {}
    sections = beat_info.get("sections") or []

    wave = np.asarray(energy_profile.get("wave", []), dtype=float)
    arc = np.asarray(energy_profile.get("arc", []), dtype=float)
    impact = np.asarray(rhythm_data.get("impact_strength", []), dtype=float)
    rhythm = np.asarray(rhythm_data.get("combined_strength", []), dtype=float)
    novelty = np.asarray(rhythm_data.get("novelty_strength", []), dtype=float)

    profiles: List[Dict] = []
    for i, duration in enumerate(segment_durations):
        start = float(cut_times[i])
        end = float(cut_times[i + 1])
        mid = (start + end) * 0.5
        local_wave = _interp_feature(mid, beat_times, wave, 0.5)
        local_arc = _interp_feature(mid, beat_times, arc, 0.5)
        local_impact = _interp_feature(start, beat_times, impact, 0.5)
        local_rhythm = _interp_feature(start, beat_times, rhythm, 0.5)
        local_novelty = _interp_feature(start, beat_times, novelty, 0.4)
        section = _section_at(sections, mid)
        target = _target_for_segment(section, local_wave, local_impact, local_rhythm, local_novelty, local_arc)
        profiles.append({
            "index": i,
            "start": start,
            "end": end,
            "duration": float(duration),
            "mid": mid,
            "wave": local_wave,
            "impact": local_impact,
            "rhythm": local_rhythm,
            "novelty": local_novelty,
            "arc": local_arc,
            "section": section,
            "section_type": section.get("type", "body") if section else "body",
            "target": target,
        })
    return profiles


def _interp_feature(time_s: float, beat_times: np.ndarray, values: np.ndarray, default: float) -> float:
    if beat_times.size == 0 or values.size != beat_times.size:
        return default
    return _clamp(np.interp(time_s, beat_times, values, left=float(values[0]), right=float(values[-1])), default=default)


def _section_at(sections: Sequence[Dict], time_s: float) -> Dict:
    for section in sections:
        if float(section.get("start", 0.0)) <= time_s < float(section.get("end", 0.0)):
            return section
    return sections[-1] if sections else {}


def _target_for_segment(section: Dict, wave: float, impact: float, rhythm: float, novelty: float, arc: float) -> str:
    section_type = section.get("type", "body")
    if section_type in {"drop", "finale"} and (wave >= 0.58 or impact >= 0.55):
        return "drop"
    if impact >= 0.76 or (wave >= 0.78 and rhythm >= 0.62):
        return "drop"
    if section_type in {"breakdown", "intro", "outro"} and wave <= 0.54:
        return "soft"
    if wave <= 0.32 and impact <= 0.48:
        return "soft"
    if section_type in {"bridge", "hook"} or novelty >= 0.68 or (arc >= 0.62 and wave >= 0.48):
        return "build"
    if wave >= 0.58 and rhythm >= 0.54:
        return "rhythm"
    return "flow"


def _choose_and_materialize_candidate(
    candidates: Sequence[Dict],
    profile: Dict,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
    used_candidate_ids: set[str],
    occupied_ranges: Dict[str, List[tuple[float, float]]],
    preferred_videos: set[str] = frozenset(),
    edge_buffer_seconds: float = 0.0,
) -> Dict | None:
    ranked: List[tuple[float, Dict]] = []
    rng = _stable_rng(index, profile.get("target"), profile.get("start"))
    required_source = max(0.05, float(profile.get("duration", 0.05)))

    for candidate in candidates:
        cid = str(candidate.get("id") or "")
        if cid and cid in used_candidate_ids:
            continue

        score = _score_candidate(candidate, profile)
        video_file = str(candidate.get("video_file") or "")
        is_preferred = video_file in preferred_videos

        if cid in recent_ids:
            score -= 0.28
        if video_file in recent_videos:
            score -= 0.05 if is_preferred else 0.10
        score -= min(0.28, usage[cid] * 0.10)
        score -= min(0.18, usage[video_file] * (0.004 if is_preferred else 0.012))
        if is_preferred:
            score += 0.32

        candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
        if candidate_duration < required_source * 0.55:
            score -= 0.18

        score += rng.random() * 0.015
        ranked.append((score, candidate))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    for _, candidate in ranked:
        start_time = _select_non_overlapping_start(candidate, profile, occupied_ranges, edge_buffer_seconds)
        if start_time is None:
            continue
        return _materialize_clip(
            candidate=candidate,
            profile=profile,
            index=index,
            start_time=start_time,
        )

    return None


def _select_non_overlapping_start(
    candidate: Dict,
    profile: Dict,
    occupied_ranges: Dict[str, List[tuple[float, float]]],
    edge_buffer_seconds: float = 0.0,
) -> float | None:
    source_duration = max(0.05, float(profile.get("duration", 0.05)))
    start = float(candidate.get("start", 0.0))
    end = float(candidate.get("end", start + source_duration))
    window_start = max(0.0, min(start, end))
    window_end = max(window_start, max(start, end))

    video_duration = float(candidate.get("video_duration", window_end))
    allowed_lo, allowed_hi = _buffered_start_bounds(video_duration, source_duration, edge_buffer_seconds)
    window_start = max(window_start, allowed_lo)
    window_end = min(window_end, allowed_hi + source_duration)

    max_start = window_end - source_duration
    if max_start < window_start:
        return None

    video_file = str(candidate.get("video_file") or "")
    preferred = _preferred_start_in_window(candidate, profile, source_duration, window_start, max_start)
    occupied = occupied_ranges.get(video_file, [])
    return _pick_start_from_available_gaps(window_start, window_end, source_duration, occupied, preferred)


def _preferred_start_in_window(
    candidate: Dict,
    profile: Dict,
    source_duration: float,
    window_start: float,
    max_start: float,
) -> float:
    target = profile.get("target", "flow")
    if target == "drop":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.36
    elif target == "soft":
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.50
    elif target == "build":
        anchor = float(candidate.get("peak_time", candidate.get("center", candidate.get("start", 0.0))))
        align = 0.48
    else:
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        align = 0.44

    preferred = anchor - source_duration * align
    return max(window_start, min(preferred, max_start))


def _pick_start_from_available_gaps(
    window_start: float,
    window_end: float,
    source_duration: float,
    occupied: List[tuple[float, float]],
    preferred_start: float,
) -> float | None:
    if window_end - window_start < source_duration:
        return None

    gaps: List[tuple[float, float]] = []
    cursor = window_start
    for occ_start, occ_end in sorted(occupied):
        occ_start = float(occ_start)
        occ_end = float(occ_end)
        if occ_end <= cursor:
            continue
        if occ_start > cursor:
            gap_start = cursor
            gap_end = min(occ_start, window_end)
            if gap_end - gap_start >= source_duration:
                gaps.append((gap_start, gap_end))
        cursor = max(cursor, occ_end)
        if cursor >= window_end:
            break

    if cursor < window_end and window_end - cursor >= source_duration:
        gaps.append((cursor, window_end))

    if not gaps:
        return None

    best_start = None
    best_distance = float("inf")
    for gap_start, gap_end in gaps:
        local_max_start = gap_end - source_duration
        start = max(gap_start, min(preferred_start, local_max_start))
        distance = abs(start - preferred_start)
        if distance < best_distance:
            best_distance = distance
            best_start = start

    return best_start


def _choose_relaxed_candidate(
    candidates: Sequence[Dict],
    profile: Dict,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
    preferred_videos: set[str] = frozenset(),
) -> Dict | None:
    best_candidate = None
    best_score = -999.0
    rng = _stable_rng(index, profile.get("target"), profile.get("start"))

    for candidate in candidates:
        score = _score_candidate(candidate, profile)
        cid = str(candidate.get("id") or "")
        video_file = str(candidate.get("video_file") or "")
        is_preferred = video_file in preferred_videos

        if cid in recent_ids:
            score -= 0.28
        if video_file in recent_videos:
            score -= 0.05 if is_preferred else 0.10
        score -= min(0.28, usage[cid] * 0.10)
        score -= min(0.18, usage[video_file] * (0.004 if is_preferred else 0.012))
        if is_preferred:
            score += 0.32

        required_source = max(0.05, float(profile.get("duration", 0.05)))
        candidate_duration = max(0.05, float(candidate.get("duration", required_source)))
        if candidate_duration < required_source * 0.55:
            score -= 0.18

        score += rng.random() * 0.015
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _score_candidate(candidate: Dict, profile: Dict) -> float:
    target = profile.get("target", "flow")
    semantic = candidate.get("semantic") or {}
    tags = {str(t).lower() for t in candidate.get("tags", [])}
    quality = _clamp(candidate.get("quality_score", semantic.get("visual_quality", 0.5)), default=0.5)
    action = _clamp(candidate.get("action_score", semantic.get("action_intensity", 0.0)))
    beauty = _clamp(candidate.get("beauty_score", semantic.get("beauty_score", 0.0)))
    tension = _clamp(candidate.get("tension_score", 0.0))
    soft = _clamp(candidate.get("soft_score", 0.0))
    motion = _clamp(candidate.get("motion", semantic.get("camera_motion", 0.0)))
    character = _clamp(semantic.get("character_focus", 0.0))
    combat = _clamp(semantic.get("combat", 0.0))
    chase = _clamp(semantic.get("chase", 0.0))
    explosion = _clamp(semantic.get("explosion", 0.0))

    tag_bonus = 0.0
    if target in tags:
        tag_bonus += 0.08
    if target == "drop" and tags.intersection({"action", "combat", "chase", "explosion", "hype"}):
        tag_bonus += 0.12
    if target == "soft" and tags.intersection({"soft", "beauty", "sad"}):
        tag_bonus += 0.10
    if target == "build" and tags.intersection({"tension", "transition"}):
        tag_bonus += 0.10

    if target == "drop":
        match = 0.46 * action + 0.16 * motion + 0.12 * combat + 0.10 * chase + 0.08 * explosion + 0.08 * quality
    elif target == "soft":
        match = 0.45 * beauty + 0.18 * soft + 0.13 * character + 0.14 * (1.0 - action) + 0.10 * quality
    elif target == "build":
        match = 0.40 * tension + 0.18 * motion + 0.15 * character + 0.14 * action + 0.13 * quality
    elif target == "rhythm":
        match = 0.30 * action + 0.24 * motion + 0.18 * quality + 0.16 * tension + 0.12 * beauty
    else:
        match = 0.28 * quality + 0.24 * beauty + 0.20 * action + 0.16 * tension + 0.12 * soft

    brightness = _clamp(candidate.get("brightness", 0.5), default=0.5)
    visibility_penalty = 0.0
    if brightness < 0.13:
        visibility_penalty += 0.18
    if quality < 0.24:
        visibility_penalty += 0.16

    return _clamp(match + tag_bonus + 0.12 * quality - visibility_penalty, lo=-1.0, hi=2.0)


def _materialize_clip(
    candidate: Dict,
    profile: Dict,
    index: int,
    start_time: float | None = None,
    edge_buffer_seconds: float = 0.0,
) -> Dict:
    final_duration = max(0.05, float(profile["duration"]))
    source_duration = final_duration
    target = profile.get("target", "flow")

    if start_time is None:
        video_duration = max(source_duration, float(candidate.get("video_duration", source_duration)))
        allowed_lo, allowed_hi = _buffered_start_bounds(video_duration, source_duration, edge_buffer_seconds)
        anchor = float(candidate.get("center", candidate.get("start", 0.0)))
        start_time = anchor - source_duration * 0.44
        start_time = max(allowed_lo, min(start_time, allowed_hi))

    return {
        "index": index,
        "video_file": candidate.get("video_file"),
        "source_name": candidate.get("source_name"),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": final_duration,
        "target": target,
        "score": _score_candidate(candidate, profile),
        "candidate_id": candidate.get("id"),
        "tags": list(candidate.get("tags", [])),
        "ai_analyzed": bool(candidate.get("ai_analyzed")),
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
    }
