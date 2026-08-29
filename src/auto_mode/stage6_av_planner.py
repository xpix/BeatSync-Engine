#!/usr/bin/env python3
"""Audio-visual clip planner for Auto Mode."""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter, deque
from typing import Dict, List, Sequence

import numpy as np

try:
    # src/ is on sys.path once the auto_mode package is imported (see __init__.py).
    from media_time import get_recording_timestamp
except Exception:  # pragma: no cover - fallback if imported in isolation
    def get_recording_timestamp(path: str, fallback_to_mtime: bool = True) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("inf")

CLIP_ORDER_MODES = ("auto", "chronological", "name")


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


def _sorted_unique_videos(candidates: Sequence[Dict], clip_order_mode: str) -> List[str]:
    """Distinct source videos in the fixed order clips should be drawn from, or [] for scored/auto selection."""
    if clip_order_mode not in ("chronological", "name"):
        return []
    videos: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        video_file = str(candidate.get("video_file") or "")
        if video_file and video_file not in seen:
            seen.add(video_file)
            videos.append(video_file)
    if clip_order_mode == "name":
        videos.sort(key=lambda v: os.path.basename(v).lower())
    else:
        # Real capture time (EXIF / container metadata); file mtime only as fallback.
        videos.sort(key=lambda v: (get_recording_timestamp(v), os.path.basename(v).lower()))
    return videos


def _forced_segment_videos(segment_count: int, first_video: str | None, last_video: str | None) -> Dict[int, str]:
    """Map segment index -> source video that must be used there (start/end pinning)."""
    forced: Dict[int, str] = {}
    if segment_count <= 0:
        return forced
    if first_video:
        forced[0] = str(first_video)
    if last_video:
        forced[segment_count - 1] = str(last_video)  # last_video wins when only one segment exists
    return forced


def _buffered_start_bounds(video_duration: float, source_duration: float, edge_buffer_seconds: float) -> tuple[float, float]:
    """Valid [lo, hi] start-time range that keeps clips out of the video's edge buffer.

    Shrinks gracefully (rather than rejecting the video) when it is too short to fit the buffer.
    """
    max_start_full = max(0.0, video_duration - source_duration)
    buffer = max(0.0, edge_buffer_seconds)
    lo = min(buffer, max_start_full)
    hi = max(lo, max_start_full - buffer)
    return lo, hi


def _is_image_loop_video(video_file: str) -> bool:
    """True for the synthetic per-image loop videos created by prepare_visual_sources()."""
    return os.path.basename(str(video_file)).startswith("image_source_")


def _effective_edge_buffer(video_file: str, edge_buffer_seconds: float) -> float:
    """A static-photo loop has no unstable start/end, so it never needs the buffer."""
    return 0.0 if _is_image_loop_video(video_file) else edge_buffer_seconds


def _forced_pin_clip(
    forced_video: str,
    profile: Dict,
    index: int,
    candidates: Sequence[Dict],
    occupied_ranges: Dict[str, List[tuple[float, float]]],
    edge_buffer_seconds: float,
    anchor: str,
) -> Dict | None:
    """Pin `forced_video` (Start-/End-Video) to this segment using the source's full
    duration rather than a single detected scene window.

    The normal candidate path only succeeds when a Qwen/heuristic scene detected
    inside the pinned video happens to be long enough -- and positioned correctly --
    to cover the segment's exact cut duration. That is frequently not the case for
    the very first/last cut, which silently drops the pin. This instead treats the
    whole (edge-buffered) source video as the usable window, so the pin is honored
    whenever the source is simply long enough: anchored at its very start for the
    first segment, or its very end for the last segment.
    """
    video_duration = None
    for c in candidates:
        if str(c.get("video_file") or "") == forced_video:
            vd = c.get("video_duration")
            if vd:
                video_duration = float(vd)
                break
    if not video_duration or video_duration <= 0:
        return None

    source_duration = max(0.05, float(profile.get("duration", 0.05)))
    effective_buffer = _effective_edge_buffer(forced_video, edge_buffer_seconds)
    allowed_lo, allowed_hi = _buffered_start_bounds(video_duration, source_duration, effective_buffer)
    region_end = allowed_hi + source_duration
    if region_end - allowed_lo < source_duration:
        return None

    preferred_start = allowed_lo if anchor == "start" else allowed_hi
    occupied = occupied_ranges.get(forced_video, [])
    start_time = _pick_start_from_available_gaps(allowed_lo, region_end, source_duration, occupied, preferred_start)
    if start_time is None:
        return None

    candidate_id = f"forced_{anchor}_{index:05d}_{int(start_time * 1000):08d}"
    return {
        "index": index,
        "video_file": forced_video,
        "source_name": os.path.basename(forced_video),
        "start_time": start_time,
        "source_duration": source_duration,
        "final_duration": source_duration,
        "target": profile.get("target", "flow"),
        "score": 0.0,
        "candidate_id": candidate_id,
        "tags": ["forced_pin"],
        "ai_analyzed": False,
        "audio_start": profile.get("start"),
        "audio_end": profile.get("end"),
        "wave": profile.get("wave"),
        "impact": profile.get("impact"),
    }


def build_planned_clip_sequence(
    cut_times: Sequence[float],
    segment_durations: Sequence[float],
    beat_info: Dict | None,
    video_files: Sequence[str],
    strict_unique_non_overlap: bool = True,
    preferred_videos: Sequence[str] = (),
    edge_buffer_seconds: float = 2.0,
    clip_order_mode: str = "auto",
    first_video: str | None = None,
    last_video: str | None = None,
    debug_callback=None,
) -> List[Dict]:
    """Build exact source clip choices for every output segment.

    Returns an empty list when no visual library is present, which tells the
    renderer to keep its old fallback sampling.

    edge_buffer_seconds: portion at the very start/end of each source video that
    is never used for a clip (e.g. to skip intros/outros or unstable footage).
    clip_order_mode: "auto" (editorial scoring, default), "chronological"
    (sources ordered by file-modified-time), or "name" (ordered by alphabetical
    filename). In the non-auto modes, one clip is drawn from each source in that
    order, then it wraps back to the first and repeats, so every source is used
    before any source is used a second time.
    first_video/last_video: when given, pin the very first/last output segment
    to a clip from that source video (falls back to normal selection if that
    video has no usable candidate for the segment).
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
    planned_by_index: Dict[int, Dict] = {}
    video_order = _sorted_unique_videos(candidates, clip_order_mode)
    active_video_idx = [0]
    exhausted_ordered_videos: set[str] = set()
    forced_segment_videos = _forced_segment_videos(len(profiles), first_video, last_video)
    failed_segment_indices: List[int] = []
    used_image_videos: set[str] = set()

    for i, profile in enumerate(profiles):
        planned_clip = None
        # A still-photo loop looks identical at every offset, so once one has been
        # used its remaining candidates are dropped from the pool entirely.
        available_candidates = (
            [c for c in candidates if str(c.get("video_file") or "") not in used_image_videos]
            if used_image_videos else candidates
        )
        forced_video = forced_segment_videos.get(i)
        if forced_video:
            pool = [c for c in available_candidates if str(c.get("video_file") or "") == forced_video]
            planned_clip = _choose_for_segment(
                candidates=pool,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                strict_unique_non_overlap=strict_unique_non_overlap,
                used_candidate_ids=used_candidate_ids,
                occupied_ranges=occupied_ranges,
                preferred_videos=preferred_set,
                edge_buffer_seconds=edge_buffer_seconds,
            ) if pool else None
            if planned_clip is None:
                # No single detected scene in the pinned source was long enough (or
                # positioned right) to cover this segment. Pin against the source's
                # full duration instead of a specific scene window, so Start-/End-
                # Video is still honored rather than silently falling through to
                # normal (unpinned) selection.
                pin_anchor = "start" if i == 0 else "end"
                planned_clip = _forced_pin_clip(
                    forced_video=forced_video,
                    profile=profile,
                    index=i,
                    candidates=candidates,
                    occupied_ranges=occupied_ranges,
                    edge_buffer_seconds=edge_buffer_seconds,
                    anchor=pin_anchor,
                )
                if planned_clip is None:
                    pin_warning = (
                        f"Start/End-Video pin could not be honored for segment {i} "
                        f"({os.path.basename(forced_video)}): source too short for this "
                        f"cut after the edge buffer, or not part of the analyzed video "
                        f"library. Falling back to normal selection for this segment."
                    )
                    print(f"   \u26a0\ufe0f  {pin_warning}", flush=True)
                    if debug_callback:
                        debug_callback(f"Warning: {pin_warning}")
        if planned_clip is None and video_order:
            planned_clip = _select_ordered_segment(
                candidates=available_candidates,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                strict_unique_non_overlap=strict_unique_non_overlap,
                used_candidate_ids=used_candidate_ids,
                occupied_ranges=occupied_ranges,
                preferred_videos=preferred_set,
                edge_buffer_seconds=edge_buffer_seconds,
                video_order=video_order,
                active_idx=active_video_idx,
                exhausted_videos=exhausted_ordered_videos,
            )
        if planned_clip is None:
            planned_clip = _choose_for_segment(
                candidates=available_candidates,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                strict_unique_non_overlap=strict_unique_non_overlap,
                used_candidate_ids=used_candidate_ids,
                occupied_ranges=occupied_ranges,
                preferred_videos=preferred_set,
                edge_buffer_seconds=edge_buffer_seconds,
            )

        if not planned_clip:
            failed_segment_indices.append(i)
            continue
        candidate_id = str(planned_clip.get("candidate_id") or "")
        video_file = str(planned_clip.get("video_file") or "")
        start_time = float(planned_clip.get("start_time") or 0.0)
        source_duration = max(0.05, float(planned_clip.get("source_duration") or 0.05))
        end_time = start_time + source_duration

        planned_by_index[i] = planned_clip
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
            if _is_image_loop_video(video_file):
                used_image_videos.add(video_file)

    if failed_segment_indices:
        preview = failed_segment_indices[:10]
        more = f" (+{len(failed_segment_indices) - 10} more)" if len(failed_segment_indices) > 10 else ""
        message = (
            f"AV planner: {len(failed_segment_indices)}/{len(durations_arr)} segment(s) had no "
            f"non-overlapping candidate (indices {preview}{more}, mode={clip_order_mode}, "
            f"candidate pool={len(candidates)}, sources={len({c.get('video_file') for c in candidates})}). "
            f"Repairing with best-scoring overlap-allowed picks so the rest of the AI-scored plan is kept."
        )
        print(f"   ⚠️  {message}", flush=True)
        if debug_callback:
            debug_callback(f"Warning: {message}")

        for i in failed_segment_indices:
            profile = profiles[i]
            repair_pool = [
                c for c in candidates if str(c.get("video_file") or "") not in used_image_videos
            ] or candidates
            candidate = _choose_relaxed_candidate(
                candidates=repair_pool,
                profile=profile,
                recent_ids=recent_ids,
                recent_videos=recent_videos,
                usage=usage,
                index=i,
                preferred_videos=preferred_set,
            )
            if not candidate:
                continue
            repaired_clip = _materialize_clip(
                candidate=candidate,
                profile=profile,
                index=i,
                edge_buffer_seconds=edge_buffer_seconds,
            )
            planned_by_index[i] = repaired_clip
            candidate_id = str(repaired_clip.get("candidate_id") or "")
            video_file = str(repaired_clip.get("video_file") or "")
            if candidate_id:
                recent_ids.append(candidate_id)
                usage[candidate_id] += 1
            if video_file:
                recent_videos.append(video_file)
                usage[video_file] += 1
                if _is_image_loop_video(video_file):
                    used_image_videos.add(video_file)

    planned = [planned_by_index[i] for i in range(len(profiles)) if i in planned_by_index]
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


def _choose_for_segment(
    candidates: Sequence[Dict],
    profile: Dict,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
    strict_unique_non_overlap: bool,
    used_candidate_ids: set[str],
    occupied_ranges: Dict[str, List[tuple[float, float]]],
    preferred_videos: set[str],
    edge_buffer_seconds: float,
) -> Dict | None:
    if strict_unique_non_overlap:
        return _choose_and_materialize_candidate(
            candidates=candidates,
            profile=profile,
            recent_ids=recent_ids,
            recent_videos=recent_videos,
            usage=usage,
            index=index,
            used_candidate_ids=used_candidate_ids,
            occupied_ranges=occupied_ranges,
            preferred_videos=preferred_videos,
            edge_buffer_seconds=edge_buffer_seconds,
        )
    candidate = _choose_relaxed_candidate(
        candidates=candidates,
        profile=profile,
        recent_ids=recent_ids,
        recent_videos=recent_videos,
        usage=usage,
        index=index,
        preferred_videos=preferred_videos,
    )
    return (
        _materialize_clip(candidate, profile, index, edge_buffer_seconds=edge_buffer_seconds)
        if candidate else None
    )


def _select_ordered_segment(
    candidates: Sequence[Dict],
    profile: Dict,
    recent_ids: deque,
    recent_videos: deque,
    usage: Counter,
    index: int,
    strict_unique_non_overlap: bool,
    used_candidate_ids: set[str],
    occupied_ranges: Dict[str, List[tuple[float, float]]],
    preferred_videos: set[str],
    edge_buffer_seconds: float,
    video_order: List[str],
    active_idx: List[int],
    exhausted_videos: set[str],
) -> Dict | None:
    """Round-robin across the source videos in the fixed order: one clip per source
    per lap, then wrap back to the first. Sources with no usable candidate left are
    marked exhausted and skipped on later laps."""
    order_len = len(video_order)
    for _ in range(order_len):
        if active_idx[0] >= order_len:
            active_idx[0] = 0
        current_video = video_order[active_idx[0]]
        if current_video in exhausted_videos:
            active_idx[0] += 1
            continue
        pool = [c for c in candidates if str(c.get("video_file") or "") == current_video]
        clip = _choose_for_segment(
            candidates=pool,
            profile=profile,
            recent_ids=recent_ids,
            recent_videos=recent_videos,
            usage=usage,
            index=index,
            strict_unique_non_overlap=strict_unique_non_overlap,
            used_candidate_ids=used_candidate_ids,
            occupied_ranges=occupied_ranges,
            preferred_videos=preferred_videos,
            edge_buffer_seconds=edge_buffer_seconds,
        ) if pool else None
        active_idx[0] += 1
        if clip:
            return clip
        exhausted_videos.add(current_video)
    return None


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
    effective_buffer = _effective_edge_buffer(str(candidate.get("video_file") or ""), edge_buffer_seconds)
    allowed_lo, allowed_hi = _buffered_start_bounds(video_duration, source_duration, effective_buffer)
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
        effective_buffer = _effective_edge_buffer(str(candidate.get("video_file") or ""), edge_buffer_seconds)
        allowed_lo, allowed_hi = _buffered_start_bounds(video_duration, source_duration, effective_buffer)
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
