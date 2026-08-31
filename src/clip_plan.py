#!/usr/bin/env python3
"""Persisted render plans.

A render plan stores everything needed to rebuild an output video without re-running
the audio/video analysis: the frame-locked timeline, the encoder settings and the
concrete source moment picked for every segment.

The output timeline is a pure function of (beat_times, audio_duration, fps) and is
completely independent of a clip's ``start_time`` -- that value only ends up as the
FFmpeg ``-ss`` seek. Shifting it therefore leaves the cut timeline frame-identical,
which is what makes manual clip nudging safe.
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ffmpeg_processing import get_video_duration

PLAN_VERSION = 1
PLAN_SUFFIX = '.plan.json'


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def plan_path_for_output(output_file: str) -> str:
    """Sidecar plan path that belongs to a rendered output video."""
    base, _ = os.path.splitext(os.path.abspath(output_file))
    return base + PLAN_SUFFIX


def build_render_plan(
    *,
    output_file: str,
    audio_file: str,
    video_files: Sequence[str],
    beat_times: Sequence[float],
    segment_durations: Sequence[float],
    clips: Sequence[Dict[str, Any]],
    fps: float,
    target_resolution: Tuple[int, int] | None,
    start_time: float = 0.0,
    end_time: float | None = None,
    processing_mode: str = 'cpu',
    lossless_mode: bool = False,
    use_gpu: bool = False,
    gpu_encoder: str = 'none',
    max_workers: int | None = None,
    strict_unique_non_overlap: bool = True,
    edge_buffer_seconds: float = 2.0,
    text_settings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble the serialisable plan dict for a finished render."""
    return {
        'version': PLAN_VERSION,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'output_file': os.path.abspath(output_file),
        'audio_file': os.path.abspath(audio_file),
        'video_files': [os.path.abspath(p) for p in video_files],
        'beat_times': [float(t) for t in beat_times],
        'segment_durations': [float(d) for d in segment_durations],
        'fps': float(fps),
        'target_resolution': list(target_resolution) if target_resolution else None,
        'start_time': float(start_time),
        'end_time': float(end_time) if end_time else None,
        'processing_mode': processing_mode,
        'lossless_mode': bool(lossless_mode),
        'use_gpu': bool(use_gpu),
        'gpu_encoder': gpu_encoder,
        'max_workers': max_workers,
        'strict_unique_non_overlap': bool(strict_unique_non_overlap),
        'edge_buffer_seconds': float(edge_buffer_seconds),
        'text_settings': dict(text_settings or {}),
        'clips': [dict(c) for c in clips],
    }


def save_render_plan(plan: Dict[str, Any], path: str) -> str:
    """Write the plan next to its output video."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def load_render_plan(path: str) -> Dict[str, Any]:
    """Read a plan file and reject anything this build cannot understand."""
    with open(path, 'r', encoding='utf-8') as handle:
        plan = json.load(handle)

    if not isinstance(plan, dict):
        raise ValueError('Plan file is not a JSON object.')
    if int(plan.get('version', 0)) != PLAN_VERSION:
        raise ValueError(
            f"Unsupported plan version {plan.get('version')} (expected {PLAN_VERSION})."
        )
    if not plan.get('clips'):
        raise ValueError('Plan file contains no clips.')
    if len(plan['clips']) != len(plan.get('segment_durations', [])):
        raise ValueError('Plan file is inconsistent: clip count differs from timeline length.')
    return plan


def list_render_plans(output_dir: str) -> List[str]:
    """All plan files in the output directory, newest first."""
    if not os.path.isdir(output_dir):
        return []
    found = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.endswith(PLAN_SUFFIX)
    ]
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found


def is_plan_inside(path: str, allowed_dir: str) -> bool:
    """Guard against reading plan files from outside the output directory."""
    try:
        resolved = os.path.realpath(path)
        root = os.path.realpath(allowed_dir)
        return os.path.commonpath([resolved, root]) == root
    except (ValueError, OSError):
        return False


def validate_plan_sources(plan: Dict[str, Any]) -> List[str]:
    """Report missing audio/source files before FFmpeg trips over them."""
    problems: List[str] = []
    audio_file = plan.get('audio_file')
    if not audio_file or not os.path.exists(audio_file):
        problems.append(f"Audio file missing: {audio_file}")

    missing_sources = sorted({
        clip.get('video_file', '')
        for clip in plan.get('clips', [])
        if not clip.get('video_file') or not os.path.exists(clip.get('video_file', ''))
    })
    for source in missing_sources:
        problems.append(f"Source video missing: {source or '<empty>'}")
    return problems


# ---------------------------------------------------------------------------
# Timecode helpers
# ---------------------------------------------------------------------------

def parse_timecode(text: str) -> float:
    """Parse ``83``, ``83.5``, ``1:23`` or ``0:01:23`` into seconds."""
    raw = str(text or '').strip().replace(',', '.')
    if not raw:
        raise ValueError('Please enter a timecode.')

    parts = raw.split(':')
    if len(parts) > 3:
        raise ValueError(f"Cannot read timecode '{text}'.")

    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"Cannot read timecode '{text}'.") from None

    if any(value < 0 for value in values):
        raise ValueError('Timecode cannot be negative.')

    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return seconds


def format_timecode(seconds: float) -> str:
    """Render seconds as ``m:ss.s`` (or ``h:mm:ss.s`` past the hour)."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600.0)
    minutes, secs = divmod(remainder, 60.0)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes)}:{secs:04.1f}"


def timeline_bounds(plan: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Start/end position of every clip inside the finished video."""
    bounds: List[Tuple[float, float]] = []
    cursor = 0.0
    for duration in plan.get('segment_durations', []):
        end = cursor + float(duration)
        bounds.append((cursor, end))
        cursor = end
    return bounds


def clip_at_time(plan: Dict[str, Any], seconds: float) -> int:
    """Index of the clip visible at ``seconds``.

    Ranges are half-open, so a timecode sitting exactly on a cut resolves to the
    following clip.
    """
    bounds = timeline_bounds(plan)
    if not bounds:
        raise ValueError('Plan contains no timeline.')

    total = bounds[-1][1]
    if seconds < 0 or seconds >= total:
        raise ValueError(
            f"Timecode {format_timecode(seconds)} is outside the video "
            f"(0:00.0 - {format_timecode(total)})."
        )

    for index, (start, end) in enumerate(bounds):
        if start <= seconds < end:
            return index
    return len(bounds) - 1


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------

def allowed_start_range(video_duration: float, source_duration: float) -> Tuple[float, float]:
    """Valid source in-points for a clip.

    Clamped to the real media boundaries only -- the planner's edge buffer is a
    selection heuristic and deliberately does not restrict manual nudging.
    """
    return 0.0, max(0.0, float(video_duration) - float(source_duration))


def _duration_cache(lookup: Callable[[str], float] | None) -> Callable[[str], float]:
    resolver = lookup or get_video_duration
    cache: Dict[str, float] = {}

    def cached(path: str) -> float:
        if path not in cache:
            cache[path] = float(resolver(path))
        return cache[path]

    return cached


def clip_offset_context(
    plan: Dict[str, Any],
    clip_index: int,
    duration_lookup: Callable[[str], float] | None = None,
) -> Dict[str, Any]:
    """Everything the editor needs to show and bound one clip."""
    clips = plan.get('clips', [])
    if not 0 <= clip_index < len(clips):
        raise ValueError(f'Clip index {clip_index} is out of range.')

    clip = clips[clip_index]
    durations = plan.get('segment_durations', [])
    segment_duration = float(durations[clip_index])
    source_duration = float(clip.get('source_duration', segment_duration))
    video_file = clip.get('video_file', '')
    video_duration = _duration_cache(duration_lookup)(video_file)
    start = float(clip.get('start_time', 0.0))
    lo, hi = allowed_start_range(video_duration, source_duration)
    timeline_start, timeline_end = timeline_bounds(plan)[clip_index]

    return {
        'index': clip_index,
        'video_file': video_file,
        'source_name': clip.get('source_name') or os.path.basename(video_file),
        'video_duration': video_duration,
        'source_duration': source_duration,
        'segment_duration': segment_duration,
        'start_time': start,
        'timeline_start': timeline_start,
        'timeline_end': timeline_end,
        'min_offset': lo - start,
        'max_offset': hi - start,
        'applied_offset': float(clip.get('manual_offset', 0.0)),
    }


def clamp_start(
    plan: Dict[str, Any],
    clip_index: int,
    offset: float,
    duration_lookup: Callable[[str], float] | None = None,
) -> Tuple[float, bool]:
    """Resulting source in-point for an offset, plus whether it had to be clamped."""
    context = clip_offset_context(plan, clip_index, duration_lookup)
    lo, hi = allowed_start_range(context['video_duration'], context['source_duration'])
    desired = context['start_time'] + float(offset)
    clamped = min(max(desired, lo), hi)
    return clamped, abs(clamped - desired) > 1e-6


def apply_offsets(
    plan: Dict[str, Any],
    offsets: Dict[int, float],
    duration_lookup: Callable[[str], float] | None = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return a clip sequence with the requested nudges baked in.

    Clip durations are never touched, so the frame-locked timeline is preserved.
    """
    resolve = _duration_cache(duration_lookup)
    durations = plan.get('segment_durations', [])
    sequence: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for index, original in enumerate(plan.get('clips', [])):
        clip = dict(original)
        offset = float(offsets.get(index, 0.0))
        if offset:
            segment_duration = float(durations[index])
            source_duration = float(clip.get('source_duration', segment_duration))
            video_duration = resolve(clip.get('video_file', ''))
            lo, hi = allowed_start_range(video_duration, source_duration)
            desired = float(clip.get('start_time', 0.0)) + offset
            clamped = min(max(desired, lo), hi)
            if abs(clamped - desired) > 1e-6:
                warnings.append(
                    f"Clip {index + 1}: offset {offset:+.2f}s clamped to the source bounds "
                    f"({format_timecode(lo)} - {format_timecode(hi)})."
                )
            clip['manual_offset'] = float(clip.get('manual_offset', 0.0)) + (clamped - float(clip.get('start_time', 0.0)))
            clip['start_time'] = clamped
        sequence.append(clip)

    return sequence, warnings
