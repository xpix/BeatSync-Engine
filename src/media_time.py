#!/usr/bin/env python3
"""Best-effort *real capture time* for a source photo or video.

Order of preference:
  1. Container / EXIF metadata written by the camera at capture time
     - videos:  QuickTime ``com.apple.quicktime.creationdate`` or ``creation_time``
       (read via ffprobe)
     - images:  EXIF ``DateTimeOriginal`` / ``DateTimeDigitized`` / ``DateTime``
       (read via Pillow), honouring ``OffsetTimeOriginal`` when present
  2. Filesystem modification time (``os.path.getmtime``) as a fallback when no
     capture metadata is available or it is obviously bogus.

``get_recording_timestamp`` returns a POSIX timestamp (float seconds) suitable
for sorting source clips chronologically.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from functools import lru_cache

# Anything before 1980-01-01 is treated as "no real date" (0 / epoch / garbage).
_MIN_PLAUSIBLE_TS = 315532800.0

_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".gif",
}
_VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".wmv", ".flv",
    ".mts", ".m2ts", ".ts", ".3gp", ".3g2", ".mpg", ".mpeg",
}

# ffprobe tag keys that carry a capture time, best first.
_VIDEO_TIME_KEYS = (
    "com.apple.quicktime.creationdate",
    "creation_time",
    "date",
    "date_recorded",
    "recorded_date",
)

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _ffprobe_path() -> str:
    try:
        from ffmpeg_processing import FFPROBE_PATH  # portable build, already resolved
        if FFPROBE_PATH:
            return FFPROBE_PATH
    except Exception:
        pass
    return shutil.which("ffprobe") or "ffprobe"


def _parse_datetime(value: str, fallback_tz: timezone | None = None) -> float | None:
    """Parse an EXIF/ISO datetime string to a POSIX timestamp, or None."""
    s = str(value or "").strip()
    if not s or s.startswith("0000"):
        return None

    # EXIF style: "2023:07:14 10:33:12" (naive, camera-local time)
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{1,2}):(\d{2}):(\d{2})", s)
    if m:
        try:
            dt = datetime(*(int(g) for g in m.groups()))
        except ValueError:
            return None
        if fallback_tz is not None:
            dt = dt.replace(tzinfo=fallback_tz)
        try:
            return dt.timestamp()
        except (OverflowError, OSError, ValueError):
            return None

    # ISO 8601, e.g. "2023-07-14T10:33:12.000000Z" or "...+0200" / "...+02:00"
    iso = s.replace("Z", "+00:00")
    iso = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", iso)  # +0200 -> +02:00
    iso = re.sub(r"(\.\d{6})\d+", r"\1", iso)            # trim over-long fractions
    for candidate in (iso, iso.split(".")[0]):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None and fallback_tz is not None:
            dt = dt.replace(tzinfo=fallback_tz)
        try:
            return dt.timestamp()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _exif_timestamp(path: str) -> float | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None

    sub: dict = {}
    try:
        sub = exif.get_ifd(0x8769)  # Exif sub-IFD (holds DateTimeOriginal)
    except Exception:
        sub = {}

    # Optional per-field UTC offset, e.g. "+02:00" (tags 0x9011 / 0x9010).
    tz = None
    for off_tag in (0x9011, 0x9010):
        off = sub.get(off_tag) if sub else None
        if off:
            mo = re.match(r"^\s*([+-])(\d{2}):?(\d{2})\s*$", str(off))
            if mo:
                sign = 1 if mo.group(1) == "+" else -1
                tz = timezone(sign * timedelta(hours=int(mo.group(2)), minutes=int(mo.group(3))))
            break

    for tag in (36867, 36868):        # DateTimeOriginal, DateTimeDigitized
        if sub and sub.get(tag):
            ts = _parse_datetime(sub[tag], fallback_tz=tz)
            if ts:
                return ts
    if exif.get(306):                 # DateTime (last resort, often "modified")
        return _parse_datetime(exif[306], fallback_tz=tz)
    return None


def _ffprobe_timestamp(path: str) -> float | None:
    cmd = [
        _ffprobe_path(), "-v", "error", "-of", "json",
        "-show_entries", "format_tags:stream_tags", path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        data = json.loads(proc.stdout)
    except (ValueError, TypeError):
        return None

    tag_maps = []
    fmt = data.get("format") or {}
    if isinstance(fmt.get("tags"), dict):
        tag_maps.append(fmt["tags"])
    for stream in data.get("streams") or []:
        if isinstance(stream, dict) and isinstance(stream.get("tags"), dict):
            tag_maps.append(stream["tags"])

    for key in _VIDEO_TIME_KEYS:
        for tags in tag_maps:
            lower = {str(k).lower(): v for k, v in tags.items()}
            if key in lower:
                ts = _parse_datetime(lower[key])
                if ts and ts >= _MIN_PLAUSIBLE_TS:
                    return ts
    return None


@lru_cache(maxsize=2048)
def _capture_timestamp_cached(path: str, _signature: tuple) -> float | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        order = (_exif_timestamp, _ffprobe_timestamp)
    else:
        order = (_ffprobe_timestamp, _exif_timestamp)
    for probe in order:
        try:
            ts = probe(path)
        except Exception:
            ts = None
        if ts and ts >= _MIN_PLAUSIBLE_TS:
            return ts
    return None


def get_capture_timestamp(path: str) -> float | None:
    """Real capture time as a POSIX timestamp, or None if no metadata is present."""
    try:
        st = os.stat(path)
        signature = (st.st_mtime_ns, st.st_size)
    except OSError:
        signature = (0, 0)
    return _capture_timestamp_cached(str(path), signature)


def get_recording_timestamp(path: str, fallback_to_mtime: bool = True) -> float:
    """Capture time for sorting: real metadata if available, else file mtime.

    Returns ``float('inf')`` when nothing at all can be read, so unreadable files
    sort last instead of first.
    """
    ts = get_capture_timestamp(path)
    if ts and ts >= _MIN_PLAUSIBLE_TS:
        return ts
    if fallback_to_mtime:
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("inf")
    return float("inf")
