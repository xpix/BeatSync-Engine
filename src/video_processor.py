import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from logger import setup_environment

# Initialize environment
setup_environment()

# NOW import other modules (after CUDA and Python environment is set)
import argparse
import random
import shutil
from typing import TypeAlias, List, Dict, Tuple, Sequence
import numpy as np
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import warnings
import time

from gpu_cpu_utils import (
    PARALLEL_WORKERS,
    GPU_INFO as gpu_info,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    set_gpu_mode,
)
from paths import (
    get_processing_dir,
)

# Import FFmpeg processing module
from ffmpeg_processing import (
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    convert_to_prores_proxy,
    extract_clip_segment_ffmpeg,
    extract_prores_segment_random,
    concatenate_videos_ffmpeg,
    seconds_to_frame_count,
    frame_count_to_seconds,
)
from auto_mode.stage6_av_planner import build_planned_clip_sequence, summarize_clip_plan

# Import mode modules
from auto_mode import analyze_beats_auto

warnings.filterwarnings('ignore', message='.*bytes wanted but 0 bytes read.*')

BeatTimes : TypeAlias = np.ndarray
VideoList : TypeAlias = List[str]


def _fmt_seconds(seconds: float) -> str:
    try:
        value = float(seconds)
    except Exception:
        value = 0.0
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def _env_int(name: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _effective_clip_workers(requested_workers: int, use_nvenc: bool) -> int:
    """Choose a stable FFmpeg worker count for clip extraction.

    Multiple simultaneous NVENC encodes can fight for one hardware encoder and
    decode/IO bandwidth. Capping the default is a performance-only change: the
    same clip plan, source times, frame counts, and encoder settings are used.
    """
    requested_workers = max(1, int(requested_workers or 1))
    if not use_nvenc:
        return requested_workers

    cpu = os.cpu_count() or 4
    default_nvenc_workers = min(requested_workers, 4, max(1, cpu // 2))
    return _env_int(
        "BEATSYNC_NVENC_CLIP_WORKERS",
        default_nvenc_workers,
        lo=1,
        hi=requested_workers,
    )


def _summarize_clip_timings(timings: List[float], total_duration: float) -> None:
    if not timings:
        return
    arr = np.asarray(timings, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    print(
        f"   ⏱ Clip extraction total: {_fmt_seconds(total_duration)} | "
        f"avg {float(np.mean(arr)):.2f}s, p50 {float(np.percentile(arr, 50)):.2f}s, "
        f"p90 {float(np.percentile(arr, 90)):.2f}s, max {float(np.max(arr)):.2f}s"
    )

# Log status on import if running directly
if __name__ == '__main__':
    if GPU_AVAILABLE:
        print(f"⚡ GPU available: {gpu_info['name']}")
    else:
        print(f"💻 GPU Acceleration: NOT AVAILABLE")


if __name__ == '__main__':
    if NVENC_AVAILABLE:
        print(f"🎬 NVIDIA NVENC: AVAILABLE - Hardware video encoding enabled")
    else:
        print(f"⚠️  NVIDIA NVENC: NOT AVAILABLE - Using CPU encoding only")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create an Auto Mode music video with rhythmic audio-video cuts'
    )
    parser.add_argument(
        'mp3_file',
        type=str,
        help='Path to the input audio file (MP3/WAV/FLAC)'
    )
    parser.add_argument(
        'video_directory',
        type=str,
        help='Directory containing MP4/MKV video files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output_music_video.mkv',
        help='Output video file path (default: output_music_video.mkv)'
    )
    parser.add_argument(
        '-s', '--start-time',
        type=float,
        default=0.0,
        help='Start time in seconds for audio processing (default: 0.0)'
    )
    parser.add_argument(
        '-e', '--end-time',
        type=float,
        default=None,
        help='End time in seconds for audio processing (default: full duration)'
    )
    parser.add_argument(
        '--lossless',
        action='store_true',
        help='Enable Lossless/Precise mode with ProRes 422 Proxy (frame-accurate cuts, no re-encoding)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration (audio analysis + NVENC encoding)'
    )
    parser.add_argument(
        '--gpu-encoder',
        type=str,
        choices=['h264_nvenc', 'hevc_nvenc', 'none'],
        default='h264_nvenc',
        help='GPU encoder: h264_nvenc (H.264), hevc_nvenc (H.265), none (CPU) (default: h264_nvenc)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output FPS (frames per second). If not specified, auto-detect from input video (default: auto)'
    )
    parser.add_argument(
        '--edge-buffer-seconds',
        type=float,
        default=5.0,
        help='Seconds ignored at the start/end of each source video when picking clips (default: 5.0)'
    )
    parser.add_argument(
        '--clip-order-mode',
        type=str,
        choices=['auto', 'chronological', 'name'],
        default='auto',
        help='Order source videos appear in the output: auto (AI editorial selection), '
             'chronological (by file modified time), or name (alphabetical filename) (default: auto)'
    )

    return parser.parse_args()


def get_video_files(directory : str) -> VideoList:
    video_extensions = ['.mp4', '.MP4', '.mkv', '.MKV']
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))

    if not video_files:
        raise ValueError(f'No MP4/MKV files found in {directory}')

    return [str(f) for f in video_files]


def _sort_video_files(video_files: Sequence[str], clip_order_mode: str) -> List[str]:
    """Reorder source videos for deterministic clip sequencing ("chronological"/"name"), unchanged for "auto"."""
    files = [str(f) for f in video_files]
    if clip_order_mode == "name":
        files.sort(key=lambda f: os.path.basename(f).lower())
    elif clip_order_mode == "chronological":
        def _mtime(f: str) -> float:
            try:
                return os.path.getmtime(f)
            except OSError:
                return float("inf")
        files.sort(key=_mtime)
    return files


def build_frame_aligned_cut_timeline(beat_times: BeatTimes, audio_duration: float,
                                     fps: float):
    """
    Build the output cut timeline on absolute video frame numbers.

    This is the critical sync stage: each cut boundary is quantized once from its
    absolute beat time. Segment durations are then frame differences between
    absolute boundaries, so rounding error cannot accumulate from clip to clip.

    The first and last boundaries stay locked to the audio timeline.
    """
    if fps <= 0:
        raise ValueError(f"Invalid FPS for cut timeline: {fps}")
    if audio_duration <= 0:
        raise ValueError(f"Invalid audio duration: {audio_duration}")

    beats = np.asarray(beat_times, dtype=float).reshape(-1)
    beats = beats[np.isfinite(beats)]
    if beats.size == 0:
        raise ValueError("No valid beat times were provided.")

    beats = np.sort(beats)

    internal_beats = beats[(beats > 0.0) & (beats < audio_duration)]

    raw_cut_times = np.concatenate(([0.0], internal_beats, [audio_duration]))

    # Quantize ABSOLUTE cut positions, not per-segment durations. This removes
    # cumulative drift caused by round((beat[i+1] - beat[i]) * fps) on each clip.
    end_frame = max(1, seconds_to_frame_count(audio_duration, fps))
    cut_frames = np.rint(raw_cut_times * fps).astype(int)
    cut_frames = np.clip(cut_frames, 0, end_frame)
    cut_frames[0] = 0
    cut_frames[-1] = end_frame
    cut_frames = np.unique(cut_frames)

    if cut_frames.size == 0 or cut_frames[0] != 0:
        cut_frames = np.insert(cut_frames, 0, 0)
    if cut_frames[-1] != end_frame:
        cut_frames = np.append(cut_frames, end_frame)

    if cut_frames.size < 2:
        cut_frames = np.array([0, end_frame], dtype=int)

    segment_frames = np.diff(cut_frames).astype(int)
    segment_durations = segment_frames / fps
    cut_times = cut_frames / fps
    dropped_boundaries = max(0, raw_cut_times.size - cut_frames.size)

    return cut_times, segment_frames, segment_durations, dropped_boundaries


def create_clip_parallel(args):
    """
    Wrapper function for parallel clip creation using FFmpeg.

    Auto Mode usually passes a planned source moment. If visual planning is not
    available, this worker samples forward source content as a fallback.
    """
    clip_started = time.perf_counter()
    planned_clip = None
    if len(args) >= 9:
        (i, video_file, final_duration, target_size,
         use_nvenc, gpu_encoder, temp_dir, fps, planned_clip) = args
    else:
        (i, video_file, final_duration, target_size,
         use_nvenc, gpu_encoder, temp_dir, fps) = args
    
    try:
        if planned_clip:
            video_file = planned_clip.get('video_file') or video_file
            video_duration = get_video_duration(video_file)
            source_duration = float(planned_clip.get('source_duration', final_duration))
            source_duration = max(0.05, min(source_duration, video_duration))
            max_start = max(0.0, video_duration - source_duration)
            clip_start = max(0.0, min(float(planned_clip.get('start_time', 0.0)), max_start))
        else:
            # Random start time from video if visual planning is unavailable.
            video_duration = get_video_duration(video_file)
            
            required_source_duration = final_duration
            
            if video_duration >= required_source_duration:
                max_start = video_duration - required_source_duration
                clip_start = random.uniform(0, max_start)
                source_duration = required_source_duration
            else:
                clip_start = 0
                source_duration = video_duration
            
        
        # Output file
        temp_clip_path = os.path.join(temp_dir, f"temp_clip_{i}_{uuid.uuid4().hex}.mp4")
        
        extract_kwargs = {
            'video_file': video_file,
            'start_time': clip_start,
            'duration': source_duration,
            'output_file': temp_clip_path,
            'fps': fps,
            'target_size': target_size,
            'use_nvenc': use_nvenc,
            'gpu_encoder': gpu_encoder,
        }

        success = extract_clip_segment_ffmpeg(**extract_kwargs)
        
        elapsed = time.perf_counter() - clip_started
        if not success:
            return (i, None, target_size, None, "FFmpeg extraction failed", elapsed)
        
        return (i, temp_clip_path, target_size, temp_clip_path, None, elapsed)
        
    except Exception as e:
        elapsed = time.perf_counter() - clip_started
        return (i, None, target_size, None, str(e), elapsed)


def _build_non_overlapping_fallback_sequence(
    segment_durations: Sequence[float],
    video_files: Sequence[str],
    preferred_videos: Sequence[str] = (),
    edge_buffer_seconds: float = 5.0,
    clip_order_mode: str = "auto",
) -> List[Dict]:
    """Create a deterministic no-overlap plan when AV candidates are unavailable.

    By default (clip_order_mode="auto") the planner places segments sequentially
    within each source file and rotates across files, so selected source ranges
    are never reused or overlapped. Preferred sources appear more often in the
    rotation so more clips are drawn from them.

    When clip_order_mode is "chronological" or "name", video_files is sorted
    accordingly and each source video is filled completely before moving on to
    the next one, so clips appear in that fixed source-video order in the output.
    """
    edge_buffer_seconds = max(0.0, float(edge_buffer_seconds))
    preferred_set = {str(p) for p in preferred_videos if p}
    sequential_fill = clip_order_mode in ("chronological", "name")
    if sequential_fill:
        video_files = _sort_video_files(video_files, clip_order_mode)
    sources: List[Dict[str, float | str]] = []
    for video_file in video_files:
        try:
            duration = float(get_video_duration(video_file))
        except Exception:
            duration = 0.0
        if duration > 0.06:
            entry = {"video_file": video_file, "video_duration": duration}
            sources.append(entry)
            # Repeat preferred sources in the rotation order to bias more clips toward them.
            if not sequential_fill and video_file in preferred_set:
                sources.extend({"video_file": video_file, "video_duration": duration} for _ in range(3))

    if not sources:
        return []

    cursors: Dict[str, float] = {}
    for src in sources:
        video_file = str(src["video_file"])
        if video_file not in cursors:
            cursors[video_file] = min(edge_buffer_seconds, float(src["video_duration"]))
    planned: List[Dict] = []
    next_source_index = 0

    for i, raw_duration in enumerate(segment_durations):
        required = max(0.05, float(raw_duration))
        chosen = None
        source_count = len(sources)
        if sequential_fill:
            candidate_indices = list(range(next_source_index, source_count))
        else:
            candidate_indices = [(next_source_index + step) % source_count for step in range(source_count)]

        for source_idx in candidate_indices:
            source = sources[source_idx]
            video_file = str(source["video_file"])
            video_duration = float(source["video_duration"])
            max_start_full = max(0.0, video_duration - required)
            max_start = max(min(edge_buffer_seconds, max_start_full), max_start_full - edge_buffer_seconds)
            cursor = max(0.0, float(cursors.get(video_file, 0.0)))
            if cursor <= max_start + 1e-9:
                start_time = min(cursor, max_start)
                end_time = start_time + required
                cursors[video_file] = end_time
                next_source_index = source_idx if sequential_fill else (source_idx + 1) % source_count
                chosen = {
                    "index": i,
                    "video_file": video_file,
                    "source_name": os.path.basename(video_file),
                    "start_time": start_time,
                    "source_duration": required,
                    "final_duration": required,
                    "target": "flow",
                    "score": 0.0,
                    "candidate_id": f"fallback_{i:05d}_{source_idx:03d}",
                    "tags": ["flow", "fallback"],
                    "ai_analyzed": False,
                }
                break

        if chosen is None:
            return []

        planned.append(chosen)

    return planned


def _build_legacy_random_fallback_sequence(
    segment_durations: Sequence[float],
    video_files: Sequence[str],
    preferred_videos: Sequence[str] = (),
    edge_buffer_seconds: float = 5.0,
    clip_order_mode: str = "auto",
) -> List[Dict]:
    """Legacy fallback: allow source reuse/overlap when strict mode is disabled.

    When clip_order_mode is "chronological" or "name", segments are assigned in
    consecutive blocks per sorted source video (wrapping/reusing footage within
    a video once it runs out) instead of a random weighted pick per segment.
    """
    if not video_files:
        return []

    edge_buffer_seconds = max(0.0, float(edge_buffer_seconds))
    preferred_set = {str(p) for p in preferred_videos if p}
    sequential = clip_order_mode in ("chronological", "name")
    ordered_files = _sort_video_files(video_files, clip_order_mode) if sequential else list(video_files)
    weights = [4.0 if str(f) in preferred_set else 1.0 for f in ordered_files]

    durations: Dict[str, float] = {}
    for video_file in ordered_files:
        try:
            durations[str(video_file)] = float(get_video_duration(video_file))
        except Exception:
            durations[str(video_file)] = 0.0

    if sequential:
        segment_count = len(segment_durations)
        file_count = len(ordered_files)
        base_share, remainder = divmod(segment_count, file_count)
        assignment: List[str] = []
        for f_i, video_file in enumerate(ordered_files):
            share = base_share + (1 if f_i < remainder else 0)
            assignment.extend([video_file] * share)
        cursors: Dict[str, float] = {}

    planned: List[Dict] = []
    for i, raw_duration in enumerate(segment_durations):
        duration = max(0.05, float(raw_duration))
        if sequential:
            chosen_file = assignment[i]
            video_duration = durations.get(str(chosen_file), 0.0)
            max_start_full = max(0.0, video_duration - duration)
            cursor = cursors.get(str(chosen_file), edge_buffer_seconds)
            if max_start_full <= 0.0:
                start_time = 0.0
            elif cursor > max_start_full + 1e-9:
                start_time = min(edge_buffer_seconds, max_start_full)  # wrap: reuse this video from the start
            else:
                start_time = min(cursor, max_start_full)
            cursors[str(chosen_file)] = start_time + duration
        else:
            chosen_file = random.choices(list(ordered_files), weights=weights, k=1)[0]
            video_duration = durations.get(str(chosen_file), 0.0)
            max_start_full = max(0.0, video_duration - duration)
            start_time = min(max(edge_buffer_seconds, 0.0), max_start_full)
        planned.append({
            "index": i,
            "video_file": chosen_file,
            "source_name": os.path.basename(chosen_file),
            "start_time": start_time,
            "source_duration": duration,
            "final_duration": duration,
            "target": "flow",
            "score": 0.0,
            "candidate_id": f"legacy_{i:05d}",
            "tags": ["flow", "legacy_fallback"],
            "ai_analyzed": False,
        })
    return planned


def create_music_video(audio_file: str, video_files: VideoList, beat_times: BeatTimes,
                      output_file: str = 'output_music_video.mkv',
                      start_time: float = 0.0, end_time: float = None,
                      max_workers: int = None,
                      beat_info: dict = None,
                      lossless_mode: bool = False, use_gpu: bool = False, 
                      gpu_encoder: str = 'h264_nvenc', fps: float = None,
                      strict_unique_non_overlap: bool = True,
                      preferred_videos: Sequence[str] = (),
                      edge_buffer_seconds: float = 5.0,
                      clip_order_mode: str = "auto") -> str:
    """
    Creates a music video with video clips cut to detected beats.
    
    **PURE FFMPEG IMPLEMENTATION - FRAME-ACCURATE**
    
    ✅ NO BATCH PROCESSING: FFmpeg handles memory independently
    ✅ FRAME-ACCURATE: Uses exact frame counts for zero drift
    ✅ NO CUMULATIVE ERROR: Each segment is precisely timed
    
    Args:
        audio_file: Path to audio file
        video_files: List of video file paths
        beat_times: Array of beat times (already processed by mode)
        output_file: Output file path
        start_time: Audio start time
        end_time: Audio end time
        max_workers: Number of parallel workers
        beat_info: Beat information dictionary
        lossless_mode: Use ProRes 422 Proxy mode
        use_gpu: Use GPU acceleration
        gpu_encoder: GPU encoder to use
        fps: Output FPS
        edge_buffer_seconds: Seconds ignored at the start/end of each source video
        clip_order_mode: "auto" (editorial AI selection, default), "chronological"
            (source videos used in file-modified-time order), or "name" (used in
            alphabetical filename order)
    
    Returns:
        Path to output video file
    """
    if len(beat_times) == 0:
        raise ValueError("No beats were detected. Cannot create video.")

    video_creation_started = time.perf_counter()
    edge_buffer_seconds = max(0.0, float(edge_buffer_seconds))

    if max_workers is None:
        max_workers = PARALLEL_WORKERS

    # Determine FPS to use
    if fps is None:
        # Auto-detect from first video file
        try:
            fps = get_video_fps(video_files[0])
            print(f"🎞️ Auto-detected FPS from input video: {fps}")
        except Exception as e:
            fps = 30.0
            print(f"⚠️ Could not detect FPS, using default: {fps}")
    else:
        print(f"🎞️ Using custom FPS: {fps}")

    # Use fixed processing directory
    session_temp_dir = get_processing_dir()
    
    # 🧹 CLEANUP: Clear processing directory for a fresh start
    try:
        if os.path.exists(session_temp_dir):
            for item in os.listdir(session_temp_dir):
                item_path = os.path.join(session_temp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
        os.makedirs(session_temp_dir, exist_ok=True)
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clear processing directory: {e}")
        
    print(f"📁 Processing directory: {session_temp_dir}")

    # Determine processing mode
    use_nvenc = use_gpu and NVENC_AVAILABLE and not lossless_mode and gpu_encoder != 'none'
    requested_workers = max_workers
    max_workers = _effective_clip_workers(max_workers, use_nvenc)
    render_info = beat_info.setdefault("render_info", {}) if isinstance(beat_info, dict) else {}
    
    gpu_status = "⚡ GPU" if use_gpu and GPU_AVAILABLE else "💻 CPU"
    encoder_status = f"⚡ {gpu_encoder.upper()}" if use_nvenc else ("🎯 Frame-Perfect ProRes" if lossless_mode else "💻 CPU (libx264)")
    render_info.update({
        "clip_workers": int(max_workers),
        "requested_workers": int(requested_workers),
        "encoder": gpu_encoder.upper() if use_nvenc else ("PRORES_PROXY" if lossless_mode else "H264_CPU"),
        "output_fps": float(fps),
    })
    # Determine mode name
    mode_name = beat_info.get('mode', 'unknown') if beat_info else 'unknown'
    
    print(f"🎬 Processing Settings:")
    if max_workers != requested_workers:
        print(f"   • Parallel clip workers: {max_workers} (requested {requested_workers}; NVENC contention cap)")
    else:
        print(f"   • Parallel clip workers: {max_workers}")
    print(f"   • Video processing: {encoder_status}")
    print(f"   • Output FPS: {fps}")
    print(f"   • Frame-accurate mode: ENABLED (zero drift)")
    print(f"   • Generation mode: {mode_name}")
    if lossless_mode:
        print(f"   • Lossless Mode: ENABLED (ProRes 422 Proxy)")
        print(f"   • Precision Mode: Frame-perfect (re-encodes all segments)")
        print(f"   • Export Format: Apple ProRes 422 Proxy (.mov)")
    else:
        print(f"   • Export Format: H.264/H.265 (.mkv)")

    # Get audio duration using ffprobe
    audio_duration = get_video_duration(audio_file)
    if end_time and end_time > start_time:
        audio_duration = end_time - start_time
    elif start_time > 0:
        audio_duration = audio_duration - start_time
    render_info["audio_duration"] = float(audio_duration)
    
    print(f"🎵 Audio duration: {audio_duration:.2f} seconds")


    # Build one frame-locked output timeline before creating clips.
    selected_beats, segment_frames, segment_durations, dropped_boundaries = build_frame_aligned_cut_timeline(
        beat_times, audio_duration, fps
    )
    total_clips = len(segment_durations)
    render_info.update({
        "render_cuts": int(total_clips),
        "timeline_boundaries": int(len(selected_beats)),
        "timeline_frames": int(sum(segment_frames)),
    })

    print(f"🎬 Creating video with {total_clips} frame-locked cuts")
    print(f"⏱️  Cut timeline: {len(selected_beats)} boundaries, {sum(segment_frames)} frames")
    if dropped_boundaries:
        print(f"   ⚠️  Dropped {dropped_boundaries} duplicate/too-close cut boundaries after frame quantization")
    planned_clip_sequence = build_planned_clip_sequence(
        cut_times=selected_beats,
        segment_durations=segment_durations,
        beat_info=beat_info,
        video_files=video_files,
        strict_unique_non_overlap=strict_unique_non_overlap,
        preferred_videos=preferred_videos,
        edge_buffer_seconds=edge_buffer_seconds,
        clip_order_mode=clip_order_mode,
    )
    if not planned_clip_sequence:
        if strict_unique_non_overlap:
            planned_clip_sequence = _build_non_overlapping_fallback_sequence(
                segment_durations, video_files, preferred_videos, edge_buffer_seconds=edge_buffer_seconds,
                clip_order_mode=clip_order_mode,
            )
        else:
            planned_clip_sequence = _build_legacy_random_fallback_sequence(
                segment_durations, video_files, preferred_videos, edge_buffer_seconds=edge_buffer_seconds,
                clip_order_mode=clip_order_mode,
            )

    if planned_clip_sequence:
        plan_summary = summarize_clip_plan(planned_clip_sequence)
        if beat_info is not None:
            beat_info['clip_plan_summary'] = plan_summary
            render_info["plan_summary"] = plan_summary
        print(f"🧠 Auto visual planner: {plan_summary['clip_count']} planned clips")
        print(f"   Sources used: {plan_summary.get('source_count', 0)}")
        print(f"   Targets: {plan_summary.get('targets', {})}")
        print(f"   AI-tagged source moments used: {plan_summary.get('ai_tagged', 0)}")
    else:
        if strict_unique_non_overlap:
            raise RuntimeError(
                "Not enough unique non-overlapping source material for the requested timeline. "
                "Add more/longer source videos or reduce output duration."
            )
        raise RuntimeError("No source clip plan could be created.")

    # LOSSLESS MODE - ProRes workflow with FRAME-PERFECT precision
    if lossless_mode:
        print(f"\n{'='*60}")
        print(f"🎯 LOSSLESS MODE: Converting videos to ProRes 422 Proxy")
        print(f"{'='*60}")
        
        # Create ProRes conversion directory
        prores_dir = os.path.join(session_temp_dir, 'prores')
        os.makedirs(prores_dir, exist_ok=True)
        
        # Use detected FPS for ProRes conversion
        prores_fps = fps
        print(f"🎞️ Using FPS: {prores_fps} (for frame-perfect precision)")
        
        # Convert all input videos to ProRes (video only, no audio)
        prores_files = []
        prores_map = {}
        for idx, video_file in enumerate(video_files, 1):
            print(f"Converting {idx}/{len(video_files)}...")
            prores_file = convert_to_prores_proxy(video_file, prores_dir, prores_fps)
            prores_files.append(prores_file)
            prores_map[os.path.abspath(video_file)] = prores_file
        
        print(f"✓ All videos converted to ProRes 422 Proxy (video only)")
        
        # Create segments from ProRes files with FRAME-PERFECT precision
        print(f"\n{'='*60}")
        print(f"✂️  EXTRACTING SEGMENTS (FRAME-PERFECT PRECISION)")
        print(f"{'='*60}")
        print(f"   Mode: Frame-accurate re-encoding")
        print(f"   Method: Exact frame count calculation")
        print(f"   Playback: forward")
        print(f"   Audio: Stripped (will add music at the end)")
        print(f"   FPS: {prores_fps} (fixed)")
        
        segment_files = []
        segments_dir = os.path.join(session_temp_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        
        for i, exact_duration in enumerate(segment_durations):
            # Duration comes from the absolute frame-locked timeline.
            frame_count = int(segment_frames[i])
            planned_clip = planned_clip_sequence[i]

            source_video = os.path.abspath(planned_clip.get('video_file', ''))
            prores_file = prores_map.get(source_video)
            if not prores_file:
                raise RuntimeError(f"Planned source video not available in ProRes map: {source_video}")
            segment_start = float(planned_clip.get('start_time', 0.0))

            # Extract segment
            segment_file = extract_prores_segment_random(
                prores_file, exact_duration, prores_fps, segments_dir, i,
                start_time=segment_start
            )
            segment_files.append(segment_file)

            if (i + 1) % 10 == 0:
                print(f"   ✓ Extracted {i + 1}/{total_clips} segments (frame-perfect)")
        
        print(f"✓ Extracted all {len(segment_files)} segments (frame-perfect, video only)")
        
        # Concatenate and add audio
        print(f"\n{'='*60}")
        print(f"🔗 LOSSLESS CONCATENATION + MUSIC")
        print(f"{'='*60}")
        
        concatenate_videos_ffmpeg(
            video_files=segment_files,
            output_file=output_file,
            audio_file=audio_file,
            start_time=start_time,
            end_time=end_time,
            use_nvenc=False,  # ProRes uses stream copy
            temp_dir=session_temp_dir
        )
        
        print(f"\n{'='*60}")
        print(f"✅ LOSSLESS VIDEO CREATION COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   Method: Frame-perfect re-encoding + lossless concatenation")
        print(f"   Quality: ProRes 422 Proxy (lossless)")
        print(f"   Audio: Music track from input file")
        print(f"   FPS: {prores_fps} (fixed)")
        print(f"   Total Segments: {len(segment_files)}")
        print(f"   Timing Precision: Frame-perfect (zero drift)")
        print(f"{'='*60}\n")
        
        # Cleanup
        print(f"🧹 Cleaning up temporary files...")
        time.sleep(1.0)
        gc.collect()
        
        for segment_file in segment_files:
            try:
                if os.path.exists(segment_file):
                    os.remove(segment_file)
            except Exception:
                pass
        
        for prores_file in prores_files:
            try:
                if os.path.exists(prores_file):
                    os.remove(prores_file)
            except Exception:
                pass
        
        try:
            if os.path.exists(segments_dir):
                shutil.rmtree(segments_dir, ignore_errors=True)
            if os.path.exists(prores_dir):
                shutil.rmtree(prores_dir, ignore_errors=True)
        except Exception:
            pass
        
        print(f"✓ Cleanup complete")
        
        return output_file
    
    # STANDARD MODE - Direct parallel processing (NO BATCHES)
    else:
        # Get target resolution from first video
        target_size = get_video_resolution(video_files[0])
        render_info["target_resolution"] = f"{target_size[0]}x{target_size[1]}"
        print(f"🎞️ Target resolution: {target_size[0]}x{target_size[1]}")
        
        print(f"\n{'='*60}")
        print(f"🎬 PROCESSING ALL CLIPS (No batch processing with FFmpeg)")
        print(f"   Total clips: {total_clips}")
        print(f"   Parallel workers: {max_workers}")
        print(f"   Frame-accurate: ENABLED")
        if use_nvenc:
            print(f"   Encoder: ⚡ NVIDIA {gpu_encoder.upper()} (GPU-accelerated)")
        else:
            print(f"   Encoder: 💻 libx264 (CPU)")
        print(f"{'='*60}\n")
        
        clip_args = []
        for i, final_duration in enumerate(segment_durations):
            # Duration comes from the absolute frame-locked cut timeline.
            planned_clip = planned_clip_sequence[i]
            video_file = planned_clip.get('video_file')
            clip_args.append((i, video_file, final_duration,
                            target_size, use_nvenc, gpu_encoder, session_temp_dir, fps,
                            planned_clip))
        
        clip_files = [None] * len(clip_args)
        clip_timings: List[float] = []
        clip_stage_started = time.perf_counter()
        
        # Process all clips in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(create_clip_parallel, args): idx 
                for idx, args in enumerate(clip_args)
            }
            
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_tuple = future.result()
                    if len(result_tuple) >= 6:
                        i, clip_path, new_target_size, temp_path, error, clip_elapsed = result_tuple
                    else:
                        i, clip_path, new_target_size, temp_path, error = result_tuple
                        clip_elapsed = 0.0
                    
                    if clip_elapsed:
                        clip_timings.append(float(clip_elapsed))
                    
                    if error:
                        print(f"⚠️  Warning: Clip {i+1} failed after {_fmt_seconds(clip_elapsed)}: {error}")
                        continue
                    
                    if clip_path is not None:
                        clip_files[idx] = clip_path
                        
                        completed += 1
                        if completed % 10 == 0 or completed == len(clip_args):
                            progress = (completed / len(clip_args)) * 100
                            elapsed = time.perf_counter() - clip_stage_started
                            rate = completed / max(0.001, elapsed)
                            print(
                                f"   ⚡ Progress: {completed}/{len(clip_args)} clips ({progress:.1f}%) "
                                f"[{_fmt_seconds(elapsed)}, {rate:.2f} clips/s]"
                            )
                    
                except Exception as e:
                    print(f"⚠️  Warning: Error processing clip: {str(e)}")
                    continue
        
        clip_stage_seconds = time.perf_counter() - clip_stage_started
        _summarize_clip_timings(clip_timings, clip_stage_seconds)

        # Do not silently drop failed clips. Dropping one segment compresses the
        # output timeline and makes every later cut drift against the audio.
        failed_count = sum(1 for f in clip_files if f is None)
        if failed_count:
            raise RuntimeError(
                f"{failed_count} clip(s) failed; refusing to concatenate an incomplete timeline."
            )

        if not clip_files:
            raise ValueError('No valid video clips could be created')
 
        print(f"\n{'='*60}")
        print(f"🎬 FINAL ASSEMBLY: Concatenating {len(clip_files)} clips")
        print(f"{'='*60}\n")
        
        # Concatenate all clips and add audio
        assembly_started = time.perf_counter()
        concatenate_videos_ffmpeg(
            video_files=clip_files,
            output_file=output_file,
            audio_file=audio_file,
            start_time=start_time,
            end_time=end_time,
            use_nvenc=use_nvenc,
            gpu_encoder=gpu_encoder,
            fps=fps,
            temp_dir=session_temp_dir
        )
        assembly_seconds = time.perf_counter() - assembly_started
        render_info["final_assembly_seconds"] = float(assembly_seconds)
        print(f"   ⏱ Final assembly total: {_fmt_seconds(assembly_seconds)}")
 
        print(f"\n🧹 Cleaning up resources...")
        
        # Cleanup clip files
        for clip_file in clip_files:
            try:
                if os.path.exists(clip_file):
                    os.remove(clip_file)
            except Exception as e:
                print(f"⚠️  Warning: Could not delete clip file: {e}")
        
        # Clean up processing directory
        try:
            if os.path.exists(session_temp_dir):
                shutil.rmtree(session_temp_dir, ignore_errors=True)
                print(f"✓ Cleaned up processing directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete processing directory: {e}")
        
        gc.collect()
 
        print(f"\n{'='*60}")
        print(f"✅ VIDEO CREATION COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   FPS: {fps} (frame-accurate)")
        print(f"   Total Cuts: {total_clips}")
        print(f"   Zero Drift: Absolute frame-locked cut timeline")
        print(f"   Total video creation time: {_fmt_seconds(time.perf_counter() - video_creation_started)}")
        print(f"{'='*60}\n")
        
        return output_file
 
 
def main() -> None:
    args = parse_arguments()
 
    if not os.path.exists(args.mp3_file):
        raise FileNotFoundError('Audio file not found: ' + args.mp3_file)
 
    if not os.path.isdir(args.video_directory):
        raise NotADirectoryError('Video directory not found: ' + args.video_directory)
 
    # Enable GPU mode if requested
    if args.gpu:
        if GPU_AVAILABLE:
            set_gpu_mode(True)
            print(f"⚡ GPU acceleration ENABLED: {gpu_info}")
        else:
            print(f"⚠️  GPU requested but CuPy not available, using CPU")
            args.gpu = False
 
    print(f"\n{'='*60}")
    print(f"🎵 BEATSYNC ENGINE - AUTO MODE")
    print(f"   Audio Analysis: {'⚡ GPU' if args.gpu else '💻 CPU'}")
    if args.gpu and NVENC_AVAILABLE and not args.lossless:
        print(f"   Video Encoding: ⚡ {args.gpu_encoder.upper()}")
    else:
        print(f"   Video Encoding: 💻 CPU")
    if args.fps:
        print(f"   FPS: {args.fps} (custom)")
    else:
        print(f"   FPS: Auto-detect from input video")
    print(f"   Mode: AUTO")
    if args.lossless:
        print(f"   Export: 🎯 Lossless/Precise (ProRes 422 Proxy - Frame Perfect)")
    else:
        print(f"   Export: 📹 H.264/H.265 (.mkv) - Frame Accurate")
    print(f"{'='*60}\n")
    
    print(f'📁 Audio file: {args.mp3_file}')
    video_files = get_video_files(args.video_directory)
    print(f'✓ Found {len(video_files)} video files')
 
    print(f"🤖 Using AUTO mode")
    selected_beats, beat_info = analyze_beats_auto(
        args.mp3_file,
        start_time=args.start_time,
        end_time=args.end_time,
        use_gpu=args.gpu,
        video_files=video_files,
    )
 
    print(f'✓ Selected {len(selected_beats)} cuts for video')
 
    output_file = args.output
    if args.lossless and not output_file.lower().endswith('.mov'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mov'
        print(f'📝 Changed output to .mov for Lossless mode: {output_file}')
    elif not args.lossless and not output_file.lower().endswith('.mkv'):
        base, _ = os.path.splitext(output_file)
        output_file = base + '.mkv'
        print(f'📝 Changed output to .mkv: {output_file}')
 
    print(f'\n🎬 Starting video creation (frame-accurate)...\n')
    output_file = create_music_video(
        args.mp3_file,
        video_files,
        selected_beats,  # Pass pre-processed beats
        output_file=output_file,
        start_time=args.start_time,
        end_time=args.end_time,
        max_workers=PARALLEL_WORKERS,
        beat_info=beat_info,
        lossless_mode=args.lossless,
        use_gpu=args.gpu,
        gpu_encoder=args.gpu_encoder,
        fps=args.fps,
        strict_unique_non_overlap=True,
        edge_buffer_seconds=args.edge_buffer_seconds,
        clip_order_mode=args.clip_order_mode,
    )
 
    print(f'✅ Music video created successfully: {output_file}')
 
 
if __name__ == '__main__':
    main()
