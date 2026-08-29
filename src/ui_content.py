#!/usr/bin/env python3
"""
UI Content for BeatSync Engine
Focused on Auto Mode.
"""

import os

# ============================================================================
# MAIN UI CONTENT
# ============================================================================

UI_TITLE = "🎵 BeatSync Engine"

UI_MAIN_DESCRIPTION = """Create music videos that cut to the beat. Upload audio and video clips 
to automatically generate a video synchronized with your music's rhythm."""

# ============================================================================
# SYSTEM STATUS
# ============================================================================

def get_system_performance_info(cpu_count, max_threads, parallel_workers, python_status, 
                                cuda_status, ffmpeg_status, gpu_status, gpu_device, nvenc_status):
    """System performance overview."""
    return f"""## 🚀 System Status
- **Python**: {python_status} | **CUDA**: {cuda_status}
- **CPU**: {cpu_count} threads | **FFmpeg**: {ffmpeg_status}
- **GPU**: {gpu_status}{gpu_device} | **NVENC**: {nvenc_status}
- **Parallel Processing**: {parallel_workers} workers | **Input**: Local project folder"""

PORTABLE_SETUP_INFO = """## 📦 Portable Setup
Self-contained installation - Python 3.13.14, CUDA 13.3, and FFmpeg included. 
No system dependencies required."""

GPU_ACCELERATION_INFO = """## ⚡ GPU Acceleration
- **Audio Analysis**: 5-10x faster with GPU (CuPy + CUDA)
- **Video Encoding**: 2-3x faster with NVENC hardware encoder
- **Auto-Detection**: Uses GPU automatically when NVIDIA card detected"""

NVENC_BENEFITS_INFO = """## 🎬 NVENC Hardware Encoding
2-3x faster than CPU encoding with comparable quality. Automatically enabled when available."""

EXPORT_OPTIONS_INFO = """## 🎬 Export Modes
- **NVIDIA NVENC H.264**: GPU-accelerated H.264 (fast, .mkv)
- **NVIDIA NVENC HEVC**: GPU-accelerated H.265 (smaller files, .mkv)
- **CPU H.264**: Software encoding (.mkv)
- **ProRes 422 Proxy**: Frame-perfect lossless (.mov)"""

PRORES_MODE_INFO = """## 🎯 ProRes 422 Proxy Mode
Frame-perfect cuts with zero quality loss. Converts input to ProRes (I-frames only), 
then uses lossless concatenation. Larger files, perfect accuracy.

**How it works:**
1. Converts input videos to ProRes 422 Proxy (all I-frames)
2. Extracts segments with exact frame counts (frame-perfect)
3. Concatenates with stream copy (zero quality loss)
4. Adds music track with PCM audio

**Best for:** Professional editing, archival, maximum quality"""

# ============================================================================
# MODE DESCRIPTION
# ============================================================================

AUTO_MODE_DESCRIPTION = """**🤖 Auto Mode - Audio-Visual Intelligence**

Fully automatic music + video analysis with rhythmic source matching:

**Features:**
- 🎵 **Song Structure Detection**: Intro/Verse/Chorus/Bridge/Outro
- ⚡ **Energy Analysis**: High/Medium/Low energy per section
- 🎯 **Rhythm Pattern Recognition**: Kick/Clap/Bass/Hi-hat patterns
- 🎬 **Video Moment Analysis**: Motion, quality, scene changes, action/beauty
- 🧠 **Qwen3-VL Semantic Tags**: Drop/soft/build/action/emotion matching
- 🎼 **Planned Clip Selection**: Chooses source moments instead of random clips

**How it works:**
- Analyzes your song's energy and structure
- Builds a cache of strong source-video moments
- Matches aggressive drops to action/high-motion scenes
- Matches soft melodic parts to beautiful or emotional scenes
- Detects dominant rhythm patterns per section
- Automatically adjusts cut frequency
- Keeps the edit rhythmic while avoiding weak or repetitive source moments

**Example:**
- Aggressive drop → Action/combat/chase/high-motion shots
- Beautiful bridge → Soft, clean, emotional shots
- Build-up → Tension/camera-motion shots
- Fast rhythmic chorus → More frequent beat-locked cuts

**Best for:** Complete hands-off AMV/GMV creation with rhythmic visual matching"""

# ============================================================================
# PERFORMANCE GUIDE
# ============================================================================

def get_performance_guide(cpu_count, parallel_workers, python_status, cuda_status, 
                         ffmpeg_status, gpu_available, gpu_info, nvenc_available):
    """Concise performance guide."""
    
    gpu_text = ""
    if gpu_available:
        nvenc_text = "NVENC: 2-3x faster encoding" if nvenc_available else ""
        gpu_text = f"""**GPU Acceleration** ({gpu_info}):
- Audio analysis: 5-10x faster
- {nvenc_text}

"""
    
    return f"""**System**: {cpu_count} CPU threads | {parallel_workers} parallel workers

**Portable Components**:
- Python: {python_status}
- CUDA: {cuda_status}
- FFmpeg: {ffmpeg_status}

{gpu_text}**Processing Modes**:
- **NVENC H.264**: GPU-accelerated (fastest)
- **NVENC HEVC**: GPU-accelerated (smaller files)
- **CPU H.264**: Software encoding
- **ProRes 422 Proxy**: Lossless (converts to ProRes, then concatenates)

**Temp Files**: `./temp/` folder (local only, auto-cleaned)

**FPS**: Leave empty to auto-detect from input video, or set custom (24/30/60)

**Speed Estimates** (3-min video):
- CPU only: ~2-3 min
- GPU + NVENC: ~45-90 sec ⚡
- ProRes: ~3-5 min (conversion) + instant (concat)

**RAM Usage**: Constant with FFmpeg (no batch processing needed)

**Parallel Workers**:
- More workers = faster processing
- Recommended: {parallel_workers} (auto-calculated from CPU)
- With NVENC: Can use more workers (GPU handles encoding)"""

# ============================================================================
# GUIDES
# ============================================================================

NVENC_GUIDE_ACTIVE = """**🚀 NVENC Auto-Enabled!**
GPU hardware encoding: 2-3x faster than CPU.
Select NVIDIA NVENC H.264 or HEVC for best performance.

---

"""

QUICK_START_GUIDE = """### 💡 Quick Start

**🤖 Auto Mode:**
1. Upload audio + videos
2. Click "Create Music Video"
3. Done! Automatic audio-visual cuts

**🎯 Lossless (ProRes):**
- Select **ProRes 422 Proxy** mode for frame-perfect quality

---

"""

# ============================================================================
# SYSTEM INFO PANEL
# ============================================================================

def get_system_info_panel(python_status, cuda_status, cpu_count, max_threads, 
                         ffmpeg_status, gpu_status, gpu_device, nvenc_status, 
                         librosa_version):
    """Compact system info."""
    return f"""**System:**
Python: {python_status} | CUDA: {cuda_status}
CPU: {cpu_count} cores ({max_threads} threads)
FFmpeg: {ffmpeg_status} | Librosa: {librosa_version}
GPU: {gpu_status}{gpu_device} | NVENC: {nvenc_status}

**Files:**
- Python: `bin/python-3.13.14-embed-amd64/`
- CUDA: `bin/CUDA/v13.3/`
- FFmpeg: `bin/ffmpeg/ffmpeg.exe`
- Temp: `./temp/` (local only)
- Output: `./output/`

**Mode:**
- 🤖 Auto: Audio-visual rhythmic intelligence"""

# ============================================================================
# STATUS MESSAGES
# ============================================================================

def get_ready_status(python_status, cuda_status, max_threads, cpu_count, ffmpeg_status,
                    gpu_available, gpu_info, nvenc_available):
    """Ready status message."""
    return '✅ Ready to process!\n\nUpload audio and video files to begin.'

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

def _format_auto_section_summary(sections_info):
    """Return a safe section summary for all Auto Mode versions.

    Supports both the older flat dictionaries:
        {"section": "chorus", "selected_beats": 8, "total_beats": 32, "selection_ratio": 0.25}
    and the newer V3.2 wave dictionaries:
        {"section": {"type": "chorus", ...}, "selected_count": 8, "beat_count": 32, "density": 0.25}
    """
    if not sections_info:
        return ""

    lines = ["Sections analyzed and processed:"]
    for item in sections_info[:12]:
        if not isinstance(item, dict):
            continue

        raw_section = item.get('section', item.get('type', 'section'))
        if isinstance(raw_section, dict):
            section_name = raw_section.get('type') or raw_section.get('section') or raw_section.get('name') or 'section'
        else:
            section_name = raw_section

        section_name = str(section_name).replace('_', ' ').strip().title() or 'Section'

        selected = item.get('selected_beats', item.get('selected_count', item.get('cuts', 0)))
        total = item.get('total_beats', item.get('beat_count', item.get('beats', 0)))
        ratio = item.get('selection_ratio', item.get('density', None))

        try:
            selected_i = int(selected)
        except Exception:
            selected_i = 0
        try:
            total_i = int(total)
        except Exception:
            total_i = 0

        if ratio is None:
            ratio = selected_i / total_i if total_i > 0 else 0.0
        try:
            ratio_f = float(ratio)
        except Exception:
            ratio_f = 0.0

        lines.append(f"      - {section_name}: {selected_i}/{total_i} beats ({ratio_f * 100:.1f}%)")

    if len(sections_info) > 12:
        lines.append(f"      - ... plus {len(sections_info) - 12} more sections")

    return "\n".join(lines)


def get_success_message_auto(total_cuts, total_beats, tempo, sections_info,
                            python_str, cuda_str, max_threads, cpu_count,
                            parallel_workers, gpu_info, encoder_info,
                            codec_info, fps_info, filename, audio_info,
                            audio_duration=None, output_fps=None,
                            total_processing_seconds=None, processing_label=None):
    """Success message for Auto mode. Compatible with Auto Mode V1/V2/V3/V3.2."""

    section_summary = _format_auto_section_summary(sections_info)
    try:
        audio_duration_f = float(audio_duration)
    except Exception:
        audio_duration_f = 0.0
    try:
        output_fps_f = float(output_fps)
    except Exception:
        output_fps_f = 0.0
    try:
        total_seconds_i = int(round(float(total_processing_seconds)))
    except Exception:
        total_seconds_i = 0
    processing_text = processing_label or encoder_info
    fps_text = f"{output_fps_f:.1f}" if output_fps_f else str(fps_info).split()[0]

    return f"""✅ Video created successfully!

Statistics:
Video processing: {processing_text}
Total cuts: {total_cuts}
Audio duration: {audio_duration_f:.2f} seconds
Output FPS: {fps_text}
{total_beats} beats detected at {tempo:.1f} BPM

{section_summary}
Total time processing: {total_seconds_i} seconds

Output: {filename}"""


# ============================================================================
# CONSOLE MESSAGES
# ============================================================================

CONSOLE_SEPARATOR = "=" * 70

def get_startup_header(cpu_count, max_threads, parallel_workers, python_status, 
                      cuda_status, librosa_version, ffmpeg_status, gpu_available, 
                      gpu_info, nvenc_available):
    """Startup header."""
    gpu_line = f"   GPU: {gpu_info} (Auto-enabled)" if gpu_available else "   GPU: Not available (CPU only)"
    nvenc_line = f"   NVENC: Available (Auto-enabled)" if nvenc_available else "   NVENC: Not available"
    
    return f"""{CONSOLE_SEPARATOR}
🎵 BeatSync Engine
{CONSOLE_SEPARATOR}
   Python: {python_status}
   CUDA: {cuda_status}
   FFmpeg: {ffmpeg_status}
   Librosa: {librosa_version}
   CPU: {cpu_count} threads ({max_threads} max per encode)
   Parallel Workers: {parallel_workers}
   {gpu_line}
   {nvenc_line}
   Mode: 🤖 Auto
   ProRes 422 Proxy: ENABLED"""

# ============================================================================
# INPUT LABELS & INFO
# ============================================================================

LABEL_AUDIO_FILE = "🎵 Audio File (MP3/WAV/FLAC)"
LABEL_VIDEO_FILES = "🎥 Video & Image Files (MP4/MKV/JPG/HEIC)"
LABEL_VIDEO_FOLDER = "📂 Oder: Lokaler Videoordner (Upload überspringen)"
INFO_VIDEO_FOLDER = "Pfad zu einem Ordner mit Videos/Bildern auf diesem PC. Umgeht den Browser-Upload und ist bei großen Dateien deutlich schneller. Hat Vorrang vor hochgeladenen Dateien."
LABEL_PREFERRED_VIDEOS = "⭐ Bevorzugte Videos (mehr Clips daraus verwenden)"
INFO_PREFERRED_VIDEOS = "Markierte Videos werden im fertigen Video bevorzugt und liefern mehr Clips."

LABEL_CUSTOM_FPS = "🎞️ Custom FPS (Frame Rate)"
INFO_CUSTOM_FPS = "Leave empty for auto-detect, or enter value (24/30/60)"

LABEL_EDGE_BUFFER_SECONDS = "✂️ Rand-Puffer (Sekunden)"
INFO_EDGE_BUFFER_SECONDS = "Anfang und Ende jedes Quellvideos werden um diese Anzahl Sekunden nicht für Clips verwendet."

LABEL_CLIP_ORDER_MODE = "🔀 Clip-Reihenfolge"
INFO_CLIP_ORDER_MODE = (
    "Legt fest, in welcher Reihenfolge die Quellvideos im fertigen Video verwendet werden. "
    "Chronologisch nutzt das echte Aufnahmedatum aus den EXIF-/Container-Metadaten; "
    "fehlt es, wird auf das Datei-Änderungsdatum zurückgegriffen."
)

LABEL_FIRST_VIDEO = "⏮️ Start-Video"
INFO_FIRST_VIDEO = "Dieses Video liefert den allerersten Clip im fertigen Video."
LABEL_LAST_VIDEO = "⏭️ End-Video"
INFO_LAST_VIDEO = "Dieses Video liefert den allerletzten Clip im fertigen Video."
NO_VIDEO_SELECTION_LABEL = "(keine Auswahl)"

LABEL_START_TEXT = "📝 Start-Text"
LABEL_END_TEXT = "📝 End-Text"
LABEL_TEXT_POSITION = "Position"
LABEL_TEXT_DURATION = "Dauer (Sekunden)"
INFO_START_TEXT = "Wird am Anfang des Zielvideos eingeblendet. Leer lassen, um keine Einblendung zu verwenden."
INFO_END_TEXT = "Wird am Ende des Zielvideos eingeblendet. Leer lassen, um keine Einblendung zu verwenden."
TEXT_POSITION_CHOICES = [
    ('Oben links', 'top_left'), ('Oben Mitte', 'top_center'), ('Oben rechts', 'top_right'),
    ('Mitte links', 'middle_left'), ('Zentriert', 'middle_center'), ('Mitte rechts', 'middle_right'),
    ('Unten links', 'bottom_left'), ('Unten Mitte', 'bottom_center'), ('Unten rechts', 'bottom_right'),
]

LABEL_TEXT_FONT = "🔤 Schriftart"
INFO_TEXT_FONT = "Schriftart für Start-/End-Text. Die Vorschau darunter zeigt, wie sie aussieht."

LABEL_FADE_ENABLED = "🎬 Ein-/Ausblenden (Schwarzblende)"
INFO_FADE_ENABLED = "Video und Ton am Anfang aus Schwarz einblenden und am Ende nach Schwarz ausblenden. Start-/End-Text blendet in der Schwarzblende mit ein bzw. aus."
LABEL_FADE_DURATION = "Blenden-Dauer (Sekunden)"

WINDOWS_FONTS_DIR = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')

# (display label, browser-preview CSS font stack, filename in the Windows fonts dir)
_TEXT_FONT_CATALOG = [
    ("Arial", "Arial, sans-serif", "arial.ttf"),
    ("Arial Black", "'Arial Black', Arial, sans-serif", "ariblk.ttf"),
    ("Impact", "Impact, 'Arial Black', sans-serif", "impact.ttf"),
    ("Bahnschrift", "Bahnschrift, 'DIN Alternate', sans-serif", "bahnschrift.ttf"),
    ("Verdana", "Verdana, sans-serif", "verdana.ttf"),
    ("Tahoma", "Tahoma, sans-serif", "tahoma.ttf"),
    ("Trebuchet MS", "'Trebuchet MS', sans-serif", "trebuc.ttf"),
    ("Segoe UI", "'Segoe UI', sans-serif", "segoeui.ttf"),
    ("Calibri", "Calibri, sans-serif", "calibri.ttf"),
    ("Franklin Gothic", "'Franklin Gothic Medium', Arial, sans-serif", "framd.ttf"),
    ("Georgia", "Georgia, serif", "georgia.ttf"),
    ("Times New Roman", "'Times New Roman', serif", "times.ttf"),
    ("Courier New", "'Courier New', monospace", "cour.ttf"),
    ("Comic Sans MS", "'Comic Sans MS', cursive", "comic.ttf"),
]

DEFAULT_TEXT_FONT = "Arial"
FONT_PREVIEW_SAMPLE = "Beispiel-Titel · AaBbGg 123"


def get_available_text_fonts() -> list:
    """(label, css_family) for the catalog fonts actually present on this machine."""
    present = [
        (label, css) for label, css, fname in _TEXT_FONT_CATALOG
        if os.path.isfile(os.path.join(WINDOWS_FONTS_DIR, fname))
    ]
    return present or [("Arial", "Arial, sans-serif")]


def get_text_font_choices() -> list:
    """Dropdown (label, value) pairs; the value is the font label itself."""
    return [(label, label) for label, _css in get_available_text_fonts()]


def get_text_font_path(font_name: str | None) -> str:
    """Absolute path to the .ttf for a font label, falling back to Arial."""
    lookup = {label: fname for label, _css, fname in _TEXT_FONT_CATALOG}
    fname = lookup.get(font_name or "", lookup[DEFAULT_TEXT_FONT])
    path = os.path.join(WINDOWS_FONTS_DIR, fname)
    if not os.path.isfile(path):
        path = os.path.join(WINDOWS_FONTS_DIR, lookup[DEFAULT_TEXT_FONT])
    return path


def render_font_preview_html(font_name: str | None, sample: str = FONT_PREVIEW_SAMPLE) -> str:
    """Small HTML card that previews `sample` in the chosen font using the browser."""
    css = dict(get_available_text_fonts()).get(font_name or "", "Arial, sans-serif")
    label = font_name or DEFAULT_TEXT_FONT
    safe = (str(sample or "").strip() or FONT_PREVIEW_SAMPLE).replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return (
        '<div style="padding:16px;border:1px solid var(--border-color-primary,#3d3d3d);'
        'border-radius:10px;background:#101010;text-align:center;">'
        f'<div style="font-family:{css};font-weight:700;color:#fff;font-size:38px;'
        'line-height:1.2;text-shadow:2px 2px 0 #000,-1px -1px 0 #000;">'
        f'{safe}</div>'
        f'<div style="font-family:{css};color:#9aa;opacity:.8;font-size:14px;margin-top:8px;">{label}</div>'
        '</div>'
    )

# GPU & Processing
LABEL_GPU_STATUS = "⚡ GPU Acceleration Status"

LABEL_PROCESSING_MODE = "🎬 Processing Mode"

# Performance
LABEL_PARALLEL_WORKERS = "⚡ Parallel Workers"
INFO_PARALLEL_WORKERS = "Clips processed simultaneously. More workers with GPU."

# Output
LABEL_OUTPUT_FILENAME = "📝 Output Filename"
INFO_OUTPUT_FILENAME = "Timestamp added automatically (.mkv or .mov)"

def get_gpu_status_info(gpu_available, gpu_info, nvenc_available):
    """GPU status info."""
    if gpu_available and nvenc_available:
        return f'✅ {gpu_info} | NVENC Enabled'
    elif gpu_available:
        return f'✅ {gpu_info} | NVENC Not available'
    else:
        return '❌ CPU mode only'

def get_processing_mode_info_nvenc():
    """Processing mode info with NVENC."""
    return 'GPU (NVENC): High quality | CPU: High quality | ProRes: Max quality'

def get_processing_mode_info_cpu():
    """Processing mode info without NVENC."""
    return 'CPU: H.264 encoding | ProRes: Max quality (NVENC not available)'

def get_parallel_workers_label(recommended_workers):
    """Parallel workers label."""
    return f'⚡ Parallel Workers (Recommended: {recommended_workers})'

def get_parallel_workers_info():
    """Parallel workers info."""
    return 'Clips processed simultaneously. More workers with GPU.'
