"""
Simple YouTube-to-Shorts engine.
Takes a video URL (direct video file link) or a local file, transcribes it,
asks Gemini to pick the best short moments, and cuts vertical (9:16) clips.

Runs on GitHub Actions (free tier) - no server needed.
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
import datetime
from pathlib import Path

import requests
from groq import Groq
from google import genai


def download_video(source: str, out_path: str) -> str:
    """Get the video to a local path. Downloads it if source is a URL,
    otherwise copies it (for running locally with a file on your PC)."""
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, stream=True, timeout=120)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        shutil.copy(source, out_path)
    return out_path


def extract_audio(video_path: str, audio_path: str) -> str:
    """Pull a small mono audio file out of the video for transcription."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            audio_path,
        ],
        check=True,
    )
    return audio_path


def transcribe_english(audio_path: str, api_key: str) -> dict:
    """Translate the audio to English with Groq's Whisper API, regardless of
    the spoken language, so captions always come out in readable English.
    Returns verbose JSON with segment-level timestamps (the translation
    endpoint does not support word-level timestamps)."""
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.translations.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    return result.model_dump() if hasattr(result, "model_dump") else json.loads(result.json())


LANGUAGE_NAME_TO_CODE = {
    "english": "en", "en": "en",
    "urdu": "ur", "ur": "ur",
    "hindi": "hi", "hi": "hi",
    "arabic": "ar", "ar": "ar",
    "spanish": "es", "es": "es",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "punjabi": "pa", "pa": "pa",
    "bengali": "bn", "bn": "bn",
    "turkish": "tr", "tr": "tr",
    "chinese": "zh", "zh": "zh",
    "russian": "ru", "ru": "ru",
    "portuguese": "pt", "pt": "pt",
    "indonesian": "id", "id": "id",
}


def resolve_language_code(value: str) -> str:
    """Accept either a language name ('urdu', 'english') or a short code
    ('ur', 'en') and return the ISO code Whisper expects."""
    key = value.strip().lower()
    return LANGUAGE_NAME_TO_CODE.get(key, key)


def transcribe_in_language(audio_path: str, api_key: str, language_code: str) -> dict:
    """Transcribe the audio in its own language (no translation), for
    caption-only mode when a specific caption language is requested."""
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            language=language_code,
        )
    return result.model_dump() if hasattr(result, "model_dump") else json.loads(result.json())


def get_video_duration(video_path: str) -> float:
    """Read the video's total duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def transliterate_segments_to_latin(segments: list, api_key: str) -> list:
    """Rewrite each segment's text using ONLY the English/Latin alphabet
    (a casual Roman transliteration, like Roman Urdu) instead of its native
    script, so captions always display in Latin letters regardless of the
    spoken language. Meaning/sounds are kept, timestamps are untouched."""
    if not segments:
        return segments

    client = genai.Client(api_key=api_key)
    texts = [s.get("text", "") for s in segments]
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))

    prompt = f"""Rewrite each numbered line below using ONLY the English/Latin alphabet -
a casual Roman transliteration of how it sounds (for example Roman Urdu, Romanized
Hindi/Arabic, etc). Do NOT translate the meaning into English and do NOT use any
native-script characters (no Urdu, Arabic, Devanagari, etc) anywhere in the output.
Keep the same line numbers and the same number of lines.

Lines:
{numbered}

Reply with ONLY a JSON object mapping each line number (as a string) to its
transliterated text. No other text, no markdown fences. Example:
{{"0": "...", "1": "..."}}
"""
    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    mapping = json.loads(text)

    new_segments = []
    for i, seg in enumerate(segments):
        new_seg = dict(seg)
        new_seg["text"] = mapping.get(str(i), seg.get("text", ""))
        new_segments.append(new_seg)
    return new_segments


def pick_clips(transcript: dict, api_key: str, min_clips: int, max_clips: int) -> list:
    """Ask Gemini which segments make good short clips."""
    client = genai.Client(api_key=api_key)

    segments = transcript.get("segments", [])
    transcript_text = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments
    )

    prompt = f"""You are picking short, engaging clips (15-60 seconds each) from a video transcript
for YouTube Shorts / TikTok / Reels. Pick between {min_clips} and {max_clips} clips.
Write each title using ONLY the English/Latin alphabet (a Roman transliteration if the
transcript isn't in English) - never native-script characters such as Urdu, Arabic, or
Devanagari letters.

Transcript with timestamps:
{transcript_text}

Reply with ONLY a JSON array, no other text, no markdown fences, in this exact format:
[{{"start": 12.5, "end": 45.0, "title": "short catchy title"}}]
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", contents=prompt
    )
    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def pick_label(transcript: dict, api_key: str) -> str:
    """Ask Gemini for one short, catchy title describing the whole video
    (used in caption-only mode, instead of picking multiple clips)."""
    client = genai.Client(api_key=api_key)
    segments = transcript.get("segments", [])
    transcript_text = "\n".join(s["text"] for s in segments)

    prompt = f"""Give one short, catchy title (under 10 words) that describes what this video is about.
Write it using ONLY the English/Latin alphabet (a Roman transliteration if needed) -
never native-script characters such as Urdu, Arabic, or Devanagari letters.

Transcript:
{transcript_text}

Reply with ONLY the title text. No quotes, no markdown, nothing else.
"""
    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    return response.text.strip().strip('"')


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "but", "it", "this", "that", "you", "i", "we",
    "they", "he", "she", "know", "just", "so", "like", "with", "for",
    "as", "be", "been", "from", "your", "my", "our", "not", "do",
    "did", "will", "would", "can", "could", "there", "what", "when",
}

WORD_COLORS = [
    (255, 235, 59),   # yellow
    (0, 229, 255),    # cyan
    (255, 64, 129),   # pink
    (255, 152, 0),    # orange
    (76, 217, 100),   # green
    (255, 82, 82),    # red
    (186, 104, 200),  # purple
]

# Caption text sizes (in pixels on the 1080x1920 frame)
DEFAULT_IMPORTANT_SIZE = 92   # important (highlighted) words
DEFAULT_COMMON_SIZE = 64      # common words like "the", "and", "in"
MIN_TEXT_SIZE = 40
MAX_TEXT_SIZE = 140

# Caption look: "classic" = captions at the bottom (original look),
# "behind" = big captions placed in the middle of the video that appear
# BEHIND the person (the person is cut out and drawn over the text).
CAPTION_STYLES = ("classic", "behind")
BEHIND_IMPORTANT_SIZE = 130   # default sizes for the "behind" look (bigger text)
BEHIND_COMMON_SIZE = 72       # size of the normal caption line at the bottom in the "behind" look
BOTTOM_WORDS_PER_LINE = 5
# What the bottom line shows in the "behind" look:
#   "full"      = the whole phrase (important words colored), so it reads naturally
#   "remaining" = only the words that are NOT shown big behind the person
BEHIND_BOTTOM_MODE = "full"
BEHIND_TEXT_Y = 900           # vertical centre of the caption on the 1920px frame
BEHIND_FPS = 30               # the "behind" look renders at a fixed 30 fps so the mask stays in sync
MASK_WORK_SIZE = 512          # the person mask is computed at this long-side size, then scaled up


def parse_caption_colors(value):
    """Turn 'FFEB3B,00E5FF,#FF4081' into a list of (r, g, b) tuples.
    Invalid entries are skipped. If nothing valid is left, the default
    color set is used."""
    if not value:
        return list(WORD_COLORS)
    colors = []
    for part in value.split(","):
        hex_value = part.strip().lstrip("#")
        if re.fullmatch(r"[0-9A-Fa-f]{6}", hex_value):
            colors.append((
                int(hex_value[0:2], 16),
                int(hex_value[2:4], 16),
                int(hex_value[4:6], 16),
            ))
    return colors or list(WORD_COLORS)


def clamp_text_size(value, default: int) -> int:
    """Keep a text size inside a safe range so captions always fit the frame."""
    try:
        size = int(value)
    except (TypeError, ValueError):
        return default
    return max(MIN_TEXT_SIZE, min(MAX_TEXT_SIZE, size))


def rgb_to_ass_inline(r: int, g: int, b: int) -> str:
    """ASS inline color override tags use BGR hex order, no alpha byte."""
    return f"\\c&H{b:02X}{g:02X}{r:02X}&"


def format_ass_time(seconds: float) -> str:
    cs = int(round(max(0, seconds) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def sanitize_ass_text(word: str) -> str:
    return word.replace("{", "(").replace("}", ")").replace("\\", "")


def _ass_header(caption_font_family: str, important_size: int) -> str:
    """Shared ASS header with the Default (classic), Behind and Title styles."""
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{caption_font_family},{important_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,2,10,10,250,1\n"
        f"Style: Behind,{caption_font_family},{important_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,4,0,5,60,60,0,1\n"
        "Style: Title,DejaVu Sans,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,4,0,8,40,40,90,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _is_common_word(word: str) -> bool:
    return word.strip(".,!?;:\"'").lower() in STOPWORDS


def build_behind_ass(
    segments: list,
    clip_start: float,
    clip_end: float,
    behind_ass_path: str,
    front_ass_path: str,
    title: str = None,
    title_duration: float = 2.5,
    caption_font_family: str = "DejaVu Sans",
    colors: list = None,
    important_size: int = BEHIND_IMPORTANT_SIZE,
    common_size: int = BEHIND_COMMON_SIZE,
):
    """'Behind the person' look, written as TWO subtitle files:
      behind_ass_path - each important word, BIG, in the middle of the frame.
                        It is drawn under the person cut-out.
      front_ass_path  - the title at the top and the normal caption line at the
                        bottom. It is drawn on top of everything."""
    palette = colors or WORD_COLORS

    chunks = []  # (start, end, words) relative to the clip
    for seg in segments:
        if not (seg["end"] > clip_start and seg["start"] < clip_end):
            continue
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = seg["text"].strip().split()
        if not words:
            continue
        n_chunks = max(1, -(-len(words) // BOTTOM_WORDS_PER_LINE))
        chunk_duration = (seg_end - seg_start) / n_chunks
        for chunk_idx, i in enumerate(range(0, len(words), BOTTOM_WORDS_PER_LINE)):
            c_start = max(0, seg_start + chunk_idx * chunk_duration - clip_start)
            chunks.append((c_start, c_start + chunk_duration, words[i:i + BOTTOM_WORDS_PER_LINE]))

    header = _ass_header(caption_font_family, important_size)
    behind_lines, front_lines = [], []

    if title:
        front_lines.append(
            f"Dialogue: 1,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,"
            f"{sanitize_ass_text(title)}"
        )

    color_i = 0
    for start, end, words in chunks:
        n = len(words)
        word_dur = (end - start) / n
        important = [k for k, w in enumerate(words) if not _is_common_word(w)]

        # color of each important word (same color behind and at the bottom)
        word_color = {}
        for k in important:
            word_color[k] = palette[color_i % len(palette)]
            color_i += 1

        # 1) big important words, one at a time, in the order they are spoken
        for pos, k in enumerate(important):
            t0 = start if pos == 0 else start + k * word_dur  # first keyword shows right away
            t1 = start + important[pos + 1] * word_dur if pos + 1 < len(important) else end
            clean = words[k].strip(".,!?;:\"'")
            if not clean:
                continue
            safe_w = sanitize_ass_text(clean.upper())
            # shrink very long words so they stay inside the 1080px frame
            fs = max(60, min(important_size, int(940 / (len(safe_w) * 0.78))))
            r, g, b = word_color[k]
            behind_lines.append(
                f"Dialogue: 0,{format_ass_time(t0)},{format_ass_time(t1)},Behind,,0,0,0,,"
                f"{{\\an5\\pos(540,{BEHIND_TEXT_Y})\\fs{fs}{rgb_to_ass_inline(r, g, b)}}}{safe_w}"
            )

        # 2) normal caption line at the bottom
        parts = []
        for k, w in enumerate(words):
            if k in word_color:
                if BEHIND_BOTTOM_MODE == "remaining":
                    continue
                r, g, b = word_color[k]
            else:
                r, g, b = 255, 255, 255
            parts.append(f"{{{rgb_to_ass_inline(r, g, b)}\\fs{common_size}}}{sanitize_ass_text(w)}")
        if parts:
            front_lines.append(
                f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Default,,0,0,0,,"
                + " ".join(parts)
            )

    with open(behind_ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(behind_lines))
        f.write("\n")
    with open(front_ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(front_lines))
        f.write("\n")


def build_captions_ass(
    segments: list,
    clip_start: float,
    clip_end: float,
    ass_path: str,
    title: str = None,
    title_duration: float = 2.5,
    words_per_line: int = 5,
    caption_font_family: str = "DejaVu Sans",
    colors: list = None,
    important_size: int = DEFAULT_IMPORTANT_SIZE,
    common_size: int = DEFAULT_COMMON_SIZE,
):
    """Build an ASS caption file for the segments inside one clip. Common
    (stop) words stay white and smaller; other words cycle through the chosen
    bright colors word by word and use the larger size, so the caption
    highlights the meaningful words.
    If a title is given, it's shown at the top for the first few seconds."""
    palette = colors or WORD_COLORS

    clip_segments = [
        s for s in segments
        if s["end"] > clip_start and s["start"] < clip_end
    ]

    lines = []
    for seg in clip_segments:
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = seg["text"].strip().split()
        if not words:
            continue
        n_chunks = max(1, -(-len(words) // words_per_line))  # ceil division
        chunk_duration = (seg_end - seg_start) / n_chunks
        for chunk_idx, i in enumerate(range(0, len(words), words_per_line)):
            chunk_words = words[i:i + words_per_line]
            c_start = seg_start + chunk_idx * chunk_duration - clip_start
            c_end = c_start + chunk_duration
            lines.append((max(0, c_start), c_end, chunk_words))

    header = _ass_header(caption_font_family, important_size)

    dialogue_lines = []

    if title:
        safe_title = sanitize_ass_text(title)
        dialogue_lines.append(
            f"Dialogue: 1,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,{safe_title}"
        )

    color_i = 0
    for start, end, chunk_words in lines:
        parts = []
        for w in chunk_words:
            safe_w = sanitize_ass_text(w)
            if w.strip(".,!?;:").lower() in STOPWORDS:
                color_tag = rgb_to_ass_inline(255, 255, 255)
                size_tag = f"\\fs{common_size}"
            else:
                r, g, b = palette[color_i % len(palette)]
                color_i += 1
                color_tag = rgb_to_ass_inline(r, g, b)
                size_tag = f"\\fs{important_size}"
            parts.append(f"{{{color_tag}{size_tag}}}{safe_w}")
        text = " ".join(parts)
        dialogue_lines.append(
            f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Default,,0,0,0,,{text}"
        )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(dialogue_lines))
        f.write("\n")


def escape_drawtext(text: str) -> str:
    """Escape special characters so ffmpeg's drawtext filter doesn't choke on them."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\u2019")
        .replace("%", "\\%")
    )


def find_system_font() -> str:
    """Find a bold Latin font that exists on this machine, Linux (GitHub Actions) or Windows."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux (GitHub Actions)
        "C:/Windows/Fonts/arialbd.ttf",  # Windows
        "C:/Windows/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def prepare_local_font() -> str:
    """Copy the system font into the current working directory. FFmpeg on
    Windows chokes on the drive-letter colon (C:) in fontfile= no matter how
    it's escaped or quoted, so a plain relative filename sidesteps it entirely."""
    local_name = "clip_font.ttf"
    if not os.path.exists(local_name):
        shutil.copy(find_system_font(), local_name)
    return local_name


def ffmpeg_path_forward_slashes(path: str) -> str:
    return path.replace("\\", "/")


FONT_PATH = prepare_local_font()

# --- Right-to-left / Urdu-Arabic script caption font support ---
# NOTE: captions are now always transliterated to Latin letters (see
# transliterate_segments_to_latin), so this RTL font path is kept for
# reference / future use but is no longer actively selected.

RTL_LANGUAGES = {"ur", "ar", "fa", "ps"}

# (file path, font family name libass should look it up by)
# NOTE: Noto Nastaliq Urdu is deliberately LAST. With ffmpeg's libass it draws empty
# boxes instead of Urdu letters, while Noto Sans Arabic / Naskh render Urdu correctly.
RTL_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf", "Noto Sans Arabic"),
    ("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf", "Noto Naskh Arabic"),
    ("C:/Windows/Fonts/tahomabd.ttf", "Tahoma"),
    ("C:/Windows/Fonts/tahoma.ttf", "Tahoma"),
    ("C:/Windows/Fonts/Urdu Typesetting.ttf", "Urdu Typesetting"),
    ("C:/Windows/Fonts/arabtype.ttf", "Arabic Typesetting"),
    ("/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Bold.ttf", "Noto Nastaliq Urdu"),
    ("/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf", "Noto Nastaliq Urdu"),
]


def resolve_caption_font(lang_code: str):
    """Pick the right font FILE and FAMILY NAME for the caption language.
    Captions are always transliterated to Latin now, so this always takes
    the default Latin-font path in practice regardless of lang_code - the
    RTL branch is kept here only for reference / future use."""
    fonts_dir = "fonts"
    os.makedirs(fonts_dir, exist_ok=True)

    if lang_code in RTL_LANGUAGES:
        for path, family in RTL_FONT_CANDIDATES:
            if os.path.exists(path):
                local_path = os.path.join(fonts_dir, os.path.basename(path))
                if not os.path.exists(local_path):
                    shutil.copy(path, local_path)
                return fonts_dir, family
        print(
            "WARNING: no Urdu/Arabic-capable font found on this system. "
            "Captions may show broken boxes. On Linux install 'fonts-noto-core'; "
            "on Windows a font like 'Tahoma' is needed."
        )

    # Default: Latin font, already copied by prepare_local_font()
    default_path = find_system_font()
    local_path = os.path.join(fonts_dir, os.path.basename(default_path))
    if not os.path.exists(local_path):
        shutil.copy(default_path, local_path)
    family = "DejaVu Sans" if "DejaVu" in default_path else "Arial"
    return fonts_dir, family


def probe_fit_geometry(video_path: str):
    """Return (fg_w, fg_h): the size the original video has after being fitted
    inside the 1080x1920 frame (same logic as the foreground in cut_vertical_clip)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,sample_aspect_ratio",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    sar = stream.get("sample_aspect_ratio") or "1:1"
    try:
        num, den = (float(x) for x in sar.split(":"))
        sar_value = num / den if num > 0 and den > 0 else 1.0
    except ValueError:
        sar_value = 1.0
    disp_w = max(2, int(w * sar_value) // 2 * 2)
    scale = min(1080 / disp_w, 1920 / h)
    fg_w = max(2, int(disp_w * scale) // 2 * 2)
    fg_h = max(2, int(h * scale) // 2 * 2)
    return fg_w, fg_h


def generate_person_mask_video(
    video_path: str, start: float, duration: float,
    fg_w: int, fg_h: int, mask_path: str,
) -> str:
    """Make a black/white video where white = the person (found with MediaPipe
    selfie segmentation) for one clip. Works on a small copy of the frames for
    speed; the mask is scaled up to the full size later. It covers the same
    time range as the clip, at BEHIND_FPS."""
    import numpy as np
    import mediapipe as mp

    k = MASK_WORK_SIZE / max(fg_w, fg_h)
    mw = max(2, int(fg_w * k) // 2 * 2)
    mh = max(2, int(fg_h * k) // 2 * 2)

    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=1 if fg_w >= fg_h else 0
    )

    decoder = subprocess.Popen(
        [
            "ffmpeg", "-v", "error",
            "-ss", str(start), "-t", str(duration), "-i", video_path,
            "-vf", f"fps={BEHIND_FPS},scale='trunc(iw*sar/2)*2':ih,setsar=1,scale={mw}:{mh}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{mw}x{mh}", "-r", str(BEHIND_FPS),
            "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gray", mask_path,
        ],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    frame_bytes = mw * mh * 3
    smoothed = None
    frames = 0
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(mh, mw, 3)
            mask = segmenter.process(frame).segmentation_mask
            # Sharpen the soft edge a little, then average with the previous
            # frame so the cut-out does not flicker.
            mask = np.clip((mask - 0.35) / 0.3, 0.0, 1.0)
            smoothed = mask if smoothed is None else 0.6 * mask + 0.4 * smoothed
            encoder.stdin.write((smoothed * 255).astype(np.uint8).tobytes())
            frames += 1
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()
        segmenter.close()

    if frames == 0 or encoder.returncode != 0:
        raise RuntimeError("person mask could not be created")
    return mask_path


def cut_vertical_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    ass_path: str = None,
    brand_text: str = None,
    fonts_dir: str = None,
    mask_path: str = None,
    title_ass_path: str = None,
    fg_size: tuple = None,
):
    """Cut a segment into a true 1080x1920 (9:16) clip: the original video
    fitted in the center (no stretching), with a blurred, cropped copy of
    the same video filling the top and bottom. Then burn in the captions
    and an optional small brand watermark."""
    duration = end - start
    if mask_path and fg_size:
        _cut_behind_clip(
            video_path, start, duration, output_path, ass_path, brand_text,
            fonts_dir, mask_path, title_ass_path, fg_size,
        )
        return
    filters = [
        # Fix non-square pixels first, then split into background/foreground
        "[0:v]scale='trunc(iw*sar/2)*2':ih,setsar=1,split=2[bgsrc][fgsrc]",
        # Background: scale to COVER 1080x1920, crop the excess, then blur
        "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=25:5,setsar=1[bg]",
        # Foreground: scale to FIT inside 1080x1920, keep aspect ratio
        "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "setsar=1[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]",
    ]
    current = "[base]"

    if brand_text:
        safe_brand = escape_drawtext(brand_text)
        filters.append(
            f"{current}drawtext=fontfile={FONT_PATH}:text='{safe_brand}':"
            "fontsize=28:fontcolor=white@0.85:borderw=2:bordercolor=black@0.6:"
            "x=w-text_w-30:y=h-60[branded]"
        )
        current = "[branded]"

    if ass_path:
        safe_ass_path = ffmpeg_path_forward_slashes(ass_path)
        if fonts_dir:
            safe_fonts_dir = ffmpeg_path_forward_slashes(fonts_dir)
            filters.append(f"{current}subtitles='{safe_ass_path}':fontsdir='{safe_fonts_dir}'[out]")
        else:
            filters.append(f"{current}subtitles='{safe_ass_path}'[out]")
        current = "[out]"

    filter_complex = ";".join(filters)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-map", current,
            "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ],
        check=True,
    )


def _subtitles_filter(ass_path: str, fonts_dir: str = None) -> str:
    safe_ass_path = ffmpeg_path_forward_slashes(ass_path)
    if fonts_dir:
        safe_fonts_dir = ffmpeg_path_forward_slashes(fonts_dir)
        return f"subtitles='{safe_ass_path}':fontsdir='{safe_fonts_dir}'"
    return f"subtitles='{safe_ass_path}'"


def _cut_behind_clip(
    video_path, start, duration, output_path, ass_path, brand_text,
    fonts_dir, mask_path, title_ass_path, fg_size,
):
    """9:16 clip where the captions sit BEHIND the person:
    blurred background -> captions -> person cut-out on top -> title/brand."""
    fg_w, fg_h = fg_size
    filters = [
        f"[0:v]fps={BEHIND_FPS},scale='trunc(iw*sar/2)*2':ih,setsar=1,split=2[bgsrc][fgsrc]",
        "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=25:5,setsar=1[bg]",
        f"[fgsrc]scale={fg_w}:{fg_h},setsar=1,split=2[fg][fgp]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]",
        # captions are drawn on the scene WITHOUT the person...
        f"[base]{_subtitles_filter(ass_path, fonts_dir)}[txt]",
        # ...then the person (video + mask as transparency) is laid back on top
        f"[1:v]scale={fg_w}:{fg_h},format=gray[m]",
        "[fgp][m]alphamerge[person]",
        "[txt][person]overlay=(W-w)/2:(H-h)/2,setsar=1[comp]",
    ]
    current = "[comp]"

    if title_ass_path:
        filters.append(f"{current}{_subtitles_filter(title_ass_path, fonts_dir)}[titled]")
        current = "[titled]"

    if brand_text:
        safe_brand = escape_drawtext(brand_text)
        filters.append(
            f"{current}drawtext=fontfile={FONT_PATH}:text='{safe_brand}':"
            "fontsize=28:fontcolor=white@0.85:borderw=2:bordercolor=black@0.6:"
            "x=w-text_w-30:y=h-60[branded]"
        )
        current = "[branded]"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration), "-i", video_path,
            "-i", mask_path,
            "-filter_complex", ";".join(filters),
            "-map", current,
            "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-shortest",
            output_path,
        ],
        check=True,
    )


def render_captioned_clip(
    video_path, start, end, clip_path, ass_path, segments, title,
    font_family, fonts_dir, colors, important_size, common_size,
    brand_text, caption_style,
):
    """Build captions and render one clip. In the 'behind' look the person is
    cut out first; if that fails for any reason the clip is still made with
    the classic bottom captions, so a job never fails because of the effect."""
    style = caption_style
    mask_path = None
    fg_size = None
    if style == "behind":
        try:
            fg_size = probe_fit_geometry(video_path)
            mask_path = str(Path(clip_path).with_suffix(".mask.mkv"))
            print("  finding the person in each frame...")
            generate_person_mask_video(
                video_path, start, end - start, fg_size[0], fg_size[1], mask_path
            )
        except Exception as e:  # missing mediapipe, odd video, etc.
            print(f"WARNING: behind-the-person captions unavailable ({e}). Using classic captions.")
            style = "classic"
            mask_path = None

    front_ass_path = None
    if style == "behind":
        front_ass_path = str(Path(ass_path).with_suffix(".front.ass"))
        build_behind_ass(
            segments, start, end, ass_path, front_ass_path,
            title=title,
            caption_font_family=font_family,
            colors=colors,
            important_size=important_size,
            common_size=common_size,
        )
    else:
        # classic look; if the user asked for 'behind' but it failed, use the
        # normal default sizes instead of the big behind-look sizes
        if caption_style == "behind":
            important_size, common_size = DEFAULT_IMPORTANT_SIZE, DEFAULT_COMMON_SIZE
        build_captions_ass(
            segments, start, end, ass_path,
            title=title,
            caption_font_family=font_family,
            colors=colors,
            important_size=important_size,
            common_size=common_size,
        )
    try:
        cut_vertical_clip(
            video_path, start, end, clip_path,
            ass_path=ass_path,
            brand_text=brand_text,
            fonts_dir=fonts_dir,
            mask_path=mask_path,
            title_ass_path=front_ass_path,
            fg_size=fg_size,
        )
    finally:
        if mask_path and os.path.exists(mask_path):
            os.remove(mask_path)  # keep the output folder small


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Direct URL to the video file (e.g. an uploaded video link) or a local file path")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--brand-text", default=None, help="Optional watermark text shown on every clip")
    parser.add_argument(
        "--caption-only", default=None,
        help="Language name or code ('en', 'english', 'urdu', ...). If given, "
             "skips picking multiple clips: captions the WHOLE video in this "
             "language, adds one title label, and converts to 9:16 - no clip splitting."
    )
    parser.add_argument(
        "--clip-language", default=None,
        help="Language name or code ('ur', 'urdu', 'hi', ...) for the captions of the "
             "short clips (normal mode). Choose the language spoken in the video. "
             "If not given, the speech is translated to English captions."
    )
    parser.add_argument(
        "--caption-colors", default=None,
        help="Comma-separated hex colors for highlighted caption words, "
             "e.g. 'FFEB3B,00E5FF,FF4081'. They are used in turn, word by word. "
             "Default: yellow, cyan, pink, orange, green, red, purple."
    )
    parser.add_argument(
        "--important-size", type=int, default=None,
        help=f"Text size of important (colored) caption words. Default {DEFAULT_IMPORTANT_SIZE} "
             f"({BEHIND_IMPORTANT_SIZE} in the 'behind' look), allowed {MIN_TEXT_SIZE}-{MAX_TEXT_SIZE}."
    )
    parser.add_argument(
        "--common-size", type=int, default=None,
        help=f"Text size of common (white) caption words like 'the', 'and'. Default {DEFAULT_COMMON_SIZE} "
             f"({BEHIND_COMMON_SIZE} in the 'behind' look), allowed {MIN_TEXT_SIZE}-{MAX_TEXT_SIZE}."
    )
    parser.add_argument(
        "--caption-style", default="classic", choices=CAPTION_STYLES,
        help="'classic' = captions at the bottom (default). 'behind' = important words big in the "
             "middle, behind the person; the rest of the caption at the bottom."
    )
    args = parser.parse_args()

    groq_key = os.environ["GROQ_API_KEY"]
    gem_key = os.environ["GEM_API_KEY"]

    caption_colors = parse_caption_colors(args.caption_colors)
    caption_style = args.caption_style if args.caption_style in CAPTION_STYLES else "classic"
    is_behind = caption_style == "behind"
    important_size = clamp_text_size(
        args.important_size, BEHIND_IMPORTANT_SIZE if is_behind else DEFAULT_IMPORTANT_SIZE)
    common_size = clamp_text_size(
        args.common_size, BEHIND_COMMON_SIZE if is_behind else DEFAULT_COMMON_SIZE)
    print(f"Caption style: {caption_style} - colors: {len(caption_colors)}, "
          f"important size: {important_size}, common size: {common_size}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"project_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")

    print("Downloading video...")
    video_path = str(out_dir / "source.mp4")
    download_video(args.video, video_path)

    print("Extracting audio...")
    audio_path = str(out_dir / "audio.mp3")
    extract_audio(video_path, audio_path)

    if args.caption_only:
        # --- Caption-only mode: whole video, one label, no clip splitting ---
        lang_code = resolve_language_code(args.caption_only)
        print(f"Caption-only mode - language: {lang_code}")

        if lang_code == "en":
            transcript = transcribe_english(audio_path, groq_key)
            caption_segments = transcript.get("segments", [])
        else:
            transcript = transcribe_in_language(audio_path, groq_key, lang_code)
            with open(out_dir / "transcript_native.json", "w") as f:
                json.dump(transcript, f, indent=2)
            print("Transliterating captions to Latin/English letters...")
            caption_segments = transliterate_segments_to_latin(transcript.get("segments", []), gem_key)

        with open(out_dir / "transcript.json", "w") as f:
            json.dump(transcript, f, indent=2)

        print("Labeling with Gemini...")
        label = pick_label(transcript, gem_key)
        with open(out_dir / "label.json", "w") as f:
            json.dump({"title": label}, f, indent=2)

        duration = get_video_duration(video_path)

        fonts_dir, font_family = resolve_caption_font("en")  # always Latin font now
        print(f"Caption font: {font_family}")

        clip_path = str(out_dir / "captioned.mp4")
        ass_path = str(out_dir / "captions.ass")
        render_captioned_clip(
            video_path, 0, duration, clip_path, ass_path, caption_segments, label,
            font_family, fonts_dir, caption_colors, important_size, common_size,
            args.brand_text, caption_style,
        )
        print(f"  saved {clip_path} - {label}")

    else:
        # --- Default mode: pick multiple short clips ---
        clip_lang = resolve_language_code(args.clip_language) if args.clip_language else "en"
        print(f"Clip mode - caption language: {clip_lang}")

        print("Transcribing with Groq...")
        if clip_lang == "en":
            transcript = transcribe_english(audio_path, groq_key)
            caption_segments = transcript.get("segments", [])
        else:
            transcript = transcribe_in_language(audio_path, groq_key, clip_lang)
            with open(out_dir / "transcript_native.json", "w") as f:
                json.dump(transcript, f, indent=2)
            print("Transliterating captions to Latin/English letters...")
            caption_segments = transliterate_segments_to_latin(transcript.get("segments", []), gem_key)

        clip_fonts_dir, clip_font_family = resolve_caption_font("en")  # always Latin font now
        print(f"Caption font: {clip_font_family}")

        with open(out_dir / "transcript.json", "w") as f:
            json.dump(transcript, f, indent=2)

        print("Picking clips with Gemini...")
        clips = pick_clips(transcript, gem_key, args.min_clips, args.max_clips)
        with open(out_dir / "clips_metadata.json", "w") as f:
            json.dump(clips, f, indent=2)

        print(f"Cutting {len(clips)} clips...")
        for i, clip in enumerate(clips, start=1):
            clip_path = str(out_dir / f"clip_{i}.mp4")
            ass_path = str(out_dir / f"clip_{i}.ass")
            render_captioned_clip(
                video_path, clip["start"], clip["end"], clip_path, ass_path,
                caption_segments, clip.get("title"),
                clip_font_family, clip_fonts_dir, caption_colors, important_size, common_size,
                args.brand_text, caption_style,
            )
            print(f"  saved {clip_path} - {clip.get('title', '')}")

    print("Done.")


if __name__ == "__main__":
    main()
