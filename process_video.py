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
BEHIND_COMMON_SIZE = 84       # size of the normal caption line at the bottom in the "behind" look
BOTTOM_WORDS_PER_LINE = 5
# What the bottom line shows in the "behind" look:
#   "full"      = the whole phrase (important words colored), so it reads naturally
#   "remaining" = only the words that are NOT shown big behind the person
BEHIND_BOTTOM_MODE = "remaining"
# Layout of the "behind" look:
#   "all_behind" = EVERY word is shown behind the person (no bottom line at all)
#   "three_zone" = older layout: key words rotate behind/top, small words at the bottom
BEHIND_LAYOUT = "stack"
#   "stack"      = tight rows (3 words in a row, or 2 big + 1 biggest under them...),
#                  upper rows behind the person, lower rows in front. Words pop in one by one.
#   "poster"     = mixed layout: small lead-in at the TOP, big key word BEHIND the person,
#                  medium word + huge bold word IN FRONT, small ending at the BOTTOM.
#                  Words pop in one by one and stay until the phrase ends.
POSTER_PHRASE_WORDS = 8        # words per on-screen phrase
POSTER_TOP_RATIO = 0.50        # text sizes, as a share of the "important" size
POSTER_MID_RATIO = 0.62
POSTER_FRONT_RATIO = 1.10
POSTER_BOTTOM_RATIO = 0.55
# which zone each key word goes to; the pattern changes from phrase to phrase for variety
POSTER_PATTERNS = {
    1: [("behind",), ("front",), ("behind",)],
    2: [("behind", "front"), ("front", "behind"), ("behind", "mid")],
    3: [("behind", "mid", "front"), ("mid", "behind", "front"), ("behind", "front", "mid")],
}
BEHIND_GROUP_MAX_WORDS = 3     # words shown together on one line behind the person
BEHIND_GROUP_MAX_CHARS = 14    # ...but never more letters than this (long words get their own line)
BEHIND_SMALL_RATIO = 0.65      # size of small connecting words (the, is, in) next to a big word
BEHIND_TEXT_Y = 900           # fallback vertical centre when no person/head is found
BEHIND_HEAD_OVERLAP = 0.35    # how much of the big word's height tucks behind the head (0 = none, 0.5 = half)
BEHIND_MIN_Y = 240            # keep the big word below the title area
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


# Rows for the "stack" look. Each pattern is a list of rows:
# (words in the row, text size as a share of the "important" size, layer)
STACK_GAP = .88          # distance between rows as a share of the text size (smaller = tighter)
STACK_ATTACH_FRONT = False  # True = front rows sit right under the behind rows (one tight block);
                           # False = front rows go down to a fixed height lower on the body
                           # (the fixed "mid_y" zone, ~62% down - below center, not at the very bottom)
STACK_TOP_FLOOR = 150     # a row never starts higher than this (keeps clear of the title)
STACK_PATTERNS = {
    "row3":        [(3, 0.80, "behind")],                                    # one row of 3 words
    "two_one":     [(2, 0.95, "behind"), (1, 1.30, "front")],                # 2 big words, then 1 BIGGEST
    "three_two":   [(3, 0.70, "behind"), (2, 1.00, "front")],
    "one_two":     [(1, 1.10, "behind"), (2, 0.90, "front")],
    "two_one_two": [(2, 0.80, "behind"), (1, 1.30, "front"), (2, 0.65, "front")],
    "ladder":      [(1, 0.55, "behind"), (1, 1.15, "behind"), (1, 0.95, "front")],
    "one":         [(1, 1.20, "behind")],
    "two":         [(2, 1.00, "behind")],
    "one_front":   [(1, 1.30, "front")],
    # Poster-style: a small row of 3 words up near the top (behind the person),
    # then ONE big key word tucked just above the head/center (behind), then the
    # remaining 2 words land below center - a clean, professional poster layout.
    "poster_3_1_2": [(3, 0.60, "behind"), (1, 1.45, "behind"), (2, 0.85, "front")],
}


def build_stack_ass(
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
    head_y_at=None,
    zones: dict = None,
):
    """'Stack' look: the speech is cut into small cards, and every card is a few
    TIGHT rows of text (for example one row of 3 words, or 2 big words and under
    them 1 biggest word). Rows near the head are drawn BEHIND the person, rows
    lower down are drawn in FRONT. Words pop in one by one inside their row.
    The pattern changes from card to card, picked so the biggest rows get the
    most important words."""
    palette = colors or WORD_COLORS
    zones = zones or {"mid_y": 1100}
    header = _ass_header(caption_font_family, important_size)
    behind_lines, front_lines = [], []

    if title:
        front_lines.append(
            f"Dialogue: 2,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,"
            f"{sanitize_ass_text(title)}"
        )

    def clean(w):
        return w.strip(".,;:\"'")

    key_i = 0
    prev_pattern = None
    for seg in segments:
        if not (seg["end"] > clip_start and seg["start"] < clip_end):
            continue
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = [w for w in seg["text"].strip().split() if clean(w)]
        if not words:
            continue
        seg_t0 = max(0.0, seg_start - clip_start)
        word_dur = (seg_end - seg_start) / len(words)

        i = 0
        while i < len(words):
            remaining = len(words) - i

            # pick the pattern whose big rows land on the most important words
            best = None
            for name, rows in STACK_PATTERNS.items():
                total = sum(n for n, _, _ in rows)
                if total > remaining:
                    continue
                score, pos = 0.15 * total, i
                for n, ratio, _layer in rows:
                    for w in words[pos:pos + n]:
                        score += ratio * (-0.5 if _is_common_word(w) else 1.0)
                    pos += n
                if name == prev_pattern:
                    score -= 1.5  # keep the look changing from card to card
                if best is None or score > best[0]:
                    best = (score, name, rows, total)
            _, prev_pattern, rows, total = best

            card_start = seg_t0 + i * word_dur
            card_end = seg_t0 + (i + total) * word_dur

            built, pos = [], i
            for n, ratio, layer in rows:
                raw = words[pos:pos + n]
                shown = [clean(w) for w in raw]
                chars = sum(len(x) for x in shown) + (n - 1)
                fs = max(40, min(int(ratio * important_size), int(960 / (chars * 0.66))))
                built.append({"raw": raw, "words": shown, "fs": fs, "layer": layer, "first": pos})
                pos += n

            # --- vertical positions: rows are packed close together ---
            b_rows = [r for r in built if r["layer"] == "behind"]
            f_rows = [r for r in built if r["layer"] == "front"]
            demote = False
            if b_rows:
                last = b_rows[-1]
                by, has_head = BEHIND_TEXT_Y, False
                if head_y_at is not None:
                    hy = head_y_at(card_start, card_end)
                    if hy is not None:
                        by, has_head = int(hy - last["fs"] * (0.5 - BEHIND_HEAD_OVERLAP)), True
                last["y"] = by
                for k in range(len(b_rows) - 2, -1, -1):
                    cur, nxt = b_rows[k], b_rows[k + 1]
                    cur["y"] = nxt["y"] - int(STACK_GAP * (cur["fs"] + nxt["fs"]) / 2)
                top_edge = b_rows[0]["y"] - b_rows[0]["fs"] // 2
                if top_edge < STACK_TOP_FLOOR:
                    for r in b_rows:
                        r["y"] += STACK_TOP_FLOOR - top_edge
                    # no room above the head: draw these rows in front so they stay readable
                    demote = has_head
            if f_rows:
                min_top = (b_rows[-1]["y"] + b_rows[-1]["fs"] // 2) if b_rows else 0
                if b_rows and STACK_ATTACH_FRONT:
                    f_rows[0]["y"] = b_rows[-1]["y"] + int(
                        STACK_GAP * (b_rows[-1]["fs"] + f_rows[0]["fs"]) / 2)
                else:
                    f_rows[0]["y"] = max(zones["mid_y"], min_top + f_rows[0]["fs"] // 2 + 10)
                for k in range(1, len(f_rows)):
                    prv, cur = f_rows[k - 1], f_rows[k]
                    cur["y"] = prv["y"] + int(STACK_GAP * (prv["fs"] + cur["fs"]) / 2)
                excess = f_rows[-1]["y"] + f_rows[-1]["fs"] // 2 - 1780
                if excess > 0:
                    for r in f_rows:
                        r["y"] -= excess

            # --- events: words appear one by one inside each row ---
            biggest = max(r["fs"] for r in built)
            for r in built:
                is_behind = r["layer"] == "behind" and not demote
                n = len(r["words"])
                word_colors = []
                for w in r["raw"]:
                    if _is_common_word(w):
                        word_colors.append((255, 255, 255))
                    else:
                        word_colors.append(palette[key_i % len(palette)])
                        key_i += 1
                layer = 1 if (r["layer"] == "front" and r["fs"] == biggest) else 0
                bord = 4 if is_behind else 5
                for j in range(n):
                    e0 = seg_t0 + (r["first"] + j) * word_dur
                    e1 = seg_t0 + (r["first"] + j + 1) * word_dur if j < n - 1 else card_end
                    if e1 - e0 < 0.04:
                        continue
                    parts = []
                    for k, wtxt in enumerate(r["words"]):
                        safe = sanitize_ass_text(wtxt)
                        if k > j:
                            parts.append("{\\alpha&HFF&}" + safe)  # not spoken yet: invisible, keeps its place
                        else:
                            cr, cg, cb = word_colors[k]
                            if k == j:
                                pop = "\\fscx78\\fscy78\\t(0,130,\\fscx106\\fscy106)\\t(130,200,\\fscx100\\fscy100)"
                            else:
                                pop = "\\fscx100\\fscy100"
                            parts.append("{\\alpha&H00&" + rgb_to_ass_inline(cr, cg, cb) + pop + "}" + safe)
                    fade_in = 40 if j == 0 else 0
                    fade_out = 60 if j == n - 1 else 0
                    fad = f"\\fad({fade_in},{fade_out})" if (fade_in or fade_out) else ""
                    tags = f"\\an5\\pos(540,{r['y']})\\fs{r['fs']}\\bord{bord}{fad}"
                    style = "Behind" if is_behind else "Default"
                    line = (f"Dialogue: {layer},{format_ass_time(e0)},{format_ass_time(e1)},{style},,0,0,0,,"
                            + "{" + tags + "}" + " ".join(parts))
                    (behind_lines if is_behind else front_lines).append(line)
            i += total

    for path, lines in ((behind_ass_path, behind_lines), (front_ass_path, front_lines)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(lines))
            f.write("\n")


def build_poster_ass(
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
    head_y_at=None,
    zones: dict = None,
):
    """'Poster' look: each phrase is built word by word from pieces placed in zones.
      top    - small lead-in words (white)
      behind - a big key word just above the head, UNDER the person cut-out
      mid    - a medium white italic key word, in front
      front  - a huge bold key word, in front
      bottom - small ending words (white)
    Pieces pop in when they are spoken and stay until the phrase ends.
      behind_ass_path - only the 'behind' pieces
      front_ass_path  - the title + all other pieces (drawn on top)."""
    palette = colors or WORD_COLORS
    zones = zones or {"top_y": 300, "mid_y": 1100, "bottom_y": 1500}
    header = _ass_header(caption_font_family, important_size)
    behind_lines, front_lines = [], []

    if title:
        front_lines.append(
            f"Dialogue: 2,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,"
            f"{sanitize_ass_text(title)}"
        )

    def clean(w):
        return w.strip(".,!?;:\"'")

    base_size = {
        "top": int(important_size * POSTER_TOP_RATIO),
        "mid": int(important_size * POSTER_MID_RATIO),
        "front": int(important_size * POSTER_FRONT_RATIO),
        "bottom": int(important_size * POSTER_BOTTOM_RATIO),
        "behind": important_size,
    }

    # 1) cut the speech into phrases
    phrases = []  # (start, end, words) in clip time
    for seg in segments:
        if not (seg["end"] > clip_start and seg["start"] < clip_end):
            continue
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = [w for w in seg["text"].strip().split() if clean(w)]
        if not words:
            continue
        n_chunks = max(1, -(-len(words) // POSTER_PHRASE_WORDS))
        chunk_dur = (seg_end - seg_start) / n_chunks
        for ci, i in enumerate(range(0, len(words), POSTER_PHRASE_WORDS)):
            c0 = max(0.0, seg_start + ci * chunk_dur - clip_start)
            phrases.append((c0, c0 + chunk_dur, words[i:i + POSTER_PHRASE_WORDS]))

    key_i = 0
    for pi, (start, end, words) in enumerate(phrases):
        word_dur = (end - start) / len(words)
        keys = [k for k, w in enumerate(words) if not _is_common_word(w)]

        # 2) choose up to 3 "hero" key words; words before them = top, after them = bottom
        if keys:
            n_h = min(3, len(keys))
            hs = (len(keys) - n_h) // 2
            heroes = keys[hs:hs + n_h]
        else:
            heroes = []
        pattern = POSTER_PATTERNS[len(heroes)][pi % 3] if heroes else ()

        pieces = []  # dicts: zone, first (word index), text
        if heroes and heroes[0] > 0:
            pieces.append({"zone": "top", "first": 0, "words": words[:heroes[0]], "prefix": []})
        for j, hk in enumerate(heroes):
            prefix = words[heroes[j - 1] + 1:hk] if j > 0 else []
            first = heroes[j - 1] + 1 if (j > 0 and prefix) else hk
            pieces.append({"zone": pattern[j], "first": first, "words": [words[hk]], "prefix": prefix})
        tail_from = heroes[-1] + 1 if heroes else 0
        if tail_from < len(words):
            pieces.append({"zone": "bottom", "first": tail_from, "words": words[tail_from:], "prefix": []})

        has_mid = any(pc["zone"] == "mid" for pc in pieces)
        mid_y = zones["mid_y"]
        front_y = mid_y + (90 if has_mid else 45)
        bottom_y = min(1700, max(zones["bottom_y"], front_y + 150))

        def geom(pc):
            zone = pc["zone"]
            t0 = start if pc["first"] == 0 or pc is pieces[0] else start + pc["first"] * word_dur
            is_hero = zone in ("behind", "mid", "front")
            main = " ".join(clean(w) for w in pc["words"]) if is_hero else " ".join(pc["words"])
            prefix = " ".join(pc["prefix"])
            n_chars = len(main) + (len(prefix) * 0.7 + 1 if prefix else 0)
            fs = max(36, min(base_size[zone], int(940 / (max(1, n_chars) * 0.68))))
            small_fs = max(32, int(fs * 0.6))
            return zone, t0, is_hero, main, prefix, fs, small_fs

        # Place the 'behind' word first: the top piece is positioned relative to it.
        behind_y, behind_in_front, top_y_eff = None, False, zones["top_y"]
        for pc in pieces:
            if pc["zone"] != "behind":
                continue
            _, bt0, _, _, _, bfs, _ = geom(pc)
            by, has_head = BEHIND_TEXT_Y, False
            if head_y_at is not None:
                hy = head_y_at(bt0, end)
                if hy is not None:
                    by, has_head = int(hy - bfs * (0.5 - BEHIND_HEAD_OVERLAP)), True
            min_y = 230 + bfs // 2
            if has_head and by < min_y:
                # no room above the head: draw the word in front so it stays readable
                behind_in_front = True
            behind_y = max(min_y, min(1700, by))
            top_y_eff = max(200, min(zones["top_y"], behind_y - bfs // 2 - int(base_size["top"] * 0.5) - 12))

        for pc in pieces:
            zone, t0, is_hero, main, prefix, fs, small_fs = geom(pc)
            t1 = end
            if t1 - t0 < 0.05:
                continue

            if zone in ("behind", "front"):
                r, g, b = palette[key_i % len(palette)]
                key_i += 1
                color = rgb_to_ass_inline(r, g, b)
            else:
                color = rgb_to_ass_inline(255, 255, 255)

            body = ""
            if prefix:
                body += f"{{\\fs{small_fs}{rgb_to_ass_inline(255, 255, 255)}}}{sanitize_ass_text(prefix)} "
            body += f"{{\\fs{fs}{color}}}{sanitize_ass_text(main)}"

            if zone == "behind":
                y = behind_y
                anim = (f"\\an5\\pos(540,{y})\\fad(70,60)"
                        f"\\fscx80\\fscy80\\t(0,110,\\fscx108\\fscy108)\\t(110,190,\\fscx100\\fscy100)")
                if behind_in_front:
                    front_lines.append(
                        f"Dialogue: 0,{format_ass_time(t0)},{format_ass_time(t1)},Default,,0,0,0,,{{{anim}\\bord5}}{body}")
                else:
                    behind_lines.append(
                        f"Dialogue: 0,{format_ass_time(t0)},{format_ass_time(t1)},Behind,,0,0,0,,{{{anim}}}{body}")
                continue

            if zone == "top":
                y = top_y_eff
                anim = f"\\an5\\move(540,{y - 50},540,{y},0,170)\\fad(90,60)\\bord4"
                layer = 0
            elif zone == "mid":
                y = mid_y
                anim = f"\\an5\\move(430,{y},540,{y},0,190)\\fad(80,60)\\i1\\bord4"
                layer = 0
            elif zone == "front":
                y = front_y
                anim = (f"\\an5\\pos(540,{y})\\fad(60,60)\\bord6"
                        f"\\fscx60\\fscy60\\t(0,120,\\fscx112\\fscy112)\\t(120,200,\\fscx100\\fscy100)")
                layer = 1
            else:  # bottom
                y = bottom_y
                anim = f"\\an5\\move(540,{y + 40},540,{y},0,160)\\fad(80,60)\\bord4"
                layer = 0
            front_lines.append(
                f"Dialogue: {layer},{format_ass_time(t0)},{format_ass_time(t1)},Default,,0,0,0,,{{{anim}}}{body}")

    for path, lines in ((behind_ass_path, behind_lines), (front_ass_path, front_lines)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(lines))
            f.write("\n")


def build_all_behind_ass(
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
    head_y_at=None,
):
    """Every caption word is shown BEHIND the person - there is no bottom line.
    Short groups of words (up to BEHIND_GROUP_MAX_WORDS) are shown one group at a
    time as a single centered line just above the head: important words big and
    colored, small connecting words (the, is, in...) smaller and white.
      behind_ass_path - the words (drawn under the person cut-out)
      front_ass_path  - only the title at the top (drawn on top)."""
    palette = colors or WORD_COLORS
    header = _ass_header(caption_font_family, important_size)
    behind_lines, front_lines = [], []

    if title:
        front_lines.append(
            f"Dialogue: 1,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,"
            f"{sanitize_ass_text(title)}"
        )

    key_i = 0
    for seg in segments:
        if not (seg["end"] > clip_start and seg["start"] < clip_end):
            continue
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = seg["text"].strip().split()
        words = [w for w in words if w.strip(".,!?;:\"'")]
        if not words:
            continue
        seg_dur = (seg_end - seg_start)
        word_dur = seg_dur / len(words)
        seg_t0 = max(0.0, seg_start - clip_start)

        # cut the words into short groups that fit on one line
        groups, cur, cur_chars = [], [], 0.0
        for k, w in enumerate(words):
            clean_len = len(w.strip(".,!?;:\"'"))
            weight = clean_len * (BEHIND_SMALL_RATIO if _is_common_word(w) else 1.0)
            if cur and (len(cur) >= BEHIND_GROUP_MAX_WORDS or cur_chars + weight > BEHIND_GROUP_MAX_CHARS):
                groups.append(cur)
                cur, cur_chars = [], 0.0
            cur.append(k)
            cur_chars += weight
        if cur:
            groups.append(cur)

        for gi, idxs in enumerate(groups):
            t0 = seg_t0 if gi == 0 else seg_t0 + idxs[0] * word_dur
            t1 = seg_t0 + groups[gi + 1][0] * word_dur if gi + 1 < len(groups) else seg_t0 + seg_dur
            if t1 - t0 < 0.05:
                continue

            has_key = any(not _is_common_word(words[k]) for k in idxs)
            ratio_small = BEHIND_SMALL_RATIO if has_key else 0.85
            # size so the whole line fits inside ~940px
            units = sum(
                len(words[k].strip(".,!?;:\"'")) * (ratio_small if _is_common_word(words[k]) else 1.0)
                for k in idxs
            ) + 0.4 * (len(idxs) - 1)
            fs = max(56, min(important_size, int(940 / (units * 0.78))))
            small_fs = max(40, int(fs * ratio_small))

            parts = []
            for k in idxs:
                clean = sanitize_ass_text(words[k].strip(".,!?;:\"'").upper())
                if _is_common_word(words[k]):
                    parts.append(f"{{\\fs{small_fs}{rgb_to_ass_inline(255, 255, 255)}}}{clean}")
                else:
                    r, g, b = palette[key_i % len(palette)]
                    key_i += 1
                    parts.append(f"{{\\fs{fs}{rgb_to_ass_inline(r, g, b)}}}{clean}")

            text_y = BEHIND_TEXT_Y
            if head_y_at is not None:
                head_y = head_y_at(t0, t1)
                if head_y is not None:
                    text_y = int(head_y - fs * (0.5 - BEHIND_HEAD_OVERLAP))
                    text_y = max(BEHIND_MIN_Y + fs // 2, min(1700, text_y))

            behind_lines.append(
                f"Dialogue: 0,{format_ass_time(t0)},{format_ass_time(t1)},Behind,,0,0,0,,"
                f"{{\\an5\\pos(540,{text_y})\\fscx85\\fscy85\\t(0,130,\\fscx100\\fscy100)}}"
                + " ".join(parts)
            )

    for path, lines in ((behind_ass_path, behind_lines), (front_ass_path, front_lines)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(lines))
            f.write("\n")


def build_behind_ass(
    segments: list,
    clip_start: float,
    clip_end: float,
    behind_ass_path: str,
    front_ass_path: str,
    caption_font_family: str = "DejaVu Sans",
    colors: list = None,
    important_size: int = BEHIND_IMPORTANT_SIZE,
    common_size: int = BEHIND_COMMON_SIZE,
    head_y_at=None,
    top_y: int = 420,
    bottom_y: int = 1500,
):
    """'Animated 3-zone' look, written as TWO subtitle files (no title, no
    repeated words - every word is shown exactly once):

      behind_ass_path - key words, BIG, just above the head, drawn UNDER the
                        person cut-out (pop-in animation).
      front_ass_path  - key words at the TOP (slide-down + fade) and the small
                        connecting words at the BOTTOM (pop-in), drawn on top.

    Each caption line is cut into short pieces (one key word, or a run of
    common words). One piece is on screen at a time, and the key words rotate
    between the behind and top positions."""
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
    key_i = 0
    top_size = max(60, int(important_size * 0.78))

    for start, end, words in chunks:
        word_dur = (end - start) / len(words)

        # cut the line into pieces: one key word, or a run of common words
        pieces = []  # [kind, [words], first_index]
        for k, w in enumerate(words):
            kind = "common" if _is_common_word(w) else "key"
            if kind == "common" and pieces and pieces[-1][0] == "common":
                pieces[-1][1].append(w)
            else:
                pieces.append([kind, [w], k])

        for n, (kind, piece_words, first) in enumerate(pieces):
            t0 = start if n == 0 else start + first * word_dur
            t1 = start + pieces[n + 1][2] * word_dur if n + 1 < len(pieces) else end
            if t1 - t0 < 0.05:
                continue
            at, to = format_ass_time(t0), format_ass_time(t1)

            if kind == "common":
                text = sanitize_ass_text(" ".join(piece_words))
                front_lines.append(
                    f"Dialogue: 0,{at},{to},Default,,0,0,0,,"
                    f"{{\\an5\\pos(540,{bottom_y})\\fs{common_size}\\bord5\\fad(70,50)"
                    f"\\fscx80\\fscy80\\t(0,150,\\fscx100\\fscy100)}}{text}"
                )
                continue

            clean = piece_words[0].strip(".,!?;:\"'")
            if not clean:
                continue
            r, g, b = palette[key_i % len(palette)]
            zone = ("behind", "behind", "top")[key_i % 3]
            key_i += 1
            word = sanitize_ass_text(clean.upper())
            color = rgb_to_ass_inline(r, g, b)

            if zone == "top":
                fs = max(56, min(top_size, int(940 / (len(word) * 0.78))))
                front_lines.append(
                    f"Dialogue: 0,{at},{to},Default,,0,0,0,,"
                    f"{{\\an5\\move(540,{top_y - 70},540,{top_y},0,180)\\fad(90,60)"
                    f"\\fs{fs}\\bord6{color}}}{word}"
                )
            else:
                fs = max(60, min(important_size, int(940 / (len(word) * 0.78))))
                text_y = BEHIND_TEXT_Y
                if head_y_at is not None:
                    head_y = head_y_at(t0, t1)
                    if head_y is not None:
                        text_y = int(head_y - fs * (0.5 - BEHIND_HEAD_OVERLAP))
                        text_y = max(BEHIND_MIN_Y + fs // 2, min(1700, text_y))
                behind_lines.append(
                    f"Dialogue: 0,{at},{to},Behind,,0,0,0,,"
                    f"{{\\an5\\pos(540,{text_y})\\fs{fs}{color}"
                    f"\\fscx80\\fscy80\\t(0,110,\\fscx108\\fscy108)\\t(110,190,\\fscx100\\fscy100)}}{word}"
                )

    for path, lines in ((behind_ass_path, behind_lines), (front_ass_path, front_lines)):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(lines))
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


# A bold, chunky poster-style display font (like the "Anton" look), used only
# for the "behind" caption style's big key words so they look like a designed
# poster instead of plain bold text. Downloaded once from Google's public
# font repo; if that fails for any reason (no internet, blocked), the normal
# bold Latin font is used instead and nothing breaks.
DISPLAY_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
DISPLAY_FONT_FAMILY = "Anton"
DISPLAY_FONT_LOCAL = "fonts/Anton-Regular.ttf"


def ensure_display_font():
    """Try to download the poster-style display font into the fonts/ folder
    (the same folder libass already searches via fontsdir). Returns the font
    family name to use, or None if it could not be obtained."""
    os.makedirs("fonts", exist_ok=True)
    if os.path.exists(DISPLAY_FONT_LOCAL) and os.path.getsize(DISPLAY_FONT_LOCAL) > 0:
        return DISPLAY_FONT_FAMILY
    try:
        response = requests.get(DISPLAY_FONT_URL, timeout=20)
        response.raise_for_status()
        with open(DISPLAY_FONT_LOCAL, "wb") as f:
            f.write(response.content)
        return DISPLAY_FONT_FAMILY
    except Exception as e:
        print(f"WARNING: could not download the poster display font ({e}); using the default font instead.")
        return None


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
):
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
    head_tops = []  # per frame: top of the person as a fraction (0-1) of the video height, or None
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
            # first row where enough of the width is "person" = top of the head
            rows = np.where((smoothed > 0.5).sum(axis=1) >= max(3, int(mw * 0.04)))[0]
            head_tops.append(float(rows[0]) / mh if len(rows) else None)
            frames += 1
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()
        segmenter.close()

    if frames == 0 or encoder.returncode != 0:
        raise RuntimeError("person mask could not be created")
    return mask_path, head_tops


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
    head_y_at = None
    if style == "behind":
        try:
            fg_size = probe_fit_geometry(video_path)
            mask_path = str(Path(clip_path).with_suffix(".mask.mkv"))
            print("  finding the person in each frame...")
            _, head_tops = generate_person_mask_video(
                video_path, start, end - start, fg_size[0], fg_size[1], mask_path
            )
            fg_top = (1920 - fg_size[1]) / 2.0

            def head_y_at(t0, t1):
                """Median top-of-head (in frame pixels) between two clip times."""
                i0 = max(0, int(t0 * BEHIND_FPS))
                i1 = min(len(head_tops), max(i0 + 1, int(t1 * BEHIND_FPS)))
                vals = sorted(v for v in head_tops[i0:i1] if v is not None)
                if not vals:
                    vals = sorted(v for v in head_tops if v is not None)
                if not vals:
                    return None
                return fg_top + vals[len(vals) // 2] * fg_size[1]
        except Exception as e:  # missing mediapipe, odd video, etc.
            print(f"WARNING: behind-the-person captions unavailable ({e}). Using classic captions.")
            style = "classic"
            mask_path = None

    front_ass_path = None
    if style == "behind":
        front_ass_path = str(Path(ass_path).with_suffix(".front.ass"))
        fg_top = (1920 - fg_size[1]) / 2.0
        poster_font_family = ensure_display_font() or font_family
        print(f"  poster caption font: {poster_font_family}")
        if BEHIND_LAYOUT == "stack":
            build_stack_ass(
                segments, start, end, ass_path, front_ass_path,
                title=title,
                caption_font_family=poster_font_family,
                colors=colors,
                important_size=important_size,
                head_y_at=head_y_at,
                zones={"mid_y": int(fg_top + 0.62 * fg_size[1])},
            )
        elif BEHIND_LAYOUT == "poster":
            build_poster_ass(
                segments, start, end, ass_path, front_ass_path,
                title=title,
                caption_font_family=poster_font_family,
                colors=colors,
                important_size=important_size,
                head_y_at=head_y_at,
                zones={
                    "top_y": int(max(230, fg_top + 0.07 * fg_size[1])),
                    "mid_y": int(fg_top + 0.62 * fg_size[1]),
                    "bottom_y": int(fg_top + 0.86 * fg_size[1]),
                },
            )
        elif BEHIND_LAYOUT == "all_behind":
            build_all_behind_ass(
                segments, start, end, ass_path, front_ass_path,
                title=title,
                caption_font_family=poster_font_family,
                colors=colors,
                important_size=important_size,
                head_y_at=head_y_at,
            )
        else:
          build_behind_ass(
            segments, start, end, ass_path, front_ass_path,
            caption_font_family=poster_font_family,
            colors=colors,
            important_size=important_size,
            common_size=common_size,
            head_y_at=head_y_at,
            # top word sits above the video (or near its top edge on a full-height video),
            # small words sit under it (or near the bottom edge on a full-height video)
            top_y=int(min(520, max(300, fg_top - 120))),
            bottom_y=int(min(1650, fg_top + fg_size[1] + 230)),
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
