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


# ============================================================
# 5-ROW POSTER LAYOUT CONFIG  (this is the new default look)
# ============================================================
# Every phrase shows up as 5 rows:
#   row 1 : normal size, front  - lead-in words, sits just above row 2
#   row 2 : BIG     size, behind - hero word #1 (tucked behind the person)
#   row 3 : normal size, front  - small connector words
#   (gap : the person's face/body fills this space)
#   row 4 : BIG     size, front - hero word #2 (drawn in front of the person)
#   row 5 : normal size, front - ending words (same look as row 3)
#
# Sizes are multipliers of the "important" size.
FIVE_ROW_SIZES = {
    "row1": 0.70,
    "row2": 1.40,
    "row3": 0.68,
    "row4": 1.35,
    "row5": 0.70,
}
# Gap between the upper block (rows 1-3) and the lower block (rows 4-5),
# as a share of the video height. This is where the person's face/body sits.
FIVE_ROW_MID_GAP = 0.14
# Tight stacking distance inside a block, as a share of the text size.
FIVE_ROW_TIGHT_GAP = 0.90
# Words that must NEVER end a phrase (so phrases don't cut on conjunctions).
PHRASE_NO_END = {
    "a", "an", "the", "of", "to", "in", "on", "at", "and", "or", "but",
    "is", "are", "was", "were", "it", "this", "that", "with", "for",
    "from", "as", "be", "been", "by", "so", "if", "than", "then",
    "your", "my", "our", "their", "his", "her", "its", "we", "you",
    "i", "he", "she", "they", "will", "would", "can", "could", "do",
    "did", "does", "not", "no", "yes", "just", "like",
}
# Target words per phrase, before smart boundary fixes.
FIVE_ROW_WORDS = 5
FIVE_ROW_MIN_WORDS = 4
FIVE_ROW_MAX_WORDS = 7


def _download_bold_font(dest_path: str) -> bool:
    """Try to download Montserrat ExtraBold from Google Fonts so captions
    look clean and bold. Returns True on success."""
    urls = [
        "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
        "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-ExtraBold.ttf",
    ]
    for url in urls:
        try:
            print(f"  downloading bold font from {url.split('/')[-1]} ...")
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 20000:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception as e:
            print(f"  font download failed ({e}), trying next mirror...")
    return False


def find_system_font() -> str:
    """Find a bold Latin font on this machine, Linux (GitHub Actions) or Windows.
    Tries Montserrat first (clean bold), then DejaVu, then Windows Arial."""
    candidates = [
        "/usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[2]  # DejaVu Bold as last resort


def prepare_local_font() -> str:
    """Copy the system (or downloaded) bold font into the working directory.
    FFmpeg on Windows chokes on 'C:' in fontfile=, so a plain relative name
    keeps everything portable."""
    local_name = "clip_font.ttf"
    if os.path.exists(local_name):
        return local_name

    # 1) try to fetch Montserrat (clean modern bold) from Google Fonts
    if _download_bold_font(local_name):
        return local_name

    # 2) fall back to a bold system font
    shutil.copy(find_system_font(), local_name)
    return local_name


def bold_font_family_name() -> str:
    """The font family name libass should look up. If we downloaded
    Montserrat we use that; otherwise DejaVu Sans / Arial."""
    if os.path.exists("clip_font.ttf"):
        # crude sniff: Montserrat files usually contain the word in the binary
        try:
            with open("clip_font.ttf", "rb") as f:
                head = f.read(4096)
            if b"Montserrat" in head:
                return "Montserrat"
        except Exception:
            pass
    default_path = find_system_font()
    return "DejaVu Sans" if "DejaVu" in default_path else "Arial"


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
    the spoken language, so captions always come out in readable English."""
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
    key = value.strip().lower()
    return LANGUAGE_NAME_TO_CODE.get(key, key)


def transcribe_in_language(audio_path: str, api_key: str, language_code: str) -> dict:
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
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def transliterate_segments_to_latin(segments: list, api_key: str) -> list:
    if not segments:
        return segments
    client = genai.Client(api_key=api_key)
    texts = [s.get("text", "") for s in segments]
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    prompt = f"""Rewrite each numbered line below using ONLY the English/Latin alphabet -
a casual Roman transliteration of how it sounds. Do NOT translate the meaning.
Keep the same line numbers and same number of lines.

Lines:
{numbered}

Reply with ONLY a JSON object mapping each line number (as a string) to its
transliterated text. No other text, no markdown fences."""
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
    client = genai.Client(api_key=api_key)
    segments = transcript.get("segments", [])
    transcript_text = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments
    )
    prompt = f"""You are picking short, engaging clips (15-60 seconds each) from a video transcript
for YouTube Shorts / TikTok / Reels. Pick between {min_clips} and {max_clips} clips.
Write each title using ONLY the English/Latin alphabet.

Transcript with timestamps:
{transcript_text}

Reply with ONLY a JSON array, no other text, no markdown fences:
[{{"start": 12.5, "end": 45.0, "title": "short catchy title"}}]
"""
    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def pick_label(transcript: dict, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    segments = transcript.get("segments", [])
    transcript_text = "\n".join(s["text"] for s in segments)
    prompt = f"""Give one short, catchy title (under 10 words) that describes what this video is about.
Write it using ONLY the English/Latin alphabet.

Transcript:
{transcript_text}

Reply with ONLY the title text. No quotes, no markdown."""
    response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    return response.text.strip().strip('"')


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

DEFAULT_IMPORTANT_SIZE = 92
DEFAULT_COMMON_SIZE = 64
MIN_TEXT_SIZE = 40
MAX_TEXT_SIZE = 140

CAPTION_STYLES = ("classic", "behind", "five_row")
BEHIND_IMPORTANT_SIZE = 130
BEHIND_COMMON_SIZE = 84
BOTTOM_WORDS_PER_LINE = 5
BEHIND_TEXT_Y = 900
BEHIND_HEAD_OVERLAP = 0.35
BEHIND_MIN_Y = 240
BEHIND_FPS = 30
MASK_WORK_SIZE = 512

# legacy 'stack' settings (kept so old layouts still work if you flip them back)
STACK_GAP = 0.88
STACK_ATTACH_FRONT = True
STACK_TOP_FLOOR = 215
STACK_PATTERNS = {
    "row3":        [(3, 0.80, "behind")],
    "two_one":     [(2, 0.95, "behind"), (1, 1.30, "front")],
    "three_two":   [(3, 0.70, "behind"), (2, 1.00, "front")],
    "one_two":     [(1, 1.10, "behind"), (2, 0.90, "front")],
    "two_one_two": [(2, 0.80, "behind"), (1, 1.30, "front"), (2, 0.65, "front")],
    "ladder":      [(1, 0.55, "behind"), (1, 1.15, "behind"), (1, 0.95, "front")],
    "one":         [(1, 1.20, "behind")],
    "two":         [(2, 1.00, "behind")],
    "one_front":   [(1, 1.30, "front")],
}
BEHIND_LAYOUT = "stack"   # only used when caption-style=behind


def parse_caption_colors(value):
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
    try:
        size = int(value)
    except (TypeError, ValueError):
        return default
    return max(MIN_TEXT_SIZE, min(MAX_TEXT_SIZE, size))


def rgb_to_ass_inline(r: int, g: int, b: int) -> str:
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
        f"Style: Title,{caption_font_family},52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,4,0,8,40,40,90,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _is_common_word(word: str) -> bool:
    return word.strip(".,!?;:\"'").lower() in STOPWORDS


def _clean_word(w: str) -> str:
    return w.strip(".,!?;:\"'")


def _smart_phrase_boundaries(words: list, target=FIVE_ROW_WORDS) -> list:
    """Split a word list into phrase chunks of ~target words, but NEVER end a
    phrase on a conjunction / connector word. Returns list of index ranges."""
    n = len(words)
    if n == 0:
        return []

    chunks = []
    i = 0
    while i < n:
        remaining = n - i
        if remaining <= FIVE_ROW_MAX_WORDS:
            chunks.append((i, n))
            break

        take = target
        # don't end on a connector: extend until the ending word is a real word
        while take < remaining and _clean_word(words[i + take - 1]).lower() in PHRASE_NO_END:
            take += 1
        take = min(take, FIVE_ROW_MAX_WORDS, remaining)
        # pull back if we extended past the max and can still make a valid phrase
        if take > FIVE_ROW_MAX_WORDS:
            take = FIVE_ROW_MAX_WORDS

        # if the last word is *still* a connector, extend to the next real word
        while take < remaining and _clean_word(words[i + take - 1]).lower() in PHRASE_NO_END:
            take += 1
        take = min(take, remaining)

        # avoid leaving a tiny 1-word tail
        if remaining - take == 1 and take > FIVE_ROW_MIN_WORDS:
            take += 1

        chunks.append((i, i + take))
        i += take
    return chunks


def build_five_row_poster_ass(
    segments: list,
    clip_start: float,
    clip_end: float,
    behind_ass_path: str,
    front_ass_path: str,
    title: str = None,
    title_duration: float = 2.5,
    caption_font_family: str = "Montserrat",
    colors: list = None,
    important_size: int = BEHIND_IMPORTANT_SIZE,
    head_y_at=None,
    zones: dict = None,
):
    """NEW 5-row poster layout:
        row 1 : normal  (front)  - lead-in words, just above row 2
        row 2 : BIG     (behind) - hero word #1, tucked behind the person
        row 3 : normal  (front)  - connector words
        (gap : the person's face/body)
        row 4 : BIG     (front)  - hero word #2, drawn in front
        row 5 : normal  (front)  - ending words (same look as row 3)

    Phrases are cut with smart boundaries so they never end on conjunctions
    like 'a', 'the', 'of', 'it', 'and' - keeping the focus on the main topic."""
    palette = colors or WORD_COLORS
    zones = zones or {"top_y": 260, "bottom_y": 1650}
    header = _ass_header(caption_font_family, important_size)
    behind_lines, front_lines = [], []

    if title:
        front_lines.append(
            f"Dialogue: 2,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,"
            f"{sanitize_ass_text(title)}"
        )

    base = {
        "row1": max(40, int(important_size * FIVE_ROW_SIZES["row1"])),
        "row2": max(60, int(important_size * FIVE_ROW_SIZES["row2"])),
        "row3": max(40, int(important_size * FIVE_ROW_SIZES["row3"])),
        "row4": max(60, int(important_size * FIVE_ROW_SIZES["row4"])),
        "row5": max(40, int(important_size * FIVE_ROW_SIZES["row5"])),
    }

    key_i = 0
    for seg in segments:
        if not (seg["end"] > clip_start and seg["start"] < clip_end):
            continue
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = [w for w in seg["text"].strip().split() if _clean_word(w)]
        if not words:
            continue
        seg_dur = seg_end - seg_start
        word_dur = seg_dur / len(words)
        seg_t0 = max(0.0, seg_start - clip_start)

        for (a, b) in _smart_phrase_boundaries(words):
            phrase = words[a:b]
            n_words = len(phrase)
            if n_words == 0:
                continue

            # which words are "heroes" (non-connector words)?
            hero_idx = [k for k, w in enumerate(phrase) if not _is_common_word(w)]

            # need 2 heroes for a proper 5-row card; if not enough, degrade gracefully
            if len(hero_idx) >= 2:
                # hero #1 = first key word; hero #2 = last key word
                h1 = hero_idx[0]
                h2 = hero_idx[-1]
                # make sure they're different and h1 < h2
                if h1 >= h2:
                    hero_idx2 = [k for k in hero_idx if k != h1]
                    h2 = hero_idx2[-1] if hero_idx2 else h1
                # row 1 = words before h1 (or first word if h1 == 0)
                row1_words = phrase[:h1] if h1 > 0 else [phrase[0]]
                if h1 == 0 and n_words >= 2:
                    row1_words = [phrase[0]]
                    h1 = 1
                # row 3 = words strictly between h1 and h2 (connectors)
                row3_words = phrase[h1 + 1:h2]
                # row 5 = words after h2
                row5_words = phrase[h2 + 1:]
            else:
                # only 0-1 key word: fall back to a simple 3-row card (no hero rotation)
                row1_words = phrase[:1] if n_words >= 3 else []
                h1 = 1 if n_words >= 3 else 0
                row3_words = []
                h2 = h1
                row5_words = phrase[h1 + 1:] if n_words > h1 + 1 else []

            row2_words = [phrase[h1]] if 0 <= h1 < n_words else []
            row4_words = [phrase[h2]] if 0 <= h2 < n_words and h2 != h1 else []

            rows = [
                ("row1", row1_words, "front", False),
                ("row2", row2_words, "behind", True),
                ("row3", row3_words, "front", False),
                ("row4", row4_words, "front", True),
                ("row5", row5_words, "front", False),
            ]

            # remove empty rows for positioning, but remember which row index they were
            active_rows = [(name, wlist, layer, is_hero, i)
                           for i, (name, wlist, layer, is_hero) in enumerate(rows) if wlist]
            if not active_rows:
                continue

            # --- phrase timing window (in clip time) ---
            t_start = seg_t0 + a * word_dur
            t_end = seg_t0 + b * word_dur
            if t_end - t_start < 0.06:
                continue

            # --- sizing: shrink each row so it fits 940px ---
            sized = []
            for (name, wlist, layer, is_hero, row_i) in active_rows:
                text = " ".join(_clean_word(w).upper() if is_hero else _clean_word(w) for w in wlist)
                chars = max(1, len(text))
                fs = max(40, min(base[name], int(980 / (chars * 0.62))))
                sized.append({
                    "name": name, "words": wlist, "layer": layer,
                    "is_hero": is_hero, "row_i": row_i, "text": text, "fs": fs,
                })

            # --- vertical positions ---
            # upper block = rows 1..3 packed tight, row 2 anchored behind the head
            upper = [r for r in sized if r["row_i"] <= 2]
            lower = [r for r in sized if r["row_i"] >= 3]

            # find the hero (row 2) and anchor it behind the head
            behind_y = BEHIND_TEXT_Y
            row2 = next((r for r in upper if r["name"] == "row2"), None)
            if row2 is not None and head_y_at is not None:
                hy = head_y_at(t_start, t_end)
                if hy is not None:
                    behind_y = int(hy - row2["fs"] * (0.5 - BEHIND_HEAD_OVERLAP))
            behind_y = max(BEHIND_MIN_Y + (row2["fs"] // 2 if row2 else 60), behind_y)

            if row2 is not None:
                row2["y"] = behind_y
            # row 1 sits just above row 2
            row1 = next((r for r in upper if r["name"] == "row1"), None)
            if row1 is not None and row2 is not None:
                row1["y"] = row2["y"] - int(FIVE_ROW_TIGHT_GAP * (row1["fs"] + row2["fs"]) / 2)
            # row 3 sits just below row 2
            row3 = next((r for r in upper if r["name"] == "row3"), None)
            if row3 is not None and row2 is not None:
                row3["y"] = row2["y"] + int(FIVE_ROW_TIGHT_GAP * (row2["fs"] + row3["fs"]) / 2)

            # upper block must not spill below the mid gap
            upper_bottom_target = int(1920 * (0.5 + FIVE_ROW_MID_GAP / 2))
            if upper:
                lowest_upper = max(upper, key=lambda r: r["y"] + r["fs"] // 2)
                spill = (lowest_upper["y"] + lowest_upper["fs"] // 2) - upper_bottom_target
                if spill > 0:
                    for r in upper:
                        r["y"] -= spill

            # lower block: row 4 = second hero (front), row 5 right under it
            lower_top = upper_bottom_target + 20
            row4 = next((r for r in lower if r["name"] == "row4"), None)
            row5 = next((r for r in lower if r["name"] == "row5"), None)
            if row4 is not None:
                row4["y"] = max(lower_top + row4["fs"] // 2, zones.get("bottom_y", 1500) - 120)
            if row5 is not None:
                anchor = row4["y"] + row4["fs"] // 2 if row4 else lower_top
                row5["y"] = anchor + int(FIVE_ROW_TIGHT_GAP * (row5["fs"] + (row4["fs"] if row4 else row5["fs"])) / 2)

            # clamp lower block to frame
            if lower:
                lowest = max(lower, key=lambda r: r["y"] + r["fs"] // 2)
                overflow = (lowest["y"] + lowest["fs"] // 2) - 1830
                if overflow > 0:
                    for r in lower:
                        r["y"] -= overflow

            # --- emit one Dialogue per row (whole row shows together for its time window) ---
            for r in sized:
                is_behind = (r["layer"] == "behind")
                style = "Behind" if is_behind else "Default"

                if r["is_hero"]:
                    cr, cg, cb = palette[key_i % len(palette)]
                    key_i += 1
                    color = rgb_to_ass_inline(cr, cg, cb)
                else:
                    color = rgb_to_ass_inline(255, 255, 255)

                word = sanitize_ass_text(r["text"])
                if is_behind:
                    tags = (f"\\an5\\pos(540,{r['y']})\\fs{r['fs']}\\bord4{color}"
                            f"\\fscx82\\fscy82\\t(0,110,\\fscx108\\fscy108)\\t(110,190,\\fscx100\\fscy100)"
                            f"\\fad(60,60)")
                elif r["is_hero"]:
                    tags = (f"\\an5\\pos(540,{r['y']})\\fs{r['fs']}\\bord6{color}"
                            f"\\fscx70\\fscy70\\t(0,120,\\fscx112\\fscy112)\\t(120,200,\\fscx100\\fscy100)"
                            f"\\fad(60,60)")
                else:
                    tags = f"\\an5\\pos(540,{r['y']})\\fs{r['fs']}\\bord4{color}\\fad(60,60)"

                layer = 1 if (not is_behind and r["is_hero"]) else 0
                line = (f"Dialogue: {layer},{format_ass_time(t_start)},{format_ass_time(t_end)},"
                        f"{style},,0,0,0,,{{{tags}}}{word}")
                (behind_lines if is_behind else front_lines).append(line)

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
    """Classic bottom captions."""
    palette = colors or WORD_COLORS
    clip_segments = [s for s in segments if s["end"] > clip_start and s["start"] < clip_end]
    lines = []
    for seg in clip_segments:
        seg_start = max(seg["start"], clip_start)
        seg_end = min(seg["end"], clip_end)
        words = seg["text"].strip().split()
        if not words:
            continue
        n_chunks = max(1, -(-len(words) // words_per_line))
        chunk_duration = (seg_end - seg_start) / n_chunks
        for chunk_idx, i in enumerate(range(0, len(words), words_per_line)):
            chunk_words = words[i:i + words_per_line]
            c_start = seg_start + chunk_idx * chunk_duration - clip_start
            c_end = c_start + chunk_duration
            lines.append((max(0, c_start), c_end, chunk_words))

    header = _ass_header(caption_font_family, important_size)
    dialogue_lines = []
    if title:
        dialogue_lines.append(
            f"Dialogue: 1,{format_ass_time(0)},{format_ass_time(title_duration)},Title,,0,0,0,,"
            f"{sanitize_ass_text(title)}"
        )
    color_i = 0
    for start, end, chunk_words in lines:
        parts = []
        for w in chunk_words:
            safe_w = sanitize_ass_text(w)
            if _is_common_word(w):
                parts.append(f"{{{rgb_to_ass_inline(255,255,255)}\\fs{common_size}}}{safe_w}")
            else:
                r, g, b = palette[color_i % len(palette)]
                color_i += 1
                parts.append(f"{{{rgb_to_ass_inline(r,g,b)}\\fs{important_size}}}{safe_w}")
        dialogue_lines.append(
            f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Default,,0,0,0,,{' '.join(parts)}"
        )
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(dialogue_lines))
        f.write("\n")


def escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\u2019")
        .replace("%", "\\%")
    )


def ffmpeg_path_forward_slashes(path: str) -> str:
    return path.replace("\\", "/")


def resolve_caption_font(lang_code: str):
    """Return (fonts_dir, family_name). Always uses the bold Latin font now."""
    fonts_dir = "fonts"
    os.makedirs(fonts_dir, exist_ok=True)
    local_path = os.path.join(fonts_dir, "clip_font.ttf")
    if not os.path.exists(local_path):
        shutil.copy("clip_font.ttf", local_path)
    family = bold_font_family_name()
    return fonts_dir, family


def probe_fit_geometry(video_path: str):
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
    head_tops = []
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(mh, mw, 3)
            mask = segmenter.process(frame).segmentation_mask
            mask = np.clip((mask - 0.35) / 0.3, 0.0, 1.0)
            smoothed = mask if smoothed is None else 0.6 * mask + 0.4 * smoothed
            encoder.stdin.write((smoothed * 255).astype(np.uint8).tobytes())
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
    fg_w, fg_h = fg_size
    filters = [
        f"[0:v]fps={BEHIND_FPS},scale='trunc(iw*sar/2)*2':ih,setsar=1,split=2[bgsrc][fgsrc]",
        "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=25:5,setsar=1[bg]",
        f"[fgsrc]scale={fg_w}:{fg_h},setsar=1,split=2[fg][fgp]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]",
        f"[base]{_subtitles_filter(ass_path, fonts_dir)}[txt]",
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
            f"{current}drawtext=fontfile=clip_font.ttf:text='{safe_brand}':"
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


def cut_vertical_clip(
    video_path, start, end, output_path, ass_path=None,
    brand_text=None, fonts_dir=None, mask_path=None,
    title_ass_path=None, fg_size=None,
):
    duration = end - start
    if mask_path and fg_size:
        _cut_behind_clip(
            video_path, start, duration, output_path, ass_path, brand_text,
            fonts_dir, mask_path, title_ass_path, fg_size,
        )
        return
    filters = [
        "[0:v]scale='trunc(iw*sar/2)*2':ih,setsar=1,split=2[bgsrc][fgsrc]",
        "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=25:5,setsar=1[bg]",
        "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]",
    ]
    current = "[base]"
    if brand_text:
        safe_brand = escape_drawtext(brand_text)
        filters.append(
            f"{current}drawtext=fontfile=clip_font.ttf:text='{safe_brand}':"
            "fontsize=28:fontcolor=white@0.85:borderw=2:bordercolor=black@0.6:"
            "x=w-text_w-30:y=h-60[branded]"
        )
        current = "[branded]"
    if ass_path:
        filters.append(f"{current}{_subtitles_filter(ass_path, fonts_dir)}[out]")
        current = "[out]"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", video_path,
            "-filter_complex", ";".join(filters),
            "-map", current,
            "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ],
        check=True,
    )


def render_captioned_clip(
    video_path, start, end, clip_path, ass_path, segments, title,
    font_family, fonts_dir, colors, important_size, common_size,
    brand_text, caption_style,
):
    style = caption_style
    mask_path = None
    fg_size = None
    head_y_at = None

    if style in ("behind", "five_row"):
        try:
            fg_size = probe_fit_geometry(video_path)
            mask_path = str(Path(clip_path).with_suffix(".mask.mkv"))
            print("  finding the person in each frame...")
            _, head_tops = generate_person_mask_video(
                video_path, start, end - start, fg_size[0], fg_size[1], mask_path
            )
            fg_top = (1920 - fg_size[1]) / 2.0

            def head_y_at(t0, t1):
                i0 = max(0, int(t0 * BEHIND_FPS))
                i1 = min(len(head_tops), max(i0 + 1, int(t1 * BEHIND_FPS)))
                vals = sorted(v for v in head_tops[i0:i1] if v is not None)
                if not vals:
                    vals = sorted(v for v in head_tops if v is not None)
                if not vals:
                    return None
                return fg_top + vals[len(vals) // 2] * fg_size[1]
        except Exception as e:
            print(f"WARNING: behind-the-person effect unavailable ({e}). Using classic captions.")
            style = "classic"
            mask_path = None

    front_ass_path = None
    if style == "five_row":
        front_ass_path = str(Path(ass_path).with_suffix(".front.ass"))
        build_five_row_poster_ass(
            segments, start, end, ass_path, front_ass_path,
            title=title,
            caption_font_family=font_family,
            colors=colors,
            important_size=important_size,
            head_y_at=head_y_at,
            zones={"top_y": 260, "bottom_y": 1600},
        )
    elif style == "behind":
        front_ass_path = str(Path(ass_path).with_suffix(".front.ass"))
        build_captions_ass(
            segments, start, end, ass_path,
            title=title,
            caption_font_family=font_family,
            colors=colors,
            important_size=important_size,
            common_size=common_size,
        )
    else:
        if caption_style in ("behind", "five_row"):
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
            os.remove(mask_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--brand-text", default=None)
    parser.add_argument("--caption-only", default=None)
    parser.add_argument("--clip-language", default=None)
    parser.add_argument("--caption-colors", default=None)
    parser.add_argument("--important-size", type=int, default=None)
    parser.add_argument("--common-size", type=int, default=None)
    parser.add_argument(
        "--caption-style", default="five_row", choices=CAPTION_STYLES,
        help="'five_row' = 5-row poster layout (default, new). "
             "'classic' = bottom captions. 'behind' = big word behind person."
    )
    args = parser.parse_args()

    groq_key = os.environ["GROQ_API_KEY"]
    gem_key = os.environ["GEM_API_KEY"]

    caption_colors = parse_caption_colors(args.caption_colors)
    caption_style = args.caption_style if args.caption_style in CAPTION_STYLES else "five_row"
    is_big = caption_style in ("behind", "five_row")
    important_size = clamp_text_size(
        args.important_size, BEHIND_IMPORTANT_SIZE if is_big else DEFAULT_IMPORTANT_SIZE)
    common_size = clamp_text_size(
        args.common_size, BEHIND_COMMON_SIZE if is_big else DEFAULT_COMMON_SIZE)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"project_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")

    print("Preparing bold caption font...")
    FONT_PATH = prepare_local_font()
    print(f"Using font file: {FONT_PATH}")
    fonts_dir, font_family = resolve_caption_font("en")
    print(f"Caption font family: {font_family}")
    print(f"Caption style: {caption_style} - important size: {important_size}")

    print("Downloading video...")
    video_path = str(out_dir / "source.mp4")
    download_video(args.video, video_path)

    print("Extracting audio...")
    audio_path = str(out_dir / "audio.mp3")
    extract_audio(video_path, audio_path)

    if args.caption_only:
        lang_code = resolve_language_code(args.caption_only)
        if lang_code == "en":
            transcript = transcribe_english(audio_path, groq_key)
            caption_segments = transcript.get("segments", [])
        else:
            transcript = transcribe_in_language(audio_path, groq_key, lang_code)
            caption_segments = transliterate_segments_to_latin(transcript.get("segments", []), gem_key)
        with open(out_dir / "transcript.json", "w") as f:
            json.dump(transcript, f, indent=2)
        label = pick_label(transcript, gem_key)
        duration = get_video_duration(video_path)
        clip_path = str(out_dir / "captioned.mp4")
        ass_path = str(out_dir / "captions.ass")
        render_captioned_clip(
            video_path, 0, duration, clip_path, ass_path, caption_segments, label,
            font_family, fonts_dir, caption_colors, important_size, common_size,
            args.brand_text, caption_style,
        )
        print(f"  saved {clip_path}")
    else:
        clip_lang = resolve_language_code(args.clip_language) if args.clip_language else "en"
        if clip_lang == "en":
            transcript = transcribe_english(audio_path, groq_key)
            caption_segments = transcript.get("segments", [])
        else:
            transcript = transcribe_in_language(audio_path, groq_key, clip_lang)
            caption_segments = transliterate_segments_to_latin(transcript.get("segments", []), gem_key)
        with open(out_dir / "transcript.json", "w") as f:
            json.dump(transcript, f, indent=2)
        clips = pick_clips(transcript, gem_key, args.min_clips, args.max_clips)
        with open(out_dir / "clips_metadata.json", "w") as f:
            json.dump(clips, f, indent=2)
        for i, clip in enumerate(clips, start=1):
            clip_path = str(out_dir / f"clip_{i}.mp4")
            ass_path = str(out_dir / f"clip_{i}.ass")
            render_captioned_clip(
                video_path, clip["start"], clip["end"], clip_path, ass_path,
                caption_segments, clip.get("title"),
                font_family, fonts_dir, caption_colors, important_size, common_size,
                args.brand_text, caption_style,
            )
            print(f"  saved {clip_path} - {clip.get('title','')}")

    print("Done.")


if __name__ == "__main__":
    main()
