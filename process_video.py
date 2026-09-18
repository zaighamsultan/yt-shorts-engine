"""
Simple YouTube-to-Shorts engine.
Takes a video URL (YouTube link or direct video file link), transcribes it,
asks Gemini to pick the best short moments, and cuts vertical (9:16) clips.

Runs on GitHub Actions (free tier) - no server needed.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import requests
from groq import Groq
from google import genai


def download_video(source: str, out_path: str) -> str:
    """Fetch a video from a direct URL (e.g. an uploaded file link) to out_path."""
    response = requests.get(source, stream=True, timeout=120)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
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


def transcribe(audio_path: str, api_key: str) -> dict:
    """Transcribe audio with Groq's Whisper API. Returns verbose JSON with
    both segment-level and word-level timestamps."""
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return result.model_dump() if hasattr(result, "model_dump") else json.loads(result.json())


def pick_clips(transcript: dict, api_key: str, min_clips: int, max_clips: int) -> list:
    """Ask Gemini which segments make good short clips."""
    client = genai.Client(api_key=api_key)

    segments = transcript.get("segments", [])
    transcript_text = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments
    )

    prompt = f"""You are picking short, engaging clips (15-60 seconds each) from a video transcript
for YouTube Shorts / TikTok / Reels. Pick between {min_clips} and {max_clips} clips.

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


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_captions_srt(words: list, clip_start: float, clip_end: float, srt_path: str, words_per_line: int = 4):
    """Build an SRT caption file for the words that fall inside one clip,
    with timestamps shifted so the clip's own start is time zero."""
    clip_words = [
        w for w in words
        if w["start"] >= clip_start and w["end"] <= clip_end
    ]

    lines = []
    for i in range(0, len(clip_words), words_per_line):
        chunk = clip_words[i:i + words_per_line]
        if not chunk:
            continue
        start = chunk[0]["start"] - clip_start
        end = chunk[-1]["end"] - clip_start
        text = " ".join(w["word"].strip() for w in chunk)
        lines.append((start, end, text))

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(lines, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            f.write(f"{text}\n\n")


def escape_drawtext(text: str) -> str:
    """Escape special characters so ffmpeg's drawtext filter doesn't choke on them."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\u2019")
        .replace("%", "\\%")
    )


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def cut_vertical_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    srt_path: str = None,
    title: str = None,
    brand_text: str = None,
):
    """Cut a segment, convert it to a 9:16 vertical clip with a blurred
    background, show the AI-picked title for the first 2.5s, burn in
    boxed captions, and add an optional small brand watermark."""
    duration = end - start
    filters = [
        "[0:v]scale=1080:1920,boxblur=20:5[bg]",
        "[0:v]scale=1080:-2[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[base]",
    ]
    current = "[base]"

    if title:
        safe_title = escape_drawtext(title)
        filters.append(
            f"{current}drawtext=fontfile={FONT_PATH}:text='{safe_title}':"
            "fontsize=52:fontcolor=white:borderw=4:bordercolor=black:"
            "x=(w-text_w)/2:y=140:enable='lt(t,2.5)'[titled]"
        )
        current = "[titled]"

    if brand_text:
        safe_brand = escape_drawtext(brand_text)
        filters.append(
            f"{current}drawtext=fontfile={FONT_PATH}:text='{safe_brand}':"
            "fontsize=28:fontcolor=white@0.85:borderw=2:bordercolor=black@0.6:"
            "x=w-text_w-30:y=h-60[branded]"
        )
        current = "[branded]"

    if srt_path:
        style = (
            "FontName=DejaVu Sans,FontSize=20,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H90000000,BorderStyle=3,Outline=1,"
            "Alignment=2,MarginV=90"
        )
        filters.append(f"{current}subtitles={srt_path}:force_style='{style}'[out]")
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
            "-c:a", "aac",
            output_path,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Direct URL to the video file (e.g. an uploaded video link)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--brand-text", default=None, help="Optional watermark text shown in the corner of every clip")
    args = parser.parse_args()

    groq_key = os.environ["GROQ_API_KEY"]
    gem_key = os.environ["GEM_API_KEY"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading video...")
    video_path = str(out_dir / "source.mp4")
    download_video(args.video, video_path)

    print("Extracting audio...")
    audio_path = str(out_dir / "audio.mp3")
    extract_audio(video_path, audio_path)

    print("Transcribing with Groq...")
    transcript = transcribe(audio_path, groq_key)
    with open(out_dir / "transcript.json", "w") as f:
        json.dump(transcript, f, indent=2)

    print("Picking clips with Gemini...")
    clips = pick_clips(transcript, gem_key, args.min_clips, args.max_clips)
    with open(out_dir / "clips_metadata.json", "w") as f:
        json.dump(clips, f, indent=2)

    words = transcript.get("words", [])

    print(f"Cutting {len(clips)} clips...")
    for i, clip in enumerate(clips, start=1):
        clip_path = str(out_dir / f"clip_{i}.mp4")
        srt_path = str(out_dir / f"clip_{i}.srt")
        build_captions_srt(words, clip["start"], clip["end"], srt_path)
        cut_vertical_clip(
            video_path, clip["start"], clip["end"], clip_path,
            srt_path=srt_path,
            title=clip.get("title"),
            brand_text=args.brand_text,
        )
        print(f"  saved {clip_path} - {clip.get('title', '')}")

    print("Done.")


if __name__ == "__main__":
    main()
