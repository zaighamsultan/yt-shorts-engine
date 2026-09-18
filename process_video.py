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
    """Transcribe audio with Groq's Whisper API. Returns verbose JSON with timestamps."""
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
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


def cut_vertical_clip(video_path: str, start: float, end: float, output_path: str):
    """Cut a segment and convert it to a 9:16 vertical clip with a blurred background."""
    duration = end - start
    filter_complex = (
        "[0:v]scale=1080:1920,boxblur=20:5[bg];"
        "[0:v]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", video_path,
            "-filter_complex", filter_complex,
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

    print(f"Cutting {len(clips)} clips...")
    for i, clip in enumerate(clips, start=1):
        clip_path = str(out_dir / f"clip_{i}.mp4")
        cut_vertical_clip(video_path, clip["start"], clip["end"], clip_path)
        print(f"  saved {clip_path} - {clip.get('title', '')}")

    print("Done.")


if __name__ == "__main__":
    main()
