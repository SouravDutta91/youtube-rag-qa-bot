#!/usr/bin/env python

"""
    
"""

import os
import json
from pathlib import Path
from yt_dlp import YoutubeDL
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "audio"
META_DIR = BASE_DIR / "data" / "metadata"

for directory in (AUDIO_DIR, META_DIR):
    directory.mkdir(parents=True, exist_ok=True)

YTDL_OPTS = {
    "format": "bestaudio/best",
    "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
    "ignoreerrors": True,
    "quiet": True,
    "no_warnings": True,
    "noplaylist": False,
    "postprocessors": [
        {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
    ]
}

def download(url: str) -> list:
    """
    Downloads audio for the given youtube video or playlist URL.
    Returns a list of dicts containing {video_id, title, filepath, duration, webpage_url}
    """
    results = []
    with YoutubeDL(YTDL_OPTS) as ydl:
        info = ydl.extract_info(url, download=False)
        entries = info.get("entries") if info.get("entries") else [info]

        # iterate through entries and download each
        for entry in tqdm(entries, desc="\nProcessing items"):
            if entry is None:
                continue
            video_id = entry.get("id")
            title = entry.get("title")
            webpage_url = entry.get("webpage_url")
            duration = entry.get("duration")
            # download single video by id
            try:
                # use a per-item download call to ensure file naming
                ydl.download([webpage_url])
            except Exception as e:
                print(f"[WARN] Failed to download {webpage_url}: {e}")
                continue

            # expected mp3 path
            mp3_path = AUDIO_DIR / f"{video_id}.mp3"
            if not mp3_path.exists():
                # sometimes extension may vary — search for files with prefix
                matches = list(AUDIO_DIR.glob(f"{video_id}.*"))
                mp3_path = matches[0] if matches else None

            meta = {
                "video_id": video_id,
                "title": title,
                "webpage_url": webpage_url,
                "duration": duration,
                "audio_path": str(mp3_path) if mp3_path else None,
            }

            # write metadata json
            meta_file = META_DIR / f"{video_id}.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            results.append(meta)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download audio from a YouTube video or playlist")
    parser.add_argument("url", help="YouTube video or playlist URL")
    args = parser.parse_args()

    out = download(args.url)
    print("\nDownloaded items:")
    for item in out:
        print(f"\n- {item['video_id']}: {item['title']} -> {item['audio_path']}")