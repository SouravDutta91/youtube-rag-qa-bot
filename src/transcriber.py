#!/usr/bin/env python3
"""
transcriber.py

Transcribes audio files in data/audio/ using faster-whisper.
Outputs transcripts to data/transcripts/<video_id>.txt and .json.
"""

import json
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "audio"
META_DIR = BASE_DIR / "data" / "metadata"
TRANS_DIR = BASE_DIR / "data" / "transcripts"
TRANS_DIR.mkdir(parents=True, exist_ok=True)

# Choose model size: "tiny", "base", "small", "medium", "large-v3"
MODEL_SIZE = "small"


def transcribe_audio(audio_path: Path, model: WhisperModel):
    """Transcribe a single audio file and return text + segments"""
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    full_text = []
    segment_list = []

    for seg in segments:
        segment_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
        full_text.append(seg.text.strip())

    return " ".join(full_text), segment_list


def main():
    model = WhisperModel(MODEL_SIZE, device="auto")

    audio_files = sorted(AUDIO_DIR.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files to transcribe.\n")

    for audio_path in tqdm(audio_files, desc="Transcribing files"):
        video_id = audio_path.stem
        transcript_path = TRANS_DIR / f"{video_id}.txt"
        transcript_json = TRANS_DIR / f"{video_id}.json"

        # Skip already processed
        if transcript_path.exists():
            continue

        try:
            text, segments = transcribe_audio(audio_path, model)

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(text)

            with open(transcript_json, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

            # update metadata with transcript path
            meta_file = META_DIR / f"{video_id}.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["transcript_path"] = str(transcript_path)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[WARN] Failed to transcribe {audio_path.name}: {e}")
            continue

    print("\nTranscription complete.")


if __name__ == "__main__":
    main()
