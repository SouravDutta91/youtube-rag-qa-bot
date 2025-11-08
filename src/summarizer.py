#!/usr/bin/env python3
"""
summarizer.py

Generates summaries for each transcript using a transformer model (abstractive) or sumy (extractive).
Saves per-video summaries in data/summaries/<video_id>_summary.txt
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import nltk

def ensure_nltk_resources():
    """Download required NLTK resources if missing."""
    required = ["punkt", "punkt_tab"]
    for resource in required:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

ensure_nltk_resources()

BASE_DIR = Path(__file__).resolve().parents[1]
TRANS_DIR = BASE_DIR / "data" / "transcripts"
SUMM_DIR = BASE_DIR / "data" / "summaries"
META_DIR = BASE_DIR / "data" / "metadata"
SUMM_DIR.mkdir(parents=True, exist_ok=True)


def abstractive_summary(text: str, summarizer):
    """Summarize text using transformer pipeline."""
    # BART models have a max token limit (1024). We'll chunk long transcripts.
    max_chunk = 5000  # roughly 3000 chars per chunk
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=400, min_length=150, do_sample=False)
        summaries.append(summary[0]["summary_text"].strip())
    return " ".join(summaries)


def extractive_summary(text: str, sentence_count: int = 5):
    """Simple LexRank extractive summarizer (fallback if model unavailable)."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


def main():
    # Try loading abstractive summarizer
    use_extractive = False
    summarizer = None
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    except Exception as e:
        print(f"[WARN] Falling back to extractive summarization due to: {e}")
        use_extractive = True

    transcripts = sorted(TRANS_DIR.glob("*.txt"))
    print(f"Found {len(transcripts)} transcripts to summarize.\n")

    for tfile in tqdm(transcripts, desc="Summarizing transcripts"):
        video_id = tfile.stem
        output_file = SUMM_DIR / f"{video_id}_summary.txt"

        # Skip if already summarized
        if output_file.exists():
            continue

        try:
            with open(tfile, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if len(text) < 100:
                continue

            summary_text = (
                extractive_summary(text)
                if use_extractive
                else abstractive_summary(text, summarizer)
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary_text)

            # update metadata with summary path
            meta_file = META_DIR / f"{video_id}.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["summary_path"] = str(output_file)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[WARN] Failed to summarize {tfile.name}: {e}")
            continue

    print("\nSummarization complete.")


if __name__ == "__main__":
    main()
