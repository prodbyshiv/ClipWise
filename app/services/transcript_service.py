import re
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=)([\w-]+)",
        r"youtu\.be/([\w-]+)",
        r"shorts/([\w-]+)",
        r"embed/([\w-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> List[Dict[str, Any]]:
    """
    Fetch transcript for a YouTube video.
    Compatible with youtube-transcript-api >= 0.7.x (instance-based API).
    Returns list of dicts: [{text, start, duration}, ...]

    Priority: manual English → auto-generated English → first available (translated to EN)
    """
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        # Try manual English first
        try:
            transcript = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
            fetched = transcript.fetch()
            return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        except Exception:
            pass

        # Try auto-generated English
        try:
            transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            fetched = transcript.fetch()
            return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        except Exception:
            pass

        # Fallback: first available transcript, translate to English
        try:
            transcript = next(iter(transcript_list))
            if transcript.language_code != "en":
                transcript = transcript.translate("en")
            fetched = transcript.fetch()
            return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        except Exception:
            pass

        raise ValueError(f"No usable transcript found for video {video_id}.")

    except ValueError:
        raise
    except Exception as e:
        err = str(e)
        if "disabled" in err.lower() or "TranscriptsDisabled" in err:
            raise ValueError(f"Transcripts are disabled for video {video_id}.")
        if "NoTranscriptFound" in err or "no transcript" in err.lower():
            raise ValueError(f"No transcript found for video {video_id}. The video may not have captions.")
        raise ValueError(f"Failed to fetch transcript: {err}")


def transcript_to_chunks_with_timing(
    transcript: List[Dict[str, Any]],
    chunk_size_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split transcript into overlapping text chunks, preserving start_time metadata.
    Each chunk: {text, start_time, chunk_index}
    """
    if not transcript:
        return []

    chunks = []
    current_text = ""
    current_start = transcript[0].get("start", 0.0)
    chunk_index = 0

    for segment in transcript:
        seg_text = str(segment.get("text", "")).strip()
        if not seg_text:
            continue

        if not current_text:
            current_start = float(segment.get("start", 0.0))

        current_text += " " + seg_text

        if len(current_text) >= chunk_size_chars:
            chunks.append({
                "text": current_text.strip(),
                "start_time": current_start,
                "chunk_index": chunk_index,
            })
            # carry over overlap to preserve context continuity
            current_text = current_text[-overlap_chars:] if len(current_text) > overlap_chars else ""
            chunk_index += 1

    # Final remaining text
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "start_time": current_start,
            "chunk_index": chunk_index,
        })

    return chunks
