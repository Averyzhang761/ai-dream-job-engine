"""Exact evidence references into public job-description text."""

from ai_dream_job_engine.models import EvidenceChunk, EvidenceDocument, EvidenceRef


def build_evidence_chunks(text: str) -> EvidenceDocument:
    """Build one deterministic, lossless chunk for the source document."""
    chunks = []
    if text:
        chunks.append(
            EvidenceChunk(
                id="chunk-0001",
                text=text,
                start_offset=0,
                end_offset=len(text),
            )
        )
    return EvidenceDocument(text=text, chunks=chunks)


def validate_quote(document: EvidenceDocument, quote: str) -> EvidenceRef:
    """Return the first exact source location for a non-empty quote."""
    if not quote:
        raise ValueError("evidence quote cannot be empty")

    start = document.text.find(quote)
    if start < 0:
        raise ValueError("evidence quote is not present in the source document")

    end = start + len(quote)
    chunk = next(
        (
            candidate
            for candidate in document.chunks
            if candidate.start_offset <= start and candidate.end_offset >= end
        ),
        None,
    )
    if chunk is None:
        raise ValueError("evidence quote is not covered by a source chunk")
    return EvidenceRef(
        chunk_id=chunk.id,
        quote=quote,
        start_offset=start,
        end_offset=end,
    )
