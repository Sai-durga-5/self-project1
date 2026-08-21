"""
Loads HuggingFace datasets into ChromaDB collections.
Skips loading if the collection already has data.
"""

import sys
import os

# Ensure project root is in path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from rag.vectorstore import add_documents, collection_count


def _chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of roughly chunk_size characters."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break on whitespace
        if end < len(text):
            break_pos = text.rfind(" ", start, end)
            if break_pos > start:
                end = break_pos
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def load_resumes() -> None:
    """Load opensporks/resumes dataset into 'resumes' ChromaDB collection."""
    print("[Datasets] Checking 'resumes' collection...")
    if collection_count("resumes") > 0:
        print("[Datasets] 'resumes' collection already populated. Skipping.")
        return

    print("[Datasets] Loading opensporks/resumes from HuggingFace...")
    try:
        ds = load_dataset("opensporks/resumes", split="train")
    except Exception as e:
        print(f"[Datasets] WARNING: Could not load opensporks/resumes: {e}")
        print("[Datasets] Skipping resumes dataset.")
        return

    rows = list(ds)[:300]
    texts = []
    metadatas = []

    for i, row in enumerate(rows):
        # Try common text field names
        raw_text = (
            row.get("resume_text")
            or row.get("text")
            or row.get("content")
            or row.get("Resume")
            or str(row)
        )
        if not raw_text:
            continue
        chunks = _chunk_text(str(raw_text))
        for j, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": "opensporks/resumes", "row": i, "chunk": j})

        if (i + 1) % 50 == 0:
            print(f"[Datasets]   Processed {i + 1}/{len(rows)} resumes...")

    print(f"[Datasets] Adding {len(texts)} resume chunks to ChromaDB...")
    add_documents("resumes", texts, metadatas)
    print(f"[Datasets] 'resumes' collection loaded: {len(texts)} chunks.")


def load_job_descriptions() -> None:
    """Load NxtGenIntern/job_titles_and_descriptions into 'job_descriptions' collection."""
    print("[Datasets] Checking 'job_descriptions' collection...")
    if collection_count("job_descriptions") > 0:
        print("[Datasets] 'job_descriptions' collection already populated. Skipping.")
        return

    print("[Datasets] Loading NxtGenIntern/job_titles_and_descriptions from HuggingFace...")
    try:
        ds = load_dataset("NxtGenIntern/job_titles_and_descriptions", split="train")
    except Exception as e:
        print(f"[Datasets] WARNING: Could not load NxtGenIntern/job_titles_and_descriptions: {e}")
        print("[Datasets] Skipping job descriptions dataset.")
        return

    rows = list(ds)
    texts = []
    metadatas = []

    for i, row in enumerate(rows):
        title = row.get("job_title") or row.get("title") or row.get("Job Title") or ""
        description = (
            row.get("job_description")
            or row.get("description")
            or row.get("Description")
            or row.get("Job Description")
            or ""
        )
        combined = f"{title}\n\n{description}".strip()
        if not combined:
            continue
        texts.append(combined)
        metadatas.append({
            "source": "NxtGenIntern/job_titles_and_descriptions",
            "row": i,
            "title": str(title)[:100],
        })

        if (i + 1) % 100 == 0:
            print(f"[Datasets]   Processed {i + 1}/{len(rows)} job descriptions...")

    print(f"[Datasets] Adding {len(texts)} job descriptions to ChromaDB...")
    add_documents("job_descriptions", texts, metadatas)
    print(f"[Datasets] 'job_descriptions' collection loaded: {len(texts)} documents.")


def load_interview_questions() -> None:
    """Load ali-alkhars/interviews into 'interview_questions' collection."""
    print("[Datasets] Checking 'interview_questions' collection...")
    if collection_count("interview_questions") > 0:
        print("[Datasets] 'interview_questions' collection already populated. Skipping.")
        return

    print("[Datasets] Loading ali-alkhars/interviews from HuggingFace...")
    try:
        ds = load_dataset("ali-alkhars/interviews", split="train")
    except Exception as e:
        print(f"[Datasets] WARNING: Could not load ali-alkhars/interviews: {e}")
        print("[Datasets] Skipping interview questions dataset.")
        return

    rows = list(ds)[:200]
    texts = []
    metadatas = []

    for i, row in enumerate(rows):
        question = (
            row.get("question")
            or row.get("Question")
            or row.get("prompt")
            or ""
        )
        answer = (
            row.get("answer")
            or row.get("Answer")
            or row.get("response")
            or row.get("completion")
            or ""
        )
        combined = f"Q: {question}\nA: {answer}".strip()
        if not combined or combined == "Q: \nA: ":
            combined = str(row)
        texts.append(combined)
        metadatas.append({
            "source": "ali-alkhars/interviews",
            "row": i,
            "question": str(question)[:200],
        })

        if (i + 1) % 50 == 0:
            print(f"[Datasets]   Processed {i + 1}/{len(rows)} interview Q&As...")

    print(f"[Datasets] Adding {len(texts)} interview Q&As to ChromaDB...")
    add_documents("interview_questions", texts, metadatas)
    print(f"[Datasets] 'interview_questions' collection loaded: {len(texts)} documents.")


def load_all_datasets() -> None:
    """Load all three datasets into ChromaDB. Skip any that are already loaded."""
    print("\n" + "=" * 60)
    print("LOADING HUGGINGFACE DATASETS INTO CHROMADB")
    print("=" * 60)
    load_resumes()
    print()
    load_job_descriptions()
    print()
    load_interview_questions()
    print("=" * 60)
    print("All datasets processed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    load_all_datasets()
