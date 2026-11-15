from pathlib import Path
from uuid import uuid4
from pypdf import PdfReader
from docx import Document

ALLOWED = {".txt", ".pdf", ".docx"}

def extract_text(name: str, content: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError("Formato permitido: TXT, PDF ou DOCX")
    if suffix == ".txt":
        return content.decode("utf-8-sig", errors="replace")
    if suffix == ".pdf":
        from io import BytesIO
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    from io import BytesIO
    doc = Document(BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)

def split_chunks(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Exige size > overlap >= 0")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(paragraphs) > 1:
        chunks = []
        for paragraph in paragraphs:
            words = paragraph.split()
            step = size - overlap
            for i in range(0, len(words), step):
                part = " ".join(words[i:i + size])
                if part:
                    chunks.append(part)
        return chunks
    words = text.split()
    chunks, step = [], size - overlap
    for i in range(0, len(words), step):
        part = " ".join(words[i:i + size])
        if part:
            chunks.append(part)
    return chunks

def unique_document_id() -> str:
    return uuid4().hex
