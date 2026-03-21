from pathlib import Path
from typing import List, Dict
import re


def load_markdown_files(markdown_dir: str = "data/markdown") -> List[Dict[str, str]]:
    docs = []
    md_path = Path(markdown_dir)

    if not md_path.exists():
        print(f"⚠️  Directorio {markdown_dir} no existe")
        return docs

    for md_file in md_path.glob("**/*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({
                    "source": md_file.name,
                    "content": content,
                    "path": str(md_file)
                })
                print(f"  ✓ Cargado: {md_file.name}")
        except Exception as e:
            print(f"  ✗ Error leyendo {md_file.name}: {e}")

    return docs


def _normalize_block(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_markdown_sections(text: str) -> List[Dict[str, str]]:
    """
    Divide el markdown por secciones principales (##) y subsecciones (###).
    Ideal para leyes, artículos y documentos con encabezados claros.
    """
    lines = text.splitlines()

    sections: List[Dict[str, str]] = []
    current_h2 = ""
    current_h3 = ""
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_h2, current_h3, sections
        content = _normalize_block("\n".join(buffer))
        if content:
            title_parts = [p for p in [current_h2, current_h3] if p]
            section_title = " | ".join(title_parts) if title_parts else "Sin título"
            sections.append({
                "title": section_title,
                "content": content
            })
        buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("## "):
            flush()
            current_h2 = line[3:].strip()
            current_h3 = ""
            buffer = [line]
        elif line.startswith("### "):
            flush()
            current_h3 = line[4:].strip()
            buffer = []
            if current_h2:
                buffer.append(f"## {current_h2}")
            buffer.append(line)
        else:
            buffer.append(line)

    flush()
    return sections


def _split_large_section(section_text: str, max_chars: int = 900) -> List[str]:
    """
    Si una sección es grande, la parte por párrafos o listas sin romper
    demasiado la unidad semántica.
    """
    section_text = _normalize_block(section_text)
    if len(section_text) <= max_chars:
        return [section_text]

    paragraphs = re.split(r"\n\s*\n", section_text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            if len(para) <= max_chars:
                current = para
            else:
                # último recurso: dividir un párrafo muy largo por líneas
                lines = para.splitlines()
                sub = ""
                for line in lines:
                    candidate_line = f"{sub}\n{line}".strip() if sub else line
                    if len(candidate_line) <= max_chars:
                        sub = candidate_line
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = line
                current = sub

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Chunking semántico para markdown legal:
    1. divide por encabezados ## y ###,
    2. conserva el contexto de la sección,
    3. solo subdivide si el bloque queda muy largo.
    """
    if chunk_size <= 200:
        raise ValueError("chunk_size demasiado pequeño para markdown legal")

    sections = _split_markdown_sections(text)
    if not sections:
        return [_normalize_block(text)] if text.strip() else []

    chunks: List[str] = []

    for section in sections:
        title = section["title"]
        content = section["content"]

        semantic_block = f"[SECCIÓN: {title}]\n{content}".strip()

        if len(semantic_block) <= chunk_size:
            chunks.append(semantic_block)
            continue

        subchunks = _split_large_section(semantic_block, max_chars=chunk_size)

        # pequeño solapamiento semántico entre subchunks
        previous_tail = ""
        for sub in subchunks:
            if previous_tail:
                merged = f"{previous_tail}\n{sub}".strip()
                chunks.append(merged)
            else:
                chunks.append(sub)

            tail = sub[-overlap:].strip()
            previous_tail = f"[CONTEXTO PREVIO]\n{tail}" if tail else ""

    # quitar duplicados exactos preservando orden
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            seen.add(key)
            unique_chunks.append(key)

    return unique_chunks


def parse_documents(raw_text: str) -> List[Dict[str, str]]:
    docs = []
    current_source = None
    current_lines = []

    for line in raw_text.splitlines():
        line = line.strip()

        if line.startswith("[DOC:") and line.endswith("]"):
            if current_source and current_lines:
                docs.append({
                    "source": current_source,
                    "content": "\n".join(current_lines).strip()
                })
                current_lines = []

            current_source = line.replace("[DOC:", "").replace("]", "").strip()
        else:
            if line:
                current_lines.append(line)

    if current_source and current_lines:
        docs.append({
            "source": current_source,
            "content": "\n".join(current_lines).strip()
        })

    return docs