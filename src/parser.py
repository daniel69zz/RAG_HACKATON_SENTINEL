from pathlib import Path
from typing import List, Dict

def load_markdown_files(markdown_dir: str = "data/markdown") -> List[Dict[str, str]]:
    """Cargar todos los archivos .md de un directorio"""
    docs = []
    md_path = Path(markdown_dir)
    
    if not md_path.exists():
        print(f"⚠️  Directorio {markdown_dir} no existe")
        return docs
    
    for md_file in md_path.glob("**/*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append({
                    "source": md_file.name,  # nombre del archivo
                    "content": content,
                    "path": str(md_file)
                })
                print(f"  ✓ Cargado: {md_file.name}")
        except Exception as e:
            print(f"  ✗ Error leyendo {md_file.name}: {e}")
    
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Dividir texto en chunks con overlap"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


def parse_documents(raw_text: str) -> List[Dict[str, str]]:
    """Parsear documentos marcados con [DOC: nombre.txt] (para compatibilidad)"""
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