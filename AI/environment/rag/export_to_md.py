#!/usr/bin/env python3
r"""
Exporta o conteúdo indexado no ChromaDB de volta para arquivos .md.
Reconstrói cada documento a partir dos chunks (na ordem certa, removendo
a sobreposição/overlap entre eles) e salva preservando a estrutura de
pastas original, dentro de uma pasta de saída.

Reexportação incremental: cada .md gerado guarda o file_hash do documento
no cabeçalho. Numa exportação seguinte, se o hash não mudou (e para PDFs,
que são imutáveis, isso é sempre verdade depois da primeira vez), o
arquivo é pulado — só reexporta o que de fato mudou desde a última vez.

Uso:
    python export_to_md.py                          # exporta/atualiza tudo
    python export_to_md.py --source biblioteca       # só uma fonte
    python export_to_md.py --out D:\Export           # pasta de saída customizada
    python export_to_md.py --force                   # ignora o cache e reexporta tudo
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

import chromadb
from tqdm import tqdm
import os


def winlong(p: Path) -> Path:
    r"""No Windows, caminhos completos com mais de ~260 caracteres falham
    silenciosamente (FileNotFoundError) mesmo que as pastas existam de
    verdade. O prefixo \\?\ diz ao Windows pra aceitar caminhos bem mais
    longos (até ~32000 caracteres) sem precisar encurtar nomes de pasta.
    Em outros sistemas operacionais, não faz nada."""
    if os.name != "nt":
        return p
    s = str(p.resolve())
    if s.startswith("\\\\?\\"):
        return p
    return Path("\\\\?\\" + s)

YURIAI = Path("C:/IA")
VECTOR_DB_PATH = YURIAI / "AI" / "environment" / "vector-db"
COLLECTION_NAME = "biblioteca"
DEFAULT_OUTPUT = YURIAI / "AI" / "environment" / "export_md"
CHUNK_OVERLAP = 150  # mesmo valor usado no ingest.py, pra remover a duplicação nas junções

FRONTMATTER_HASH_RE = re.compile(r"^file_hash:\s*(\S+)\s*$", re.MULTILINE)


def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos para nomes de arquivo no Windows."""
    return re.sub(r'[<>:"|?*]', "_", name)


def merge_chunks(chunks_ordered: list[str], overlap: int) -> str:
    """Reconstrói o texto original a partir dos chunks, removendo a
    sobreposição (overlap) que existe entre um chunk e o próximo."""
    if not chunks_ordered:
        return ""
    merged = chunks_ordered[0]
    for chunk in chunks_ordered[1:]:
        overlap_slice = merged[-overlap:] if len(merged) >= overlap else merged
        if chunk.startswith(overlap_slice):
            merged += chunk[len(overlap_slice):]
        else:
            merged += "\n" + chunk
    return merged


def existing_hash(md_path: Path) -> str | None:
    """Lê só o começo do .md já exportado (se existir) e extrai o
    file_hash guardado no cabeçalho, sem carregar o arquivo inteiro."""
    md_path = winlong(md_path)
    if not md_path.exists():
        return None
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            head = f.read(1000)  # cabeçalho é pequeno, não precisa ler tudo
        match = FRONTMATTER_HASH_RE.search(head)
        return match.group(1) if match else None
    except Exception:
        return None


def export(source_filter: str, output_dir: Path, force: bool):
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Coleção não encontrada. Rode o ingest.py primeiro.")
        return

    where = {"source_type": source_filter} if source_filter else None

    # count() conta a coleção inteira, ignorando o filtro --source — então
    # pegamos só os IDs (leve, sem o texto) pra saber o total real filtrado.
    id_only = collection.get(where=where, include=[])
    total_chunks = len(id_only["ids"])
    if total_chunks == 0:
        print("Nada encontrado para exportar (confira o filtro --source).")
        return

    docs, metas = [], []
    BATCH_SIZE = 500
    read_bar = tqdm(total=total_chunks, desc="Lendo chunks da coleção", unit="chunk", dynamic_ncols=True)
    offset = 0
    while True:
        batch = collection.get(where=where, include=["documents", "metadatas"], limit=BATCH_SIZE, offset=offset)
        batch_docs = batch["documents"]
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(batch["metadatas"])
        read_bar.update(len(batch_docs))
        offset += len(batch_docs)
        if len(batch_docs) < BATCH_SIZE:
            break
    read_bar.close()

    if not docs:
        print("Nada encontrado para exportar (confira o filtro --source).")
        return

    grouped = defaultdict(list)  # doc_id -> lista de (chunk_index, texto, meta)
    for doc, meta in zip(docs, metas):
        grouped[meta["doc_id"]].append((meta.get("chunk_index", 0), doc, meta))

    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    bar = tqdm(grouped.items(), total=len(grouped), desc="Exportando", unit="doc", dynamic_ncols=True)
    for doc_id, chunk_list in bar:
        chunk_list.sort(key=lambda x: x[0])
        first_meta = chunk_list[0][2]
        current_hash = first_meta.get("file_hash", "desconhecido")

        tag, _, rel_path = doc_id.partition("::")
        rel_path = Path(rel_path)
        md_relative = rel_path.with_suffix(".md")
        out_path = output_dir / tag / md_relative
        out_path = out_path.parent / sanitize_filename(out_path.name)

        # Pula se já foi exportado com esse mesmo hash antes (PDF já
        # exportado uma vez nunca muda; editáveis só reexportam se o
        # conteúdo mudou de verdade desde a última exportação).
        if not force and existing_hash(out_path) == current_hash:
            skipped += 1
            bar.set_postfix(exportados=exported, pulados=skipped)
            continue

        ordered_texts = [c[1] for c in chunk_list]
        full_text = merge_chunks(ordered_texts, CHUNK_OVERLAP)

        winlong(out_path.parent).mkdir(parents=True, exist_ok=True)
        frontmatter = (
            "---\n"
            f"source_type: {first_meta.get('source_type', tag)}\n"
            f"original_path: {first_meta.get('source_path', '')}\n"
            f"doc_id: {doc_id}\n"
            f"file_hash: {current_hash}\n"
            "---\n\n"
        )
        winlong(out_path).write_text(frontmatter + full_text, encoding="utf-8")
        exported += 1
        bar.set_postfix(exportados=exported, pulados=skipped)

    bar.close()
    print(f"Concluído. {exported} arquivo(s) exportado(s)/atualizado(s), {skipped} pulado(s) (sem mudança), em: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None, choices=["biblioteca", "affine"],
                         help="Exportar só uma fonte específica (padrão: todas)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT),
                         help=f"Pasta de saída (padrão: {DEFAULT_OUTPUT})")
    parser.add_argument("--force", action="store_true",
                         help="Ignora o cache de hash e reexporta tudo do zero")
    args = parser.parse_args()

    export(args.source, Path(args.out), args.force)
