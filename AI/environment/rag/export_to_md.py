#!/usr/bin/env python3
r"""
Exporta o conteúdo indexado no ChromaDB de volta para arquivos .md.
Reconstrói cada documento a partir dos chunks (na ordem certa, removendo
a sobreposição/overlap entre eles) e salva preservando a estrutura de
pastas original, dentro de uma pasta de saída.

É incremental: se o .md de destino já existe, o documento é PULADO (não
reprocessa nem reescreve). Use --force pra reexportar tudo de novo.

Processa vários documentos em paralelo (--workers, padrão 4) e mostra:
    - uma barra de progresso da BIBLIOTECA COMPLETA (conta o que já foi
      exportado em sessões anteriores + o que está sendo exportado agora)
    - uma barra de progresso da SESSÃO ATUAL (só o que falta exportar agora)
    - uma barra de progresso por ARQUIVO sendo processado no momento
      (uma por worker, aparece e some conforme os arquivos são concluídos)

Depende da biblioteca tqdm:
    pip install tqdm

Uso:
    python export_to_md.py                          # exporta só o que falta
    python export_to_md.py --source biblioteca       # só uma fonte, só o que falta
    python export_to_md.py --out D:\Export           # pasta de saída customizada
    python export_to_md.py --force                   # reexporta tudo, mesmo o que já existe
    python export_to_md.py --workers 8               # mais/menos processamento em paralelo
"""

import re
import sys
import argparse
import threading
from pathlib import Path
from queue import Queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb

try:
    from tqdm import tqdm
except ImportError:
    print("Esse script precisa da biblioteca 'tqdm' pras barras de progresso.")
    print("Instale com:  pip install tqdm")
    sys.exit(1)

YURIAI = Path("C:/IA")
VECTOR_DB_PATH = YURIAI / "AI" / "environment" / "vector-db"
COLLECTION_NAME = "biblioteca"
DEFAULT_OUTPUT = YURIAI / "AI" / "environment" / "export_md"
CHUNK_OVERLAP = 150       # mesmo valor usado no ingest.py, pra remover a duplicação nas junções
DEFAULT_WORKERS = 4       # documentos processados em paralelo


def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos para nomes de arquivo no Windows."""
    return re.sub(r'[<>:"|?*]', "_", name)


def merge_chunks(chunks_ordered: list[str], overlap: int, on_chunk=None) -> str:
    """Reconstrói o texto original a partir dos chunks, removendo a
    sobreposição (overlap) que existe entre um chunk e o próximo.
    Se `on_chunk` for passado, é chamado a cada chunk processado (usado
    pra avançar a barra de progresso do arquivo)."""
    if not chunks_ordered:
        return ""
    merged = chunks_ordered[0]
    if on_chunk:
        on_chunk()
    for chunk in chunks_ordered[1:]:
        overlap_slice = merged[-overlap:] if len(merged) >= overlap else merged
        if chunk.startswith(overlap_slice):
            merged += chunk[len(overlap_slice):]
        else:
            # Overlap não bateu exatamente (chunk pode ter sido editado
            # depois) — só concatena com uma quebra de linha por segurança.
            merged += "\n" + chunk
        if on_chunk:
            on_chunk()
    return merged


def resolve_out_path(doc_id: str, output_dir: Path) -> Path:
    """Calcula o caminho de saída .md pra um doc_id, sem escrever nada."""
    tag, _, rel_path = doc_id.partition("::")
    rel_path = Path(rel_path)
    md_relative = rel_path.with_suffix(".md")
    out_path = output_dir / tag / md_relative
    out_path = out_path.parent / sanitize_filename(out_path.name)
    return out_path


def process_doc(doc_id, chunk_list, output_dir, slot_pool):
    """Processa um documento inteiro (merge + escrita), com sua própria
    barra de progresso (uma posição/slot emprestada do pool de workers)."""
    slot = slot_pool.get()
    position = 2 + slot  # 0 e 1 são as barras de biblioteca completa e sessão
    try:
        chunk_list.sort(key=lambda x: x[0])
        ordered_texts = [c[1] for c in chunk_list]
        first_meta = chunk_list[0][2]

        out_path = resolve_out_path(doc_id, output_dir)
        label = out_path.name
        if len(label) > 28:
            label = label[:25] + "..."

        with tqdm(total=len(ordered_texts), desc=label, position=position,
                  leave=False, unit="chunk", dynamic_ncols=True) as file_bar:
            full_text = merge_chunks(ordered_texts, CHUNK_OVERLAP, on_chunk=lambda: file_bar.update(1))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = (
            "---\n"
            f"source_type: {first_meta.get('source_type', doc_id.partition('::')[0])}\n"
            f"original_path: {first_meta.get('source_path', '')}\n"
            f"doc_id: {doc_id}\n"
            "---\n\n"
        )
        out_path.write_text(frontmatter + full_text, encoding="utf-8")
        return doc_id, True, None
    except Exception as e:
        return doc_id, False, str(e)
    finally:
        slot_pool.put(slot)


def export(source_filter: str, output_dir: Path, force: bool, workers: int):
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Coleção não encontrada. Rode o ingest.py primeiro.")
        return

    where = {"source_type": source_filter} if source_filter else None
    print("Lendo todos os chunks da coleção...")
    all_data = collection.get(where=where, include=["documents", "metadatas"])

    docs = all_data["documents"]
    metas = all_data["metadatas"]

    if not docs:
        print("Nada encontrado para exportar (confira o filtro --source).")
        return

    # Agrupa os chunks por doc_id
    grouped = defaultdict(list)  # doc_id -> lista de (chunk_index, texto, meta)
    for doc, meta in zip(docs, metas):
        grouped[meta["doc_id"]].append((meta.get("chunk_index", 0), doc, meta))

    total_biblioteca = len(grouped)
    print(f"{total_biblioteca} documento(s) na base.")

    # Descobre o que já foi exportado (inclusive em sessões anteriores)
    pendentes = {}
    ja_existentes = 0
    for doc_id, chunk_list in grouped.items():
        out_path = resolve_out_path(doc_id, output_dir)
        if not force and out_path.exists():
            ja_existentes += 1
            continue
        pendentes[doc_id] = chunk_list

    if not force:
        print(f"{ja_existentes} já exportado(s) anteriormente (pulando).")

    if not pendentes:
        print("Nada novo pra exportar. Use --force pra reexportar tudo.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exportando {len(pendentes)} documento(s) pendente(s) para {output_dir} "
          f"usando {workers} worker(s) em paralelo...\n")

    slot_pool = Queue()
    for i in range(workers):
        slot_pool.put(i)

    exported = 0
    falhas = []

    with tqdm(total=total_biblioteca, initial=ja_existentes, position=0,
              desc="Biblioteca completa", unit="doc", dynamic_ncols=True) as overall_bar, \
         tqdm(total=len(pendentes), position=1,
              desc="Sessão atual     ", unit="doc", dynamic_ncols=True) as session_bar:

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_doc, doc_id, chunk_list, output_dir, slot_pool): doc_id
                for doc_id, chunk_list in pendentes.items()
            }
            for future in as_completed(futures):
                doc_id, ok, err = future.result()
                if ok:
                    exported += 1
                else:
                    falhas.append((doc_id, err))
                overall_bar.update(1)
                session_bar.update(1)

    print(f"\nConcluído. {exported} arquivo(s) .md gerado(s)/atualizado(s) em: {output_dir}")
    if falhas:
        print(f"\n{len(falhas)} documento(s) falharam:")
        for doc_id, err in falhas:
            print(f"  - {doc_id}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, default=None, choices=["biblioteca", "affine"],
                         help="Exportar só uma fonte específica (padrão: todas)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT),
                         help=f"Pasta de saída (padrão: {DEFAULT_OUTPUT})")
    parser.add_argument("--force", action="store_true",
                         help="Reexporta tudo, mesmo os .md que já existem (por padrão, pula quem já foi exportado)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Documentos processados em paralelo (padrão: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    export(args.source, Path(args.out), args.force, max(1, args.workers))
