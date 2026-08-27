#!/usr/bin/env python3
"""
research.py — Fichamento / pesquisa direta na Biblioteca vetorizada.

Ao contrário do query.py e do server.py, essa ferramenta NÃO chama o modelo
de chat (qwen3:8b) pra sintetizar uma resposta. Ela usa só o modelo de
embeddings (nomic-embed-text — bem mais leve e rápido que gerar texto) pra
localizar trechos relevantes na base vetorial e te devolve TUDO organizado
por arquivo, com o texto original — como um fichamento de pesquisa: você lê
os trechos e tira suas próprias conclusões, sem uma IA "mastigando" antes.

É o plano B pra quando o modelo de chat estiver indisponível, lento demais,
ou quando você simplesmente quiser ver todo o material relevante (e de
temas relacionados) sem passar pelo filtro de uma síntese gerada.

Uso:
    # pesquisa simples — semântica + palavra-chave automática
    python research.py "Tratado de Madri"

    # pesquisa com temas relacionados (buscados separadamente, mas exibidos
    # juntos, agrupados por arquivo — cada trecho mostra qual tema achou ele)
    python research.py "Tratado de Madri" --relacionados "Uti possidetis" "fronteiras coloniais"

    # forçar busca literal por frases exatas (não depende de a frase ter
    # Maiúsculas — útil pra termos em minúsculo, jargão, siglas etc.)
    python research.py --exato "uti possidetis" "linha de Tordesilhas"

    # trazer bem mais trechos (levantamento amplo)
    python research.py "Tratado de Madri" --top-k 40

    # salvar o fichamento completo em markdown (sem truncar nada)
    python research.py "Tratado de Madri" --out fichamento_madri.md

    # filtrar por tipo de fonte
    python research.py "Tratado de Madri" --fonte biblioteca

    # modo interativo (pergunta uma consulta por vez)
    python research.py
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import chromadb
import ollama

# ---------------- CONFIG (mesma dos outros scripts) ----------------
YURIAI = Path("C:/IA")
VECTOR_DB_PATH = YURIAI / "AI" / "environment" / "vector-db"
COLLECTION_NAME = "biblioteca"
EMBED_MODEL = "nomic-embed-text"

TOP_K_DEFAULT = 20                # bem mais generoso que o das outras ferramentas,
                                   # já que aqui não tem custo de janela de contexto do chat
KEYWORD_MATCHES_PER_TERM = 15      # chunks trazidos por cada termo-chave/frase exata
MAX_KEYWORD_TERMS = 6

PREVIEW_CHARS = 500                # tamanho do preview no console (--full mostra tudo)

_KEYWORD_PATTERN = re.compile(
    r"[A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:d[aeo]s?|e)?\s*[A-ZÀ-Ú][a-zà-ú]+)*"
)
# ---------------------------------------------------------------------


def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Coleção vazia ou não encontrada. Rode o ingest.py primeiro.")
        sys.exit(1)


def extract_keywords(text: str):
    candidates = _KEYWORD_PATTERN.findall(text)
    seen = set()
    keywords = []
    for c in candidates:
        c = c.strip()
        if len(c) < 4:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(c)
        if len(keywords) >= MAX_KEYWORD_TERMS:
            break
    return keywords


def semantic_search(collection, theme: str, k: int, where):
    """Retorna [(chunk_dict, distance), ...] — distance menor = mais parecido."""
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=theme)["embedding"]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    out = []
    for d, m, i, dist in zip(docs, metas, ids, dists):
        out.append((_chunk_from(d, m, i), dist))
    return out


def keyword_search(collection, term: str, where):
    """Busca literal (contains). Sem distância — é match exato ou não é."""
    try:
        results = collection.get(
            where=where,
            where_document={"$contains": term},
            limit=KEYWORD_MATCHES_PER_TERM,
        )
    except Exception:
        return []
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []
    ids = results.get("ids", []) or []
    return [_chunk_from(d, m, i) for d, m, i in zip(docs, metas, ids)]


def _chunk_from(doc, meta, cid):
    return {
        "id": cid,
        "text": doc,
        "source": meta.get("doc_id", "desconhecido"),
        "source_type": meta.get("source_type", "desconhecido"),
    }


def run_research(collection, themes, exact_phrases, top_k, source_filter):
    """
    themes: lista de temas (cada um passa por busca semântica + palavra-chave
            automática a partir do próprio texto do tema)
    exact_phrases: frases pra busca literal forçada, sem heurística

    Retorna um dict: source -> lista de entradas
        {"chunk": ..., "distance": float|None, "matched": set(labels)}
    ordenado por relevância dentro de cada arquivo.
    """
    where = {"source_type": source_filter} if source_filter else None
    by_id = {}  # chunk_id -> entry

    def register(chunk, distance, label):
        entry = by_id.get(chunk["id"])
        if entry is None:
            entry = {"chunk": chunk, "distance": distance, "matched": set()}
            by_id[chunk["id"]] = entry
        else:
            if distance is not None and (entry["distance"] is None or distance < entry["distance"]):
                entry["distance"] = distance
        entry["matched"].add(label)

    for theme in themes:
        label = f'semântica: "{theme}"'
        for chunk, dist in semantic_search(collection, theme, top_k, where):
            register(chunk, dist, label)

        kw_label_base = f'palavra-chave (de "{theme}")'
        for kw in extract_keywords(theme):
            for chunk in keyword_search(collection, kw, where):
                register(chunk, None, f'{kw_label_base}: "{kw}"')

    for phrase in exact_phrases:
        for chunk in keyword_search(collection, phrase, where):
            register(chunk, None, f'frase exata: "{phrase}"')

    # agrupa por arquivo
    by_source = {}
    for entry in by_id.values():
        by_source.setdefault(entry["chunk"]["source"], []).append(entry)

    # dentro de cada arquivo, prioriza quem tem distância (mais relevante primeiro);
    # quem só veio de match literal (sem distância) vai depois, na ordem de descoberta
    for source, entries in by_source.items():
        entries.sort(key=lambda e: (e["distance"] is None, e["distance"] if e["distance"] is not None else 0))

    return by_source


def format_report(by_source, themes, exact_phrases, full=False):
    lines = []
    total_chunks = sum(len(v) for v in by_source.values())
    query_desc = ", ".join(f'"{t}"' for t in themes) if themes else ""
    if exact_phrases:
        if query_desc:
            query_desc += " + "
        query_desc += "frases exatas: " + ", ".join(f'"{p}"' for p in exact_phrases)

    lines.append(f"# Fichamento — {query_desc}")
    lines.append(f"\n_Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} — "
                 f"{total_chunks} trecho(s) em {len(by_source)} arquivo(s)._\n")

    for source in sorted(by_source.keys()):
        entries = by_source[source]
        lines.append(f"\n## 📄 {source}  ({len(entries)} trecho(s))\n")
        for i, entry in enumerate(entries, 1):
            chunk = entry["chunk"]
            dist = entry["distance"]
            matched = " | ".join(sorted(entry["matched"]))
            score = f"distância: {dist:.4f}" if dist is not None else "match literal"
            lines.append(f"**Trecho {i}** — {score}")
            lines.append(f"_achado por: {matched}_\n")
            text = chunk["text"] if full else (
                chunk["text"] if len(chunk["text"]) <= PREVIEW_CHARS
                else chunk["text"][:PREVIEW_CHARS] + " […]"
            )
            lines.append(f"> {text}\n")

    return "\n".join(lines)


def do_one_research(collection, themes, exact_phrases, top_k, source_filter, out_path):
    by_source = run_research(collection, themes, exact_phrases, top_k, source_filter)

    if not by_source:
        print("Nada encontrado pra esse(s) tema(s)/frase(s).")
        return

    # console: preview truncado (a menos que vá salvar em arquivo, aí mostra full só no arquivo)
    console_report = format_report(by_source, themes, exact_phrases, full=False)
    print("\n" + console_report + "\n")

    if out_path:
        full_report = format_report(by_source, themes, exact_phrases, full=True)
        Path(out_path).write_text(full_report, encoding="utf-8")
        print(f"(Fichamento completo salvo em: {out_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tema", nargs="?", help="Tema principal da pesquisa")
    parser.add_argument("--relacionados", nargs="+", default=[],
                         help="Temas relacionados, pesquisados junto e mostrados no mesmo fichamento")
    parser.add_argument("--exato", nargs="+", default=[],
                         help="Frase(s) exata(s) pra busca literal, sem depender de Maiúsculas")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT,
                         help=f"Chunks semânticos por tema (padrão: {TOP_K_DEFAULT})")
    parser.add_argument("--fonte", choices=["biblioteca", "affine"], default=None,
                         help="Filtra por tipo de fonte")
    parser.add_argument("--out", help="Salva o fichamento completo (sem truncar) em um .md")
    args = parser.parse_args()

    collection = get_collection()
    themes = ([args.tema] if args.tema else []) + args.relacionados

    if themes or args.exato:
        do_one_research(collection, themes, args.exato, args.top_k, args.fonte, args.out)
    else:
        print("Modo interativo. Digite um tema (ou 'sair' pra encerrar).")
        print("Dica: pra frase exata, prefixe com 'exato:' — ex: exato:uti possidetis\n")
        while True:
            consulta = input("Tema: ").strip()
            if consulta.lower() in ("sair", "exit", "quit"):
                break
            if not consulta:
                continue
            if consulta.lower().startswith("exato:"):
                do_one_research(collection, [], [consulta.split(":", 1)[1].strip()],
                                 args.top_k, args.fonte, None)
            else:
                do_one_research(collection, [consulta], [], args.top_k, args.fonte, None)
