#!/usr/bin/env python3
"""
Consulta restrita à Biblioteca (estilo NotebookLM).
O modelo só pode responder com base nos trechos recuperados do vetorial.
Se a resposta não estiver nos documentos, ele deve dizer isso claramente.

Busca híbrida:
    - Semântica: os TOP_K chunks mais parecidos com a pergunta (bom pra
      perguntas conceituais, tipo "quais fatores levaram a X").
    - Palavra-chave: qualquer chunk que contenha LITERALMENTE algum termo
      próprio da pergunta (ex: "Tratado de Madri"). Isso cobre o caso em
      que a busca semântica não prioriza um trecho que cita o termo exato,
      mas cujo texto ao redor não é "parecido o bastante" com a pergunta —
      comum em perguntas do tipo "resuma tudo sobre X" onde X está
      espalhado em poucos parágrafos dentro de um documento grande.
    Os dois conjuntos são combinados e, se passarem do orçamento de
    contexto, cortados de forma a sempre manter os resultados semânticos
    (mais relevantes em geral) e completar com os de palavra-chave.

Uso:
    python query.py "sua pergunta aqui"
    python query.py            # modo interativo
    python query.py --top-k 10 "resuma tudo sobre o Tratado de Madri"
"""

import re
import sys
import argparse
from pathlib import Path

import chromadb
import ollama

YURIAI = Path("C:/IA")
VECTOR_DB_PATH = YURIAI / "AI" / "environment" / "vector-db"
COLLECTION_NAME = "biblioteca"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:8b"     # ajuste se o nome do seu modelo no ollama list for diferente
KEEP_ALIVE = "10m"          # tempo que o modelo fica carregado na RAM após a última pergunta

# --------------- BUSCA ---------------
TOP_K = 8                       # chunks trazidos pela busca semântica (antes: 3 — baixo demais pra perguntas amplas tipo "resuma")
KEYWORD_MATCHES_PER_TERM = 4     # chunks trazidos por CADA termo-chave encontrado literalmente
MAX_KEYWORD_TERMS = 4            # no máximo essa quantidade de termos-chave é pesquisada por pergunta
MAX_CONTEXT_CHUNKS = 16          # teto absoluto de chunks combinados (semântico + palavra-chave), pra não estourar o contexto

# --------------- CONTEXTO / TOKENS DO MODELO ---------------
# num_ctx é o tamanho da "janela" que o modelo enxerga de uma vez (pergunta +
# trechos recuperados + resposta, tudo junto). Se os trechos recuperados já
# ocupam quase toda a janela, sobra pouco espaço pra resposta e ela sai
# cortada quase no início — foi o que aconteceu antes. Aumentamos aqui.
CONTEXT_WINDOW = 12288
# Quanto o modelo pode escrever de resposta. Também era baixo demais (400)
# pra pedidos de resumo mais longos.
MAX_RESPONSE_TOKENS = 1200
# Orçamento aproximado de caracteres pros trechos recuperados, reservando
# espaço na janela pro prompt de sistema, a pergunta e a resposta.
# Regra grosseira: ~4 caracteres por token em português.
CONTEXT_CHAR_BUDGET = (CONTEXT_WINDOW - MAX_RESPONSE_TOKENS - 600) * 4

SYSTEM_PROMPT = """Você é um assistente que responde EXCLUSIVAMENTE com base nos trechos de documentos fornecidos abaixo (a "Biblioteca" do usuário).

Regras estritas:
1. Use APENAS as informações presentes nos trechos fornecidos. Nunca use conhecimento externo ou geral.
2. Se a resposta não estiver nos trechos fornecidos, diga claramente: "Não encontrei essa informação nos documentos da Biblioteca."
3. Sempre que possível, cite de qual arquivo veio a informação (o campo 'fonte' de cada trecho).
4. Se os trechos vierem de partes diferentes ou de documentos diferentes, cruze e organize as informações num resumo coerente, em vez de listar cada trecho isoladamente.
5. Não invente, complete ou extrapole informações além do que está escrito nos trechos.
6. Responda em português.
"""

# Sequências de palavras capitalizadas (com acentos), tipo "Tratado de
# Madri", "Rio Branco", "Organização das Nações Unidas" — heurística simples
# pra achar nomes próprios/termos específicos dentro da pergunta.
_KEYWORD_PATTERN = re.compile(
    r"[A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:d[aeo]s?|e)?\s*[A-ZÀ-Ú][a-zà-ú]+)*"
)


def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Coleção vazia ou não encontrada. Rode o ingest.py primeiro.")
        sys.exit(1)


def extract_keywords(question: str):
    """Extrai candidatos a termo-chave (nomes próprios/expressões
    capitalizadas) da pergunta, pra busca literal complementar à semântica."""
    candidates = _KEYWORD_PATTERN.findall(question)
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


def _chunk_from(doc, meta, cid):
    return {
        "id": cid,
        "text": doc,
        "source": meta.get("doc_id", "desconhecido"),
        "source_type": meta.get("source_type", "desconhecido"),
    }


def semantic_search(collection, question: str, k: int, where):
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    results = collection.query(query_embeddings=[q_emb], n_results=k, where=where)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    return [_chunk_from(d, m, i) for d, m, i in zip(docs, metas, ids)]


def keyword_search(collection, keywords, where):
    """Busca literal: qualquer chunk que contenha o termo, palavra por
    palavra (usa o operador $contains do Chroma), sem depender de
    similaridade semântica."""
    found = []
    for term in keywords:
        try:
            results = collection.get(
                where=where,
                where_document={"$contains": term},
                limit=KEYWORD_MATCHES_PER_TERM,
            )
        except Exception:
            continue
        docs = results.get("documents", []) or []
        metas = results.get("metadatas", []) or []
        ids = results.get("ids", []) or []
        for d, m, i in zip(docs, metas, ids):
            found.append(_chunk_from(d, m, i))
    return found


def retrieve(collection, question: str, k: int = TOP_K, source_filter: str = None):
    where = {"source_type": source_filter} if source_filter else None

    semantic_chunks = semantic_search(collection, question, k, where)
    keywords = extract_keywords(question)
    keyword_chunks = keyword_search(collection, keywords, where) if keywords else []

    # Combina mantendo prioridade pro que a busca semântica já trouxe,
    # completando com os achados literais que ainda não estavam na lista.
    seen_ids = set()
    combined = []
    for c in semantic_chunks + keyword_chunks:
        if c["id"] in seen_ids:
            continue
        seen_ids.add(c["id"])
        combined.append(c)

    combined = combined[:MAX_CONTEXT_CHUNKS]

    # Proteção contra estouro de contexto: se mesmo assim os trechos juntos
    # passarem do orçamento de caracteres, corta os últimos (que são os
    # menos prioritários — semânticos vêm primeiro na lista).
    total = 0
    trimmed = []
    for c in combined:
        total += len(c["text"])
        if total > CONTEXT_CHAR_BUDGET and trimmed:
            break
        trimmed.append(c)

    return trimmed


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Trecho {i} — origem: {c['source_type']} — arquivo: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)


def ask(collection, question: str, source_filter: str = None, top_k: int = TOP_K):
    chunks = retrieve(collection, question, k=top_k, source_filter=source_filter)
    if not chunks:
        print("Não encontrei essa informação nos documentos indexados.")
        return

    context = build_context(chunks)
    user_message = f"Trechos recuperados:\n\n{context}\n\nPergunta: {question}"

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        keep_alive=KEEP_ALIVE,
        options={"num_predict": MAX_RESPONSE_TOKENS, "num_ctx": CONTEXT_WINDOW},
    )
    print("\n" + response["message"]["content"] + "\n")

    if response.get("done_reason") == "length":
        print(
            "⚠️  A resposta foi cortada por atingir o limite de tokens "
            f"({MAX_RESPONSE_TOKENS}). Se precisar de mais detalhe, peça "
            "de forma mais específica ou aumente MAX_RESPONSE_TOKENS no script.\n"
        )

    fontes = sorted(set(c["source"] for c in chunks))
    print(f"(Fontes consultadas — {len(chunks)} trecho(s): {', '.join(fontes)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Pergunta (se omitida, entra em modo interativo)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                         help=f"Quantos chunks a busca semântica traz (padrão: {TOP_K}). "
                              "Aumente para perguntas amplas tipo 'resuma tudo sobre X'.")
    args = parser.parse_args()

    collection = get_collection()

    if args.question:
        pergunta = " ".join(args.question)
        ask(collection, pergunta, top_k=args.top_k)
    else:
        print("Modo interativo. Digite 'sair' para encerrar.\n")
        while True:
            pergunta = input("Você: ").strip()
            if pergunta.lower() in ("sair", "exit", "quit"):
                break
            if pergunta:
                ask(collection, pergunta, top_k=args.top_k)
