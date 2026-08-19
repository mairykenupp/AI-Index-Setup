#!/usr/bin/env python3
"""
Consulta restrita à Biblioteca (estilo NotebookLM).
O modelo só pode responder com base nos trechos recuperados do vetorial.
Se a resposta não estiver nos documentos, ele deve dizer isso claramente.

Uso:
    python query.py "sua pergunta aqui"
    python query.py            # modo interativo
"""

import sys
from pathlib import Path

import chromadb
import ollama

YURIAI = Path("C:/IA")
VECTOR_DB_PATH = YURIAI / "AI" / "environment" / "vector-db"
COLLECTION_NAME = "biblioteca"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:8b"     # ajuste se o nome do seu modelo no ollama list for diferente
TOP_K = 3
KEEP_ALIVE = "10m"          # tempo que o modelo fica carregado na RAM após a última pergunta
MAX_RESPONSE_TOKENS = 400   # limite de tamanho da resposta gerada

SYSTEM_PROMPT = """Você é um assistente que responde EXCLUSIVAMENTE com base nos trechos de documentos fornecidos abaixo (a "Biblioteca" do usuário).

Regras estritas:
1. Use APENAS as informações presentes nos trechos fornecidos. Nunca use conhecimento externo ou geral.
2. Se a resposta não estiver nos trechos fornecidos, diga claramente: "Não encontrei essa informação nos documentos da Biblioteca."
3. Sempre que possível, cite de qual arquivo veio a informação (o campo 'fonte' de cada trecho).
4. Não invente, complete ou extrapole informações além do que está escrito nos trechos.
5. Responda em português.
"""


def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Coleção vazia ou não encontrada. Rode o ingest.py primeiro.")
        sys.exit(1)


def retrieve(collection, question: str, k: int = TOP_K, source_filter: str = None):
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    where = {"source_type": source_filter} if source_filter else None
    results = collection.query(query_embeddings=[q_emb], n_results=k, where=where)

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        chunks.append({
            "text": doc,
            "source": meta.get("doc_id", "desconhecido"),
            "source_type": meta.get("source_type", "desconhecido"),
        })
    return chunks


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Trecho {i} — origem: {c['source_type']} — arquivo: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)


def ask(collection, question: str, source_filter: str = None):
    chunks = retrieve(collection, question, source_filter=source_filter)
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
        options={"num_predict": MAX_RESPONSE_TOKENS},
    )
    print("\n" + response["message"]["content"] + "\n")

    fontes = sorted(set(c["source"] for c in chunks))
    print(f"(Fontes consultadas: {', '.join(fontes)})")


if __name__ == "__main__":
    collection = get_collection()

    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
        ask(collection, pergunta)
    else:
        print("Modo interativo. Digite 'sair' para encerrar.\n")
        while True:
            pergunta = input("Você: ").strip()
            if pergunta.lower() in ("sair", "exit", "quit"):
                break
            if pergunta:
                ask(collection, pergunta)
