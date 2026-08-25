#!/usr/bin/env python3
"""
Interface web leve para consultar a Biblioteca (RAG) via Ollama.
Roda um servidor Flask local com uma página HTML simples de chat.

Usa a mesma busca híbrida do query.py (semântica + palavra-chave) e
permite, por pergunta, ativar um modo de "contexto amplo" — equivalente
ao `--top-k` do query.py — pra perguntas do tipo "resuma tudo sobre X".

Uso:
    python server.py
    Depois abra http://localhost:5000 no navegador.
"""

import re
from pathlib import Path

import chromadb
import ollama
from flask import Flask, request, jsonify, render_template_string

# ---------------- CONFIG (mesma do query.py) ----------------
YURIAI = Path("C:/IA")
VECTOR_DB_PATH = YURIAI / "AI" / "environment" / "vector-db"
COLLECTION_NAME = "biblioteca"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:8b"
KEEP_ALIVE = "10m"
PORT = 5000

# --------------- BUSCA ---------------
TOP_K_NORMAL = 8                 # chunks semânticos no modo normal (igual ao padrão do query.py)
TOP_K_AMPLO = 16                 # chunks semânticos no modo "contexto amplo"
KEYWORD_MATCHES_PER_TERM = 4      # chunks trazidos por CADA termo-chave encontrado literalmente
MAX_KEYWORD_TERMS = 4             # no máximo essa quantidade de termos-chave é pesquisada por pergunta
MAX_CONTEXT_CHUNKS_NORMAL = 16    # teto de chunks combinados no modo normal
MAX_CONTEXT_CHUNKS_AMPLO = 28     # teto de chunks combinados no modo amplo

# --------------- CONTEXTO / TOKENS DO MODELO ---------------
CONTEXT_WINDOW_NORMAL = 8192
CONTEXT_WINDOW_AMPLO = 16384
MAX_RESPONSE_TOKENS_NORMAL = 700
MAX_RESPONSE_TOKENS_AMPLO = 1400
# Orçamento aproximado de caracteres pros trechos recuperados (~4 chars/token em pt-BR)
CHAR_BUDGET_NORMAL = (CONTEXT_WINDOW_NORMAL - MAX_RESPONSE_TOKENS_NORMAL - 600) * 4
CHAR_BUDGET_AMPLO = (CONTEXT_WINDOW_AMPLO - MAX_RESPONSE_TOKENS_AMPLO - 600) * 4

SYSTEM_PROMPT = """Você é um assistente que responde EXCLUSIVAMENTE com base nos trechos de documentos fornecidos abaixo (a "Biblioteca" do usuário).

Regras estritas:
1. Use APENAS as informações presentes nos trechos fornecidos. Nunca use conhecimento externo ou geral.
2. Se a resposta não estiver nos trechos fornecidos, diga claramente: "Não encontrei essa informação nos documentos da Biblioteca."
3. Sempre que possível, cite de qual arquivo veio a informação (o campo 'fonte' de cada trecho).
4. Se os trechos vierem de partes diferentes ou de documentos diferentes, cruze e organize as informações num resumo coerente, em vez de listar cada trecho isoladamente.
5. Não invente, complete ou extrapole informações além do que está escrito nos trechos.
6. Responda em português.
"""

# Heurística simples pra achar nomes próprios/termos específicos na pergunta
# (mesma do query.py), usada na busca por palavra-chave.
_KEYWORD_PATTERN = re.compile(
    r"[A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:d[aeo]s?|e)?\s*[A-ZÀ-Ú][a-zà-ú]+)*"
)
# --------------------------------------------------------------

app = Flask(__name__)
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


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


def semantic_search(question: str, k: int, where):
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    results = get_collection().query(query_embeddings=[q_emb], n_results=k, where=where)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    return [_chunk_from(d, m, i) for d, m, i in zip(docs, metas, ids)]


def keyword_search(keywords, where):
    """Busca literal: qualquer chunk que contenha o termo, usando o
    operador $contains do Chroma, sem depender de similaridade semântica."""
    found = []
    for term in keywords:
        try:
            results = get_collection().get(
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


def retrieve(question: str, source_filter: str = None, broad: bool = False):
    where = {"source_type": source_filter} if source_filter else None

    top_k = TOP_K_AMPLO if broad else TOP_K_NORMAL
    max_chunks = MAX_CONTEXT_CHUNKS_AMPLO if broad else MAX_CONTEXT_CHUNKS_NORMAL
    char_budget = CHAR_BUDGET_AMPLO if broad else CHAR_BUDGET_NORMAL

    semantic_chunks = semantic_search(question, top_k, where)
    keywords = extract_keywords(question)
    keyword_chunks = keyword_search(keywords, where) if keywords else []

    # Combina mantendo prioridade pro que a busca semântica já trouxe,
    # completando com os achados literais que ainda não estavam na lista.
    seen_ids = set()
    combined = []
    for c in semantic_chunks + keyword_chunks:
        if c["id"] in seen_ids:
            continue
        seen_ids.add(c["id"])
        combined.append(c)

    combined = combined[:max_chunks]

    # Proteção contra estouro de contexto: corta os últimos (menos
    # prioritários) se passar do orçamento de caracteres.
    total = 0
    trimmed = []
    for c in combined:
        total += len(c["text"])
        if total > char_budget and trimmed:
            break
        trimmed.append(c)

    return trimmed


def ask(question: str, source_filter: str = None, broad: bool = False):
    chunks = retrieve(question, source_filter, broad=broad)
    if not chunks:
        return {"answer": "Não encontrei essa informação nos documentos indexados.", "sources": []}

    context = "\n\n".join(
        f"[Trecho {i} — origem: {c['source_type']} — arquivo: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    user_message = f"Trechos recuperados:\n\n{context}\n\nPergunta: {question}"

    context_window = CONTEXT_WINDOW_AMPLO if broad else CONTEXT_WINDOW_NORMAL
    max_response_tokens = MAX_RESPONSE_TOKENS_AMPLO if broad else MAX_RESPONSE_TOKENS_NORMAL

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        keep_alive=KEEP_ALIVE,
        options={"num_predict": max_response_tokens, "num_ctx": context_window},
    )

    answer = response["message"]["content"]
    if response.get("done_reason") == "length":
        answer += (
            "\n\n⚠️ A resposta foi cortada por atingir o limite de tokens "
            f"({max_response_tokens}). Ative o contexto amplo ou peça de forma "
            "mais específica se precisar de mais detalhe."
        )

    fontes = sorted(set(c["source"] for c in chunks))
    return {"answer": answer, "sources": fontes, "num_chunks": len(chunks)}


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    source_filter = data.get("source_filter") or None
    broad = bool(data.get("broad"))
    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400
    try:
        result = ask(question, source_filter, broad=broad)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


HTML_PAGE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Biblioteca — Consulta Local</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f7f7f8; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  #chat { display: flex; flex-direction: column; gap: 14px; margin-bottom: 140px; }
  .msg { padding: 12px 16px; border-radius: 10px; max-width: 90%; white-space: pre-wrap; line-height: 1.5; }
  .user { align-self: flex-end; background: #2563eb; color: white; }
  .bot { align-self: flex-start; background: white; border: 1px solid #ddd; }
  .sources { font-size: 0.8rem; color: #666; margin-top: 6px; }
  .loading { color: #888; font-style: italic; }
  form { position: fixed; bottom: 0; left: 0; right: 0; background: #f7f7f8; padding: 12px 16px; max-width: 800px; margin: 0 auto; }
  .row { display: flex; gap: 8px; }
  .options { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; font-size: 0.85rem; color: #444; }
  input[type=text] { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #ccc; font-size: 1rem; }
  select { padding: 12px; border-radius: 8px; border: 1px solid #ccc; }
  button { padding: 12px 20px; border-radius: 8px; border: none; background: #2563eb; color: white; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  label.toggle { display: flex; align-items: center; gap: 6px; cursor: pointer; }
</style>
</head>
<body>
<h1>📚 Consulta à Biblioteca</h1>
<div id="chat"></div>
<form id="form">
  <div class="options">
    <label class="toggle" title="Traz mais trechos e permite respostas mais longas. Use pra 'resuma tudo sobre X'.">
      <input type="checkbox" id="broadContext" />
      Contexto amplo (pra resumos e perguntas abrangentes)
    </label>
  </div>
  <div class="row">
    <select id="sourceFilter">
      <option value="">Todas as fontes</option>
      <option value="biblioteca">Só Biblioteca</option>
      <option value="affine">Só AFFiNE</option>
    </select>
    <input type="text" id="question" placeholder="Pergunte algo sobre seus documentos..." autocomplete="off" />
    <button type="submit" id="sendBtn">Enviar</button>
  </div>
</form>

<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('question');
const sendBtn = document.getElementById('sendBtn');
const sourceFilter = document.getElementById('sourceFilter');
const broadContext = document.getElementById('broadContext');

function addMessage(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  const broad = broadContext.checked;
  addMessage(question + (broad ? '  [contexto amplo]' : ''), 'user');
  input.value = '';
  sendBtn.disabled = true;
  const loadingMsg = broad
    ? 'Pensando com contexto amplo... (mais trechos, pode demorar mais)'
    : 'Pensando... (pode levar um tempo em CPU)';
  const loadingDiv = addMessage(loadingMsg, 'bot loading');

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, source_filter: sourceFilter.value, broad })
    });
    const data = await res.json();
    loadingDiv.remove();

    if (data.error) {
      addMessage('Erro: ' + data.error, 'bot');
    } else {
      const botDiv = addMessage(data.answer, 'bot');
      if (data.sources && data.sources.length) {
        const src = document.createElement('div');
        src.className = 'sources';
        src.textContent = 'Fontes (' + data.num_chunks + ' trecho(s)): ' + data.sources.join(', ');
        botDiv.appendChild(src);
      }
    }
  } catch (err) {
    loadingDiv.remove();
    addMessage('Erro de conexão: ' + err, 'bot');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Abra http://localhost:{PORT} no navegador.")
    app.run(host="127.0.0.1", port=PORT, debug=False)
