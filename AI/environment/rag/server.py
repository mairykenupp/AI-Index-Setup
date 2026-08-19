#!/usr/bin/env python3
"""
Interface web leve para consultar a Biblioteca (RAG) via Ollama.
Roda um servidor Flask local com uma página HTML simples de chat.

Uso:
    python server.py
    Depois abra http://localhost:5000 no navegador.
"""

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
TOP_K = 3
KEEP_ALIVE = "10m"
MAX_RESPONSE_TOKENS = 400
PORT = 5000

SYSTEM_PROMPT = """Você é um assistente que responde EXCLUSIVAMENTE com base nos trechos de documentos fornecidos abaixo (a "Biblioteca" do usuário).

Regras estritas:
1. Use APENAS as informações presentes nos trechos fornecidos. Nunca use conhecimento externo ou geral.
2. Se a resposta não estiver nos trechos fornecidos, diga claramente: "Não encontrei essa informação nos documentos da Biblioteca."
3. Sempre que possível, cite de qual arquivo veio a informação (o campo 'fonte' de cada trecho).
4. Não invente, complete ou extrapole informações além do que está escrito nos trechos.
5. Responda em português.
"""
# --------------------------------------------------------------

app = Flask(__name__)
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def retrieve(question: str, source_filter: str = None):
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    where = {"source_type": source_filter} if source_filter else None
    results = get_collection().query(query_embeddings=[q_emb], n_results=TOP_K, where=where)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [
        {"text": d, "source": m.get("doc_id", "desconhecido"), "source_type": m.get("source_type", "?")}
        for d, m in zip(docs, metas)
    ]


def ask(question: str, source_filter: str = None):
    chunks = retrieve(question, source_filter)
    if not chunks:
        return {"answer": "Não encontrei essa informação nos documentos indexados.", "sources": []}

    context = "\n\n".join(
        f"[Trecho {i} — origem: {c['source_type']} — arquivo: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
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

    fontes = sorted(set(c["source"] for c in chunks))
    return {"answer": response["message"]["content"], "sources": fontes}


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    source_filter = data.get("source_filter") or None
    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400
    try:
        result = ask(question, source_filter)
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
  #chat { display: flex; flex-direction: column; gap: 14px; margin-bottom: 100px; }
  .msg { padding: 12px 16px; border-radius: 10px; max-width: 90%; white-space: pre-wrap; line-height: 1.5; }
  .user { align-self: flex-end; background: #2563eb; color: white; }
  .bot { align-self: flex-start; background: white; border: 1px solid #ddd; }
  .sources { font-size: 0.8rem; color: #666; margin-top: 6px; }
  .loading { color: #888; font-style: italic; }
  form { position: fixed; bottom: 0; left: 0; right: 0; background: #f7f7f8; padding: 16px; display: flex; gap: 8px; max-width: 800px; margin: 0 auto; }
  input[type=text] { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #ccc; font-size: 1rem; }
  select { padding: 12px; border-radius: 8px; border: 1px solid #ccc; }
  button { padding: 12px 20px; border-radius: 8px; border: none; background: #2563eb; color: white; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<h1>📚 Consulta à Biblioteca</h1>
<div id="chat"></div>
<form id="form">
  <select id="sourceFilter">
    <option value="">Todas as fontes</option>
    <option value="biblioteca">Só Biblioteca</option>
    <option value="affine">Só AFFiNE</option>
  </select>
  <input type="text" id="question" placeholder="Pergunte algo sobre seus documentos..." autocomplete="off" />
  <button type="submit" id="sendBtn">Enviar</button>
</form>

<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('question');
const sendBtn = document.getElementById('sendBtn');
const sourceFilter = document.getElementById('sourceFilter');

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

  addMessage(question, 'user');
  input.value = '';
  sendBtn.disabled = true;
  const loadingDiv = addMessage('Pensando... (pode levar um tempo em CPU)', 'bot loading');

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, source_filter: sourceFilter.value })
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
        src.textContent = 'Fontes: ' + data.sources.join(', ');
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
